"""Loading and basic reduction of the annual-mean 2D CAM fields.

All cases share the same 365-day calendar and the same annual time stamps: the
perturbation runs cover model years 2051-2150, which are also the last 100 years
of the 301-year `picontrol` record. Loading indexes by integer calendar `year`
by default, so cases align directly under xarray arithmetic.

Note that year alignment is bookkeeping, not pairing: the perturbation runs are
branches, so their weather is uncorrelated with `picontrol` in the same year.
Difference time means, not individual years.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .config import ANALYSIS_YEARS, Case, get_case_by_name

#: Annual means shorter than this many days are partial years (see below).
FULL_YEAR_DAYS = 360


def _as_case(case: Case | str) -> Case:
    return get_case_by_name(case) if isinstance(case, str) else case


def variables_in(case: Case | str) -> list[str]:
    """Variable names available for a case, from its file names."""
    case = _as_case(case)
    suffix = case.suffix
    return sorted(p.name[: -len(suffix)] for p in case.path.glob(f"*{suffix}"))


def var_path(case: Case | str, var: str) -> Path:
    case = _as_case(case)
    return case.path / f"{var}{case.suffix}"


def all_variables() -> list[str]:
    """Every variable archived by at least one case that has run.

    The union, not the intersection: `TREFHT` exists only in `picontrol`, and a
    field present in some runs is still worth plotting for those runs.
    """
    from .config import available_cases

    return sorted(set().union(*(set(variables_in(c)) for c in available_cases())))


def cases_with(var: str) -> list[Case]:
    """Available cases that archive this variable."""
    from .config import available_cases

    return [c for c in available_cases() if var in variables_in(c)]


def _year_span_days(ds: xr.Dataset) -> np.ndarray:
    """Length in days of each annual-mean interval, from time_bnds."""
    bnds = ds["time_bnds"].values
    return np.asarray(bnds[:, 1] - bnds[:, 0], dtype=float)


def load_var(
    case: Case | str,
    var: str,
    years: slice | None = ANALYSIS_YEARS,
    index_by: str = "year",
    drop_partial: bool = True,
    **kwargs,
) -> xr.DataArray:
    """Load one variable for one case.

    Parameters
    ----------
    years
        Calendar-year range to keep, defaulting to `ANALYSIS_YEARS` (2052-2150),
        the window common to every case. This also trims `picontrol` from its
        full 1850-2150 record. Pass ``None`` for everything on disk — e.g. to
        use the long control for internal-variability statistics.
    index_by
        ``"year"`` (default) replaces the time dimension with an integer
        ``year`` coordinate so cases align across the ensemble; ``"time"``
        keeps the original cftime axis.
    drop_partial
        Drop records whose ``time_bnds`` span less than ``FULL_YEAR_DAYS``.
        Two such records exist: `picontrol` year 1850 and `4xCO2_noh` year 2051
        are both 11-month means (January missing), so their values are biased by
        the omitted month.
    """
    case = _as_case(case)
    path = var_path(case, var)
    ds = xr.open_dataset(path, **kwargs)
    spans = _year_span_days(xr.open_dataset(path, decode_times=False))

    da = ds[var]
    file_years = da["time"].dt.year.values
    da = da.assign_coords(year=("time", file_years))

    if drop_partial:
        keep = spans >= FULL_YEAR_DAYS
        if not keep.all():
            dropped = file_years[~keep]
            da = da.isel(time=np.flatnonzero(keep))
            da.attrs["dropped_partial_years"] = ", ".join(str(y) for y in dropped)

    da = da.swap_dims({"time": "year"})
    if years is not None:
        da = da.sel(year=years)
    if index_by == "time":
        da = da.swap_dims({"year": "time"})

    da.attrs["case"] = case.name
    da.attrs["co2"] = case.co2
    da.attrs["hosing_Sv"] = case.hosing
    return da


def load_case(
    case: Case | str,
    variables: list[str] | None = None,
    **kwargs,
) -> xr.Dataset:
    """Load several variables for one case into a single Dataset."""
    case = _as_case(case)
    variables = variables if variables is not None else variables_in(case)
    # Strict on purpose: every file for a case must be on the same grid and the
    # same years. compat="equals" fails on conflicting coordinate values,
    # join="exact" fails on misaligned indexes instead of NaN-padding them.
    return xr.merge(
        [load_var(case, v, **kwargs) for v in variables],
        compat="equals",
        join="exact",
    )


def load_ensemble(
    var: str,
    cases: list[Case | str] | None = None,
    years: slice | None = ANALYSIS_YEARS,
    **kwargs,
) -> xr.DataArray:
    """Stack one variable across cases along a new ``case`` dimension.

    Cases are aligned on `year` over ``years`` (the common analysis window by
    default). Widening it past that window leaves NaN where a case has no data.
    """
    from .config import available_cases

    case_objs = [_as_case(c) for c in (cases if cases is not None else available_cases())]
    das = [
        load_var(case, var, years=years, index_by="year", **kwargs).drop_vars("time")
        for case in case_objs
    ]
    out = xr.concat(das, dim=xr.DataArray([c.name for c in case_objs], dims="case", name="case"))
    out = out.assign_coords(
        co2=("case", [c.co2 for c in case_objs]),
        hosing=("case", [c.hosing for c in case_objs]),
    )
    return out


def common_years(*das: xr.DataArray) -> np.ndarray:
    """Calendar years present in every one of the given arrays."""
    sets = [set(np.asarray(d["year"].values).tolist()) for d in das]
    return np.array(sorted(set.intersection(*sets)))


def area_weights(da: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights for a regular lat-lon grid."""
    return np.cos(np.deg2rad(da["lat"]))


def global_mean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over lat/lon, preserving remaining dimensions."""
    return da.weighted(area_weights(da)).mean(dim=("lat", "lon"), keep_attrs=True)


def climatology(
    da: xr.DataArray,
    years: slice | None = None,
    last_n_years: int | None = None,
) -> xr.DataArray:
    """Mean over the year (or time) dimension, optionally over a subset.

    ``years`` selects a calendar-year range (requires year indexing);
    ``last_n_years`` takes the final N records instead.
    """
    dim = "year" if "year" in da.dims else "time"
    if years is not None:
        da = da.sel({dim: years})
    if last_n_years is not None:
        da = da.isel({dim: slice(-last_n_years, None)})
    return da.mean(dim=dim, keep_attrs=True)
