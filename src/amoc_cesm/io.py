"""Loading and basic reduction of the annual-mean 2D CAM fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .config import FILE_SUFFIX, Case, get_case_by_name


def _as_case(case: Case | str) -> Case:
    return get_case_by_name(case) if isinstance(case, str) else case


def variables_in(case: Case | str) -> list[str]:
    """Variable names available for a case, from its file names."""
    case = _as_case(case)
    if not case.exists:
        return []
    return sorted(
        p.name[: -len(FILE_SUFFIX)]
        for p in case.path.glob(f"*{FILE_SUFFIX}")
    )


def var_path(case: Case | str, var: str) -> Path:
    return _as_case(case).path / f"{var}{FILE_SUFFIX}"


def load_var(case: Case | str, var: str, **kwargs) -> xr.DataArray:
    """Load one variable for one case as a (time, lat, lon) DataArray."""
    case = _as_case(case)
    path = var_path(case, var)
    if not path.exists():
        raise FileNotFoundError(f"{var} not available for case {case.name}: {path}")
    ds = xr.open_dataset(path, **kwargs)
    da = ds[var]
    da.attrs.setdefault("case", case.name)
    return da


def load_case(case: Case | str, variables: list[str] | None = None) -> xr.Dataset:
    """Load several variables for one case into a single Dataset."""
    case = _as_case(case)
    variables = variables if variables is not None else variables_in(case)
    return xr.merge([load_var(case, v) for v in variables], compat="override")


def area_weights(da: xr.DataArray) -> xr.DataArray:
    """cos(latitude) weights for a regular lat-lon grid."""
    return np.cos(np.deg2rad(da["lat"]))


def global_mean(da: xr.DataArray) -> xr.DataArray:
    """Area-weighted mean over lat/lon, preserving remaining dimensions."""
    return da.weighted(area_weights(da)).mean(dim=("lat", "lon"))


def climatology(da: xr.DataArray, last_n_years: int | None = None) -> xr.DataArray:
    """Time mean, optionally over only the last ``last_n_years`` years."""
    if last_n_years is not None:
        da = da.isel(time=slice(-last_n_years, None))
    return da.mean(dim="time", keep_attrs=True)
