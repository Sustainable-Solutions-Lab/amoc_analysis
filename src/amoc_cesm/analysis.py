"""Differencing cases, in the two modes this project uses.

**Transient**: subtract year for year against the control over the common
analysis window. The runs are branches, so same-year weather is uncorrelated and
this cancels no noise — but it does remove any slow drift shared with the
control, which is why `picontrol` is trimmed to the same years rather than
averaged over its full 1850-2150 record.

**Quasi-steady-state**: average the last 50 years of each simulation first, then
difference. This is the mode for the 3x3 factorial comparisons.
"""

from __future__ import annotations

import xarray as xr

from .config import CONTROL, STEADY_STATE_YEARS, Case
from .io import climatology, load_var


def transient_anomaly(
    case: Case | str,
    var: str,
    reference: Case | str = CONTROL,
    **kwargs,
) -> xr.DataArray:
    """Year-by-year difference of ``case`` minus ``reference``.

    Both are loaded over the common analysis window and aligned on `year`, so
    the result carries only years present in both.
    """
    da = load_var(case, var, **kwargs)
    ref = load_var(reference, var, **kwargs)
    out = da - ref
    out.attrs = dict(da.attrs)
    out.attrs["reference_case"] = ref.attrs["case"]
    out.attrs["anomaly_mode"] = "transient (year-by-year)"
    out.name = var
    return out


def steady_state(
    case: Case | str,
    var: str,
    years: slice = STEADY_STATE_YEARS,
    **kwargs,
) -> xr.DataArray:
    """Time mean over the quasi-steady-state window.

    Asserts the window is fully present. The +0.3 Sv runs end at 2150, 2140 and
    2135, so a window reaching past the shortest of them would average a
    different number of years in different panels of the same page and say
    nothing about it — a difference in run length showing up as a difference in
    climate.
    """
    da = load_var(case, var, **kwargs)
    present = da.sel(year=years)["year"].values
    expected = years.stop - years.start + 1
    assert len(present) == expected, (
        f"{_as_name(case)}: {var} has {len(present)} of the {expected} years in "
        f"{years.start}-{years.stop}; the steady-state window is not fully covered"
    )
    out = climatology(da, years=years)
    out.attrs["steady_state_years"] = f"{years.start}-{years.stop}"
    return out


def _as_name(case: Case | str) -> str:
    return case if isinstance(case, str) else case.name


def steady_state_anomaly(
    case: Case | str,
    var: str,
    reference: Case | str = CONTROL,
    years: slice = STEADY_STATE_YEARS,
    **kwargs,
) -> xr.DataArray:
    """Difference of quasi-steady-state means, ``case`` minus ``reference``."""
    da = steady_state(case, var, years=years, **kwargs)
    ref = steady_state(reference, var, years=years, **kwargs)
    out = da - ref
    out.attrs = dict(da.attrs)
    out.attrs["reference_case"] = ref.attrs["case"]
    out.attrs["anomaly_mode"] = f"steady state ({years.start}-{years.stop})"
    out.name = var
    return out
