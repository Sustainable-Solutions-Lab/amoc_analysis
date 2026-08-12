"""Whether a steady-state difference is distinguishable from internal variability.

The test is a Welch t-test at each grid point between the annual values of a
case and those of the reference over the same window (50 years each by default).
Welch rather than Student because hosing changes the variance, not just the mean.

Annual means of these fields are close to serially independent, so the 50 years
are treated as 50 samples. That assumption is worth revisiting for the slowly
varying ocean-influenced fields if this ever moves beyond atmospheric 2D output.

Testing ~14k grid points at alpha = 0.05 yields ~700 false positives by
construction, so `false_discovery_rate=True` applies the Benjamini-Hochberg
control that Wilks (2016) recommends for exactly this situation.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats

from .config import CONTROL, STEADY_STATE_YEARS, Case
from .io import load_var


def significance_mask(
    case: Case | str,
    var: str,
    reference: Case | str = CONTROL,
    years: slice = STEADY_STATE_YEARS,
    alpha: float = 0.05,
    false_discovery_rate: bool = False,
) -> xr.DataArray:
    """Boolean (lat, lon) mask: True where the difference is significant."""
    a = load_var(case, var, years=years)
    b = load_var(reference, var, years=years)
    _, p = stats.ttest_ind(a.values, b.values, axis=0, equal_var=False)

    if false_discovery_rate:
        keep = _benjamini_hochberg(p, alpha)
    else:
        keep = p < alpha

    mask = xr.DataArray(keep, coords={"lat": a["lat"], "lon": a["lon"]}, dims=("lat", "lon"))
    mask.attrs["test"] = "Welch t-test, annual values"
    mask.attrs["alpha"] = alpha
    mask.attrs["fdr"] = false_discovery_rate
    return mask


def _benjamini_hochberg(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg field significance; returns the rejection mask."""
    flat = p.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    threshold_rank = np.flatnonzero(ranked <= alpha * (np.arange(1, ranked.size + 1) / ranked.size))
    keep = np.zeros(flat.size, dtype=bool)
    if threshold_rank.size:
        keep[order[: threshold_rank[-1] + 1]] = True
    return keep.reshape(p.shape)
