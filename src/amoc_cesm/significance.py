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

    # Double precision for the moment calculations. The files are float32, and
    # for a field like SOLIN — mean ~340 W/m2, interannual spread ~1e-4 —
    # subtracting the mean in float32 loses the spread entirely to cancellation.
    # In float64 the same arithmetic is exact enough to be meaningless-but-safe:
    # SOLIN is bit-identical across cases, so the test correctly returns p = 1.
    values_a = a.values.astype(np.float64)
    values_b = b.values.astype(np.float64)

    # Some grid points are exactly constant in time and so have no distribution
    # to test: polar-night SOLIN, ocean cells in SNOWHICE/SNOWHLND, the tropics
    # in PRECSC. Testing them produces NaN plus a scipy precision warning, so
    # they are decided directly instead — a difference between two constants is
    # deterministic, and no difference at all is not a finding.
    testable = ~((values_a.std(axis=0) == 0) & (values_b.std(axis=0) == 0))

    p = np.full(testable.shape, np.nan)
    if testable.any():
        _, p[testable] = stats.ttest_ind(
            values_a[:, testable], values_b[:, testable], axis=0, equal_var=False
        )

    if false_discovery_rate:
        keep = _benjamini_hochberg(p, alpha, testable)
    else:
        keep = p < alpha  # NaN compares False, so untestable points stay unmarked
    keep |= ~testable & (values_a[0] != values_b[0])

    mask = xr.DataArray(keep, coords={"lat": a["lat"], "lon": a["lon"]}, dims=("lat", "lon"))
    mask.attrs["test"] = "Welch t-test, annual values"
    mask.attrs["alpha"] = alpha
    mask.attrs["fdr"] = false_discovery_rate
    mask.attrs["untestable_points"] = int((~testable).sum())
    return mask


def _benjamini_hochberg(p: np.ndarray, alpha: float, testable: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg field significance; returns the rejection mask.

    The denominator counts only points that were actually tested. Including the
    untestable ones would make the threshold far too strict — for SNOWHICE they
    are 74% of the grid, which alone costs ~60 real detections.
    """
    valid = p[testable]
    order = np.argsort(valid)
    ranked = valid[order]
    below = np.flatnonzero(ranked <= alpha * (np.arange(1, ranked.size + 1) / ranked.size))

    keep_valid = np.zeros(valid.size, dtype=bool)
    if below.size:
        keep_valid[order[: below[-1] + 1]] = True

    keep = np.zeros(p.shape, dtype=bool)
    keep[testable] = keep_valid
    return keep
