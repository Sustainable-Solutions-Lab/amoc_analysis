"""Quasi-steady-state workflow: the last 50 years treated as an equilibrium.

Each variable yields two pages — the absolute climatology across the 3x3 design,
then the anomaly relative to `picontrol` with significance marked.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..analysis import steady_state, steady_state_anomaly
from ..config import CONTROL, STEADY_STATE_YEARS, available_cases
from ..plotting import grid_3x3
from ..significance import significance_mask
from ..variables import display_name, info, to_display_units

WINDOW = f"{STEADY_STATE_YEARS.start}-{STEADY_STATE_YEARS.stop}"


def _key(case) -> tuple[int, float]:
    return (case.co2, case.hosing)


def absolute_page(var: str) -> plt.Figure:
    """All available cases' steady-state climatologies on one shared scale."""
    fields = {
        _key(case): to_display_units(steady_state(case, var), var)
        for case in available_cases()
    }
    meta = info(var, next(iter(fields.values())))
    vmin = min(float(da.min()) for da in fields.values())
    vmax = max(float(da.max()) for da in fields.values())

    return grid_3x3(
        fields,
        title=f"{var} — {display_name(var, next(iter(fields.values())))}",
        subtitle=f"Quasi-steady-state mean, {WINDOW}",
        units=meta.units,
        cmap=meta.sequential_cmap,
        vmin=vmin,
        vmax=vmax,
    )


def anomaly_page(
    var: str,
    significance_style: str = "field",
    alpha: float = 0.05,
    false_discovery_rate: bool = True,
) -> plt.Figure:
    """Steady-state anomalies vs. the control, with significance contoured.

    The control cell is zero by construction and is labeled as the reference
    rather than given its own color scale — its absolute field is on page 1.
    """
    fields: dict[tuple[int, float], xr.DataArray] = {}
    masks: dict[tuple[int, float], xr.DataArray] = {}

    for case in available_cases():
        # check_range=False: these are differences, not absolute values.
        fields[_key(case)] = to_display_units(
            steady_state_anomaly(case, var), var, check_range=False
        )
        if case.name != CONTROL.name:
            masks[_key(case)] = significance_mask(
                case, var, alpha=alpha, false_discovery_rate=false_discovery_rate
            )

    meta = info(var, next(iter(fields.values())))

    # Scale to the 98th percentile of |anomaly| rather than the maximum. A few
    # extreme polar grid cells otherwise set the range and wash out the pattern
    # everywhere else. Nothing is hidden: the colorbar carries extend arrows
    # showing that values run past both ends. The reference panel is excluded
    # because its identical zeros would drag the percentile down.
    perturbed = [da for key, da in fields.items() if key != _key(CONTROL)]
    span = float(np.percentile(np.abs(np.concatenate([da.values.ravel() for da in perturbed])), 98))

    test = f"Welch t-test, p < {alpha}" + (", FDR controlled" if false_discovery_rate else "")
    marking = (
        "contours enclose significant regions"
        if significance_style == "outline"
        else "contours at colorbar values, drawn only where significant"
    )

    return grid_3x3(
        fields,
        title=f"{var} — anomaly vs. {CONTROL.name}",
        subtitle=f"Quasi-steady-state mean, {WINDOW}   ·   {test} ({marking})",
        units=f"Δ {meta.units}",
        cmap=meta.diverging_cmap,
        vmin=-span,
        vmax=span,
        extend="both",
        significance=masks,
        significance_style=significance_style,
        annotations={_key(CONTROL): "reference (zero by construction)"},
    )


def pages(var: str, **kwargs) -> Iterator[plt.Figure]:
    """The full page sequence for one variable."""
    yield absolute_page(var)
    yield anomaly_page(var, **kwargs)


def book_pages(variables: list[str], **kwargs) -> Iterator[plt.Figure]:
    """Page sequence for a whole book, kept lazy so memory stays flat.

    Variables are always alphabetised, whatever order they were requested in, so
    a book has one predictable order and pages stay findable as fields are added.
    Each variable keeps its absolute page immediately followed by its anomaly
    page — the pairing is never split.
    """
    for var in sorted(variables):
        yield from pages(var, **kwargs)
