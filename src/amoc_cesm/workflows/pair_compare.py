"""A four-case comparison built around AMOC state rather than hosing level.

Hosing is the knob; the AMOC is what it turns. Adding freshwater (+0.3 Sv)
weakens or shuts the overturning down, removing it (-0.3 Sv) strengthens it, and
CO2 weakens it too — so the *same* hosing level means a different AMOC state at
different CO2. This page pairs the runs by the state they end up in instead:

    row 1xCO2   picontrol (0 Sv)       vs  1xCO2_hosing_FIX (+0.3 Sv)
    row 4xCO2   4xCO2_neghos (-0.3)    vs  4xCO2_noh (0 Sv)

Column 1 is the vigorous-AMOC member of each pair, column 2 the shut-down
member. At 1xCO2 no hosing already gives a vigorous AMOC and it takes +0.3 Sv to
shut it down; at 4xCO2 the warming has already weakened it, so no hosing is the
shut-down case and it takes -0.3 Sv to keep it vigorous. Both rows are still a
0.3 Sv contrast — just placed differently on the hosing axis.

That makes the two axes of the page mean what they say:

    column 3   vigorous minus shut down — the effect of the AMOC, at fixed CO2
    row 3      4xCO2 minus 1xCO2, column by column — the effect of CO2, at
               *matched* AMOC state, which is the comparison the design is for

The bottom-right cell would be the interaction of the two, and is given to the
colorbars instead: it is a difference of differences over pairs that sit at
different points on the hosing axis, so it would not mean what its position on
the grid implies.

Four panels are absolute fields and share one sequential scale; four are
differences and share one diverging scale. Both scales span every panel they
serve, so panels compare directly by eye.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..analysis import steady_state, steady_state_anomaly
from ..config import STEADY_STATE_YEARS, get_case_by_name
from ..io import cases_with, global_mean
from ..plotting import (
    MAP_PROJECTION,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _draw_map,
    tick_levels,
)
from ..significance import significance_mask
from ..variables import display_name, info, to_display_units

WINDOW = f"{STEADY_STATE_YEARS.start}-{STEADY_STATE_YEARS.stop}"

#: The vigorous-AMOC and shut-down-AMOC member of each 0.3 Sv pair.
PAIRS: dict[int, dict[str, str]] = {
    1: {"vigorous": "picontrol",    "shutdown": "1xCO2_hosing_FIX"},
    4: {"vigorous": "4xCO2_neghos", "shutdown": "4xCO2_noh"},
}
ROWS = (1, 4)
SIDES = ("vigorous", "shutdown")
COLUMN_TITLES = ("vigorous AMOC", "AMOC shut down", "vigorous − shut down")


def _hosing(name: str) -> str:
    case = get_case_by_name(name)
    return "0 Sv" if case.hosing == 0 else f"{case.hosing:+.1f} Sv"


def _label(name: str) -> str:
    return f"{name}  ({_hosing(name)})"


def required_cases() -> list[str]:
    return [name for row in PAIRS.values() for name in row.values()]


def _fields(var: str, alpha: float, false_discovery_rate: bool) -> tuple[dict, dict, dict]:
    """Absolute panels, difference panels, and significance masks for one field."""
    absolute: dict[tuple[int, int], xr.DataArray] = {}
    difference: dict[tuple[int, int], xr.DataArray] = {}
    masks: dict[tuple[int, int], xr.DataArray] = {}
    notes: dict[tuple[int, int], str] = {}

    for r, co2 in enumerate(ROWS):
        for c, side in enumerate(SIDES):
            name = PAIRS[co2][side]
            absolute[(r, c)] = to_display_units(steady_state(name, var), var)
            notes[(r, c)] = _label(name)

        # Column 3: the AMOC effect at this CO2 level.
        strong, weak = PAIRS[co2]["vigorous"], PAIRS[co2]["shutdown"]
        difference[(r, 2)] = to_display_units(
            steady_state_anomaly(strong, var, reference=weak), var, check_range=False
        )
        masks[(r, 2)] = significance_mask(
            strong, var, reference=weak, alpha=alpha,
            false_discovery_rate=false_discovery_rate,
        )
        notes[(r, 2)] = f"{_hosing(strong)} − {_hosing(weak)}"

    # Row 3: the CO2 difference at matched AMOC state, column by column.
    for c, side in enumerate(SIDES):
        hi, lo = PAIRS[4][side], PAIRS[1][side]
        difference[(2, c)] = to_display_units(
            steady_state_anomaly(hi, var, reference=lo), var, check_range=False
        )
        masks[(2, c)] = significance_mask(
            hi, var, reference=lo, alpha=alpha,
            false_discovery_rate=false_discovery_rate,
        )
        notes[(2, c)] = f"{hi} − {lo}"

    return absolute, difference, masks, notes


def _limits(fields: dict, symmetric: bool, percentile: float | None, pad: float = 0.04):
    pooled = np.concatenate([da.values.ravel() for da in fields.values()])
    pooled = pooled[np.isfinite(pooled)]
    if symmetric:
        mag = np.abs(pooled)
        limit = float(mag.max() if percentile is None else np.percentile(mag, percentile))
        limit *= 1.0 + pad
        return -limit, limit
    lo, hi = float(pooled.min()), float(pooled.max())
    if percentile is not None:
        lo = float(np.percentile(pooled, 100 - percentile))
        hi = float(np.percentile(pooled, percentile))
    margin = pad * (hi - lo)
    return lo - margin, hi + margin


def page(
    var: str,
    significance_style: str = "field",
    alpha: float = 0.05,
    false_discovery_rate: bool = True,
    color_percentile: float | None = None,
) -> plt.Figure:
    """The 3x3 comparison page for one variable."""
    absolute, difference, masks, notes = _fields(var, alpha, false_discovery_rate)
    meta = info(var, next(iter(absolute.values())))

    vmin, vmax = _limits(absolute, symmetric=False, percentile=color_percentile)
    dmin, dmax = _limits(difference, symmetric=True, percentile=color_percentile)
    value_ticks = tick_levels(vmin, vmax)
    diff_ticks = tick_levels(dmin, dmax)
    diff_ticks = diff_ticks[diff_ticks != 0]
    extend = "both" if color_percentile is not None else "neither"

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(
        3, 3, left=0.055, right=0.975, top=0.845, bottom=0.075,
        wspace=0.09, hspace=0.32,
    )

    value_mesh = diff_mesh = None
    for r in range(3):
        for c in range(3):
            if (r, c) == (2, 2):
                continue
            ax = fig.add_subplot(gs[r, c], projection=MAP_PROJECTION)
            if (r, c) in absolute:
                value_mesh = _draw_map(
                    ax, absolute[(r, c)], meta.sequential_cmap, vmin, vmax,
                    None, significance_style, value_ticks, footnote=notes[(r, c)],
                )
            else:
                diff_mesh = _draw_map(
                    ax, difference[(r, c)], meta.diverging_cmap, dmin, dmax,
                    masks[(r, c)], significance_style, diff_ticks,
                    footnote=notes[(r, c)],
                )
            if r == 0:
                ax.set_title(COLUMN_TITLES[c], fontsize=10, color=TEXT_PRIMARY, pad=8)
            if c == 0:
                row_label = (f"{ROWS[r]}$\\times$CO$_2$" if r < 2
                             else f"{ROWS[1]}$\\times$ − {ROWS[0]}$\\times$CO$_2$")
                ax.text(-0.07, 0.5, row_label, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=10, color=TEXT_PRIMARY)

    fig.canvas.draw()
    _legend_cell(fig, gs, value_mesh, value_ticks, meta.units, extend,
                 diff_mesh, diff_ticks, f"Δ {meta.units}")

    fig.suptitle(f"{var} — {display_name(var, next(iter(absolute.values())))}",
                 fontsize=13, color=TEXT_PRIMARY, x=0.055, ha="left", y=0.975)
    test = f"Welch t-test, p < {alpha}" + (", FDR controlled" if false_discovery_rate else "")
    fig.text(0.055, 0.943,
             "Rows pair a vigorous AMOC against a shut-down one, 0.3 Sv apart in "
             "hosing — so column 3 is the AMOC effect and row 3 the CO₂ effect at "
             "matched AMOC state.", fontsize=9, color=TEXT_SECONDARY, ha="left")
    fig.text(0.055, 0.918,
             f"Quasi-steady-state mean, {WINDOW}   ·   differences: {test}",
             fontsize=9, color=TEXT_SECONDARY, ha="left")
    return fig


def _legend_cell(fig, gs, value_mesh, value_ticks, value_units, extend,
                 diff_mesh, diff_ticks, diff_units) -> None:
    """Both colorbars in the vacant bottom-right cell."""
    box = gs[2, 2].get_position(fig)
    width = box.width * 0.86
    left = box.x0 + (box.width - width) / 2

    for i, (mesh, ticks, title) in enumerate((
        (value_mesh, value_ticks, f"values — rows 1-2, columns 1-2   [{value_units}]"),
        (diff_mesh, diff_ticks, f"differences — column 3 and row 3   [{diff_units}]"),
    )):
        y = box.y0 + box.height * (0.66 - 0.38 * i)
        cax = fig.add_axes([left, y, width, 0.016])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                            extend=extend, ticks=ticks)
        cbar.ax.tick_params(labelsize=7.5, colors=TEXT_SECONDARY, length=2)
        cbar.outline.set_edgecolor("#c9c8c3")
        fig.text(left, y + 0.028, title, fontsize=8, color=TEXT_SECONDARY, ha="left")


def book_pages(variables: list[str], **kwargs) -> Iterator[plt.Figure]:
    """One page per variable, alphabetised."""
    for var in sorted(variables):
        missing = [n for n in required_cases() if n not in
                   {c.name for c in cases_with(var)}]
        if missing:
            print(f"({var}: missing {', '.join(missing)})", end=" ", flush=True)
            continue
        print(var, end=" ", flush=True)
        yield page(var, **kwargs)
    print()
