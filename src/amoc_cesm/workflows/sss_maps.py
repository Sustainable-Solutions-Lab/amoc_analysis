"""End-of-run maps of sea-surface salinity across the 3x3 design.

Each simulation is averaged over its own **last ten years**, not over a window
shared by all of them. The ocean cases do not have a common end year --
`4xCO2_poshos` stops at 2120, `2xCO2_poshos` at 2130, `picontrol` runs on to
2155 -- so a fixed window would either throw away the end of the long runs or
reach past the end of the short ones. The cost is that panels are not strictly
contemporaneous, which is why every panel prints its own averaging window and
the full span of the run it came from.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..config import CO2_LEVELS, HOSING_LEVELS
from ..plotting import (
    LAND_FACE,
    MAP_PROJECTION,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    _draw_map,
    grid_3x3,
    hosing_label,
    tick_levels,
)
from ..regrid import ALTERNATE_CASES, load_sss

VAR = "SALT"
UNITS = "g kg$^{-1}$"
N_YEARS = 10

#: Which case fills each cell of the 3x3 design. The +0.3 Sv column uses the
#: canonical NAHosMIP_FIX runs; see `ALTERNATE_CASES` for the yr200 twins.
CASE_GRID: dict[tuple[int, float], str] = {
    (1, -0.3): "1xCO2_neghos", (1, 0.0): "picontrol",  (1, +0.3): "1xCO2_poshos",
    (2, -0.3): "2xCO2_neghos", (2, 0.0): "2xCO2_noh",  (2, +0.3): "2xCO2_poshos",
    (4, -0.3): "4xCO2_neghos", (4, 0.0): "4xCO2_noh",  (4, +0.3): "4xCO2_poshos",
}

REFERENCE = "picontrol"


def end_of_run(case: str, n_years: int = N_YEARS) -> xr.DataArray:
    """Mean of the last ``n_years`` of a case, carrying its window in attrs."""
    ds = load_sss(case, freq="ann")
    years = ds["year"].values
    window = years[-n_years:]
    out = ds[VAR].sel(year=window).mean("year", keep_attrs=True)
    out.attrs.update(
        case=case,
        window_start=int(window[0]), window_end=int(window[-1]),
        run_start=int(years[0]), run_end=int(years[-1]), run_length=len(years),
        ocean_area=ds["ocean_area"],
    )
    return out


def _ocean_mean(da: xr.DataArray) -> float:
    """Area-weighted mean over ocean only, using the regridder's cell areas."""
    return float(da.weighted(da.attrs["ocean_area"].fillna(0.0)).mean(("lat", "lon")))


def _footnote(da: xr.DataArray) -> str:
    a = da.attrs
    return (f"{a['window_start']}–{a['window_end']}  "
            f"of {a['run_start']}–{a['run_end']} ({a['run_length']} yr)")


def _symmetric_limit(fields, percentile: float = 98.0) -> float:
    """98th percentile of |value|, so a few extreme cells don't set the scale."""
    pooled = np.concatenate([np.abs(da.values[np.isfinite(da.values)]) for da in fields])
    return float(np.percentile(pooled, percentile))


def _percentile_limits(fields, low: float = 1.0, high: float = 99.0):
    pooled = np.concatenate([da.values[np.isfinite(da.values)] for da in fields])
    return float(np.percentile(pooled, low)), float(np.percentile(pooled, high))


def absolute_page(n_years: int = N_YEARS) -> plt.Figure:
    """End-of-run SSS in every cell of the design."""
    fields = {key: end_of_run(case, n_years) for key, case in CASE_GRID.items()}
    vmin, vmax = _percentile_limits(fields.values())

    return grid_3x3(
        fields,
        title="Sea-surface salinity at the end of each simulation",
        subtitle=(f"Mean of the last {n_years} years of each run, on the CAM 144×96 grid. "
                  "Runs end at different years — each panel gives its own window."),
        units=UNITS,
        cmap="viridis",
        vmin=vmin, vmax=vmax,
        # Zero is not a meaningful isoline for salinity, but it is also never
        # reached, so keeping it costs nothing and the contours stay on ticks.
        drop_zero_contour=False,
        footnotes={k: _footnote(v) for k, v in fields.items()},
        mean_fn=_ocean_mean,
        nan_face=LAND_FACE,
        extend="both",
        # Per-panel windows sit under each map, so the rows need more air
        # between them than a page whose panels share one window.
        figsize=(13.5, 8.3), bottom=0.145, hspace=0.34,
    )


def anomaly_page(n_years: int = N_YEARS) -> plt.Figure:
    """End-of-run SSS minus the end-of-run control.

    Both sides are end-of-run means, so this differences equilibrated states,
    not the same calendar years. `picontrol` drifts, and its own last decade is
    the fairest single baseline for runs that end at four different years.
    """
    ends = {key: end_of_run(case, n_years) for key, case in CASE_GRID.items()}
    ref = end_of_run(REFERENCE, n_years)

    fields = {}
    for key, da in ends.items():
        diff = da - ref
        diff.attrs = dict(da.attrs)
        fields[key] = diff

    limit = _symmetric_limit(fields.values())
    annotations = {
        key: "reference (zero by construction)"
        for key, case in CASE_GRID.items() if case == REFERENCE
    }

    return grid_3x3(
        fields,
        title="Sea-surface salinity anomaly at the end of each simulation",
        subtitle=(f"Last {n_years} years of each run minus the last {n_years} years of "
                  f"{REFERENCE} ({ref.attrs['window_start']}–{ref.attrs['window_end']}). "
                  f"Color limits are the 98th percentile of |anomaly|."),
        units=UNITS,
        cmap="RdBu_r",
        vmin=-limit, vmax=limit,
        annotations=annotations,
        footnotes={k: _footnote(v) for k, v in fields.items()},
        mean_fn=_ocean_mean,
        nan_face=LAND_FACE,
        extend="both",
        figsize=(13.5, 8.3), bottom=0.145, hspace=0.34,
    )


def hosing_variants_page(n_years: int = N_YEARS) -> plt.Figure:
    """The two deliveries of each +0.3 Sv run, side by side and differenced.

    Rows are CO2 level; columns are the canonical `NAHosMIP_FIX` run, its
    `yr200` twin, and FIX minus yr200. The two are separate integrations that
    diverge from the first month, so this page is the record of how much that
    choice matters. Their end-of-run windows differ by decades, which the
    footnotes state; the difference column is therefore between two different
    points in time as well as between two runs.
    """
    fig = plt.figure(figsize=(13.5, 8.3))
    gs = fig.add_gridspec(
        3, 3, left=0.055, right=0.975, top=0.885, bottom=0.145, wspace=0.09, hspace=0.34,
    )

    fix = {co2: end_of_run(CASE_GRID[(co2, +0.3)], n_years) for co2 in CO2_LEVELS}
    alt = {co2: end_of_run(name, n_years)
           for co2, name in zip(CO2_LEVELS, ALTERNATE_CASES)}
    diff, diff_notes = {}, {}
    for co2 in CO2_LEVELS:
        d = fix[co2] - alt[co2]
        d.attrs = dict(fix[co2].attrs)
        diff[co2] = d
        # The two runs end decades apart, so this differences two different
        # points in time as well as two integrations. Say both windows.
        a, b = fix[co2].attrs, alt[co2].attrs
        diff_notes[co2] = (f"{a['window_start']}–{a['window_end']}  minus  "
                           f"{b['window_start']}–{b['window_end']}")

    vmin, vmax = _percentile_limits(list(fix.values()) + list(alt.values()))
    limit = _symmetric_limit(diff.values())
    abs_ticks = tick_levels(vmin, vmax)
    diff_ticks = tick_levels(-limit, limit)

    columns = [
        ("+0.3 Sv, NAHosMIP_FIX (canonical)", fix, "viridis", vmin, vmax, abs_ticks, None),
        ("+0.3 Sv, yr200", alt, "viridis", vmin, vmax, abs_ticks, None),
        ("FIX − yr200", diff, "RdBu_r", -limit, limit,
         diff_ticks[diff_ticks != 0], diff_notes),
    ]
    meshes = {}
    for c, (heading, fields, cmap, lo, hi, levels, notes) in enumerate(columns):
        for r, co2 in enumerate(CO2_LEVELS):
            ax = fig.add_subplot(gs[r, c], projection=MAP_PROJECTION)
            note = notes[co2] if notes is not None else _footnote(fields[co2])
            meshes[c] = _draw_map(
                ax, fields[co2], cmap, lo, hi, None, "field", levels,
                mean_fn=_ocean_mean, footnote=note, nan_face=LAND_FACE,
            )
            if r == 0:
                ax.set_title(heading, fontsize=10, color=TEXT_PRIMARY, pad=8)
            if c == 0:
                ax.text(-0.06, 0.5, f"{co2}$\\times$CO$_2$", transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=10,
                        color=TEXT_PRIMARY)

    fig.canvas.draw()
    _add_colorbar(fig, meshes[0], columns[0][5], (0, 1), UNITS, "both")
    _add_colorbar(fig, meshes[2], columns[2][5], (2, 2), UNITS, "both")

    fig.suptitle("The +0.3 Sv hosing runs were delivered twice",
                 fontsize=13, color=TEXT_PRIMARY, x=0.055, ha="left", y=0.965)
    fig.text(0.055, 0.925,
             f"Mean of the last {n_years} years of each run. These are separate "
             "integrations, not one run truncated two ways.",
             fontsize=9, color=TEXT_SECONDARY, ha="left")
    return fig


def _add_colorbar(fig, mesh, ticks, span: tuple[int, int], units: str, extend: str) -> None:
    """A horizontal colorbar under the columns it describes."""
    axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
    boxes = [ax.get_position() for ax in axes]
    left = min(b.x0 for b, ax in zip(boxes, axes)
               if ax.get_subplotspec().colspan.start == span[0])
    right = max(b.x1 for b, ax in zip(boxes, axes)
                if ax.get_subplotspec().colspan.start == span[1])
    cax = fig.add_axes([left, 0.055, right - left, 0.018])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend, ticks=ticks)
    cbar.set_label(units, fontsize=8.5, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=7.5, colors=TEXT_SECONDARY, length=2)
    cbar.outline.set_edgecolor("#c9c8c3")


def pages(n_years: int = N_YEARS) -> Iterator[plt.Figure]:
    yield absolute_page(n_years)
    yield anomaly_page(n_years)
    yield hosing_variants_page(n_years)
