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
from ..config import (
    CASES,
    CO2_LEVELS,
    CONTROL,
    HOSING_LEVELS,
    STEADY_STATE_YEARS,
    available_cases,
    get_case,
)
from ..io import cases_with, load_var, variables_in
from ..plotting import TEXT_PRIMARY, TEXT_SECONDARY, grid_3x3
from ..significance import significance_mask
from ..variables import display_name, info, to_display_units

WINDOW = f"{STEADY_STATE_YEARS.start}-{STEADY_STATE_YEARS.stop}"


def _key(case) -> tuple[int, float]:
    return (case.co2, case.hosing)


def _missing_notes(var: str) -> dict[tuple[int, float], str]:
    """Distinguish a case that has not run from one that ran without this field.

    Both leave an empty panel, but they mean different things: one will fill in
    when the simulation finishes, the other needs the field requested from the
    run archive. `TREFHT`, archived only for `picontrol`, is the live example.
    """
    return {
        _key(case): f"{case.name}\nno {var} archived"
        for case in available_cases()
        if case not in cases_with(var)
    }


def absolute_page(var: str) -> plt.Figure:
    """Steady-state climatologies of every case that archives this field."""
    fields = {
        _key(case): to_display_units(steady_state(case, var), var)
        for case in cases_with(var)
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
        missing_notes=_missing_notes(var),
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

    for case in cases_with(var):
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
        missing_notes=_missing_notes(var),
    )


def pages(var: str, **kwargs) -> Iterator[plt.Figure]:
    """The page sequence for one variable: absolute, then anomaly.

    The anomaly page needs both the control and at least one perturbed case to
    archive the field. `TREFHT` has only the control, so it gets an absolute
    page alone rather than a page of empty panels.
    """
    yield absolute_page(var)
    archiving = cases_with(var)
    if CONTROL in archiving and len(archiving) > 1:
        yield anomaly_page(var, **kwargs)


def status_page(variables: list[str]) -> plt.Figure:
    """Front page: what has run, what has not, and which fields are incomplete.

    Seeing what is missing is one of the book's jobs, so it leads with that
    rather than leaving it to be inferred from empty panels later on.
    """
    fig = plt.figure(figsize=(13.5, 7.6))
    fig.suptitle("Quasi-steady-state book — coverage", fontsize=14,
                 color=TEXT_PRIMARY, x=0.06, ha="left", y=0.95)
    fig.text(0.06, 0.905, f"Steady-state window {WINDOW}   ·   built from "
             f"{len(available_cases())} of {len(CASES)} cases   ·   "
             f"{len(variables)} variables", fontsize=10, color=TEXT_SECONDARY)

    lines = ["Simulations", ""]
    for co2 in CO2_LEVELS:
        for hosing in HOSING_LEVELS:
            case = get_case(co2, hosing)
            cell = f"  {co2}xCO2, {hosing:+.1f} Sv   {case.name:<16}"
            if case.exists:
                years = load_var(case, variables[0] if variables[0] in variables_in(case)
                                 else variables_in(case)[0], years=None)["year"].values
                lines.append(f"{cell} {len(years)} yr, {years[0]}-{years[-1]}")
            else:
                lines.append(f"{cell} NOT YET RUN")
    fig.text(0.06, 0.83, "\n".join(lines), fontsize=9.5, color=TEXT_PRIMARY,
             va="top", family="monospace", linespacing=1.6)

    incomplete = {v: [c.name for c in available_cases() if c not in cases_with(v)]
                  for v in variables}
    incomplete = {v: miss for v, miss in incomplete.items() if miss}
    right = ["Fields missing from cases that have run", ""]
    if incomplete:
        for v, miss in incomplete.items():
            right.append(f"  {v:<10} absent from {', '.join(miss)}")
    else:
        right.append("  none — every field is present in every case that has run")
    right += ["", "Records dropped as partial years", "",
              "  picontrol 1850 and 4xCO2_noh 2051 are 11-month",
              "  means (January missing), so the common window",
              "  starts at 2052."]
    fig.text(0.52, 0.83, "\n".join(right), fontsize=9.5, color=TEXT_PRIMARY,
             va="top", family="monospace", linespacing=1.6)

    fig.text(0.06, 0.06, "Each field gets an absolute page followed by an anomaly page, "
             "alphabetically. A field archived only by the control gets no anomaly page.",
             fontsize=8.5, color=TEXT_SECONDARY)
    return fig


def book_pages(variables: list[str], **kwargs) -> Iterator[plt.Figure]:
    """Page sequence for a whole book, kept lazy so memory stays flat.

    Variables are always alphabetised, whatever order they were requested in, so
    a book has one predictable order and pages stay findable as fields are added.
    Each variable keeps its absolute page immediately followed by its anomaly
    page — the pairing is never split. The coverage page leads.
    """
    variables = sorted(variables)
    yield status_page(variables)
    for var in variables:
        yield from pages(var, **kwargs)
