"""Quasi-steady-state workflow: the last 50 years treated as an equilibrium.

Each variable yields four pages: the absolute climatology across the 3x3 design,
then three differences — the total response against `picontrol`, and the two
axes of the design taken separately, warming at fixed hosing and the AMOC at
fixed CO2. The last two are the ones that separate the effects the project set
out to distinguish.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..analysis import steady_state, steady_state_anomaly
from ..config import (
    CASES,
    Case,
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
        # No mask, so contours cover the whole panel: on an absolute field a
        # contour is an isoline, not a claim about significance. Zero is kept
        # for the same reason — SHFLX genuinely crosses zero.
        drop_zero_contour=False,
        missing_notes=_missing_notes(var),
    )


def _difference_page(
    var: str,
    reference_for: Callable[[Case], Case],
    title: str,
    describe: str,
    label: str,
    significance_style: str = "field",
    alpha: float = 0.05,
    false_discovery_rate: bool = True,
) -> plt.Figure | None:
    """One page of case-minus-reference differences, one reference per panel.

    ``reference_for`` picks each panel's reference, which is what distinguishes
    the three difference pages: a fixed control, the 1xCO2 case in the same
    column, or the no-hosing case in the same row. Panels whose reference is
    itself are zero by construction and are labeled rather than dropped, so
    every page keeps the same 3x3 skeleton.

    Returns None when nothing differs anywhere — SOLIN is the same prescribed
    insolation in every run, so its difference pages would be blank panels under
    an empty colorbar. That is a fact worth stating, not a page worth printing.
    """
    fields: dict[tuple[int, float], xr.DataArray] = {}
    masks: dict[tuple[int, float], xr.DataArray] = {}
    annotations: dict[tuple[int, float], str] = {}
    missing = _missing_notes(var)
    archiving = cases_with(var)

    for case in archiving:
        reference = reference_for(case)
        if reference not in archiving:
            missing[_key(case)] = f"{case.name}\nreference {reference.name}\nunavailable"
            continue
        # check_range=False: these are differences, not absolute values.
        fields[_key(case)] = to_display_units(
            steady_state_anomaly(case, var, reference=reference), var, check_range=False
        )
        if case == reference:
            annotations[_key(case)] = "reference (zero by construction)"
        else:
            masks[_key(case)] = significance_mask(
                case, var, reference=reference,
                alpha=alpha, false_discovery_rate=false_discovery_rate,
            )

    differenced = [da for key, da in fields.items() if key not in annotations]
    if not differenced:
        print(f"({var}: no {label} pairs available)", end=" ", flush=True)
        return None

    # Scale to the 98th percentile of |difference| rather than the maximum. A
    # few extreme polar grid cells otherwise set the range and wash out the
    # pattern everywhere else. Nothing is hidden: the colorbar carries extend
    # arrows showing that values run past both ends. Reference panels are
    # excluded because their identical zeros would drag the percentile down.
    stacked = np.abs(np.concatenate([da.values.ravel() for da in differenced]))
    if stacked.max() == 0.0:
        print(f"({var} identical across {label} — no page)", end=" ", flush=True)
        return None

    meta = info(var, next(iter(fields.values())))
    test = f"Welch t-test, p < {alpha}" + (", FDR controlled" if false_discovery_rate else "")
    marking = (
        "contours enclose significant regions"
        if significance_style == "outline"
        else "contours at colorbar values, drawn only where significant"
    )

    return grid_3x3(
        fields,
        title=title,
        subtitle=f"{describe}   ·   {WINDOW}   ·   {test} ({marking})",
        units=f"Δ {meta.units}",
        cmap=meta.diverging_cmap,
        vmin=-float(np.percentile(stacked, 98)),
        vmax=float(np.percentile(stacked, 98)),
        extend="both",
        significance=masks,
        significance_style=significance_style,
        annotations=annotations,
        missing_notes=missing,
    )


def anomaly_page(var: str, **kwargs) -> plt.Figure | None:
    """Every case minus the control — the total response to both perturbations."""
    return _difference_page(
        var,
        reference_for=lambda case: CONTROL,
        title=f"{var} — anomaly vs. {CONTROL.name}",
        describe=f"Each case minus {CONTROL.name}",
        label="control anomaly",
        **kwargs,
    )


def warming_effect_page(var: str, **kwargs) -> plt.Figure | None:
    """CO2 differences at fixed hosing: warming with the AMOC state held.

    Reading down a column of this page shows how the response to CO2 grows;
    reading across a row shows whether that response depends on the AMOC state,
    which is precisely the interaction the project is after.
    """
    return _difference_page(
        var,
        reference_for=lambda case: get_case(1, case.hosing),
        title=f"{var} — warming effect",
        describe="Each case minus 1xCO2 at the same hosing",
        label="CO2",
        **kwargs,
    )


def amoc_effect_page(var: str, **kwargs) -> plt.Figure | None:
    """Hosing differences at fixed CO2: the AMOC effect with forcing held.

    Reading across a row shows the response to hosing; reading down a column
    shows whether it depends on the CO2 level.
    """
    return _difference_page(
        var,
        reference_for=lambda case: get_case(case.co2, 0.0),
        title=f"{var} — AMOC effect",
        describe="Each case minus no-hosing at the same CO2",
        label="hosing",
        **kwargs,
    )


def pages(var: str, **kwargs) -> Iterator[plt.Figure]:
    """The page sequence for one variable.

    Absolute climatology, then the three differences: total response against the
    control, then the two axes of the design separated — warming at fixed
    hosing, and the AMOC at fixed CO2.

    A difference page needs at least one case-reference pair to exist. `TREFHT`
    is archived only by the control, so it gets an absolute page alone rather
    than three pages of empty panels.
    """
    yield absolute_page(var)
    if len(cases_with(var)) < 2:
        return
    for build in (anomaly_page, warming_effect_page, amoc_effect_page):
        page = build(var, **kwargs)
        if page is not None:
            yield page


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

    # Progress goes out on one line as each field starts, since a full book is
    # several minutes of silence otherwise. flush because stdout is block-
    # buffered when redirected to a file.
    print("coverage", end=" ", flush=True)
    yield status_page(variables)
    for var in variables:
        print(var, end=" ", flush=True)
        yield from pages(var, **kwargs)
    print()
