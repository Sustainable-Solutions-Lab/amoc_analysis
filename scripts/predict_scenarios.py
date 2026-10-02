"""Predicted decadal-mean field changes on a warming x AMOC-decline grid.

Uses the decadal (10-year block-mean) pooled regressions in
``data/output/regression/<predictand>_coef.nc`` to map predicted changes in the
gridded predictand fields (``--variables``, default all), for set 5
(Tglob + AMOC) and set 10 (Tglob + AMOC + Tglob*AMOC interaction).

Addresses: where does an AMOC decline exacerbate vs. ameliorate the response to
global warming? Four (Tglob, AMOC) states form a 2 x 2 factorial:

    reference  : Tglob = T0,       AMOC = 20 Sv
    weak       : Tglob = T0,       AMOC =  6 Sv
    warm       : Tglob = T0 + 3 K, AMOC = 20 Sv
    warm-weak  : Tglob = T0 + 3 K, AMOC =  6 Sv

T0 is the 1xCO2 control's global-mean tas over its AMOC-present years (the years
it contributes to the pooled fit). Each page is a 3 x 3 grid: the corners are the
four states' changes from the reference, and each edge is the difference of its
two neighbouring corners:

    [0,0] reference (= 0)    [0,1] AMOC effect at +0 K   [0,2] weak - reference
    [1,0] warming at 20 Sv   [1,1] interaction (set 10)  [1,2] warming at 6 Sv
    [2,0] warm - reference   [2,1] AMOC effect at +3 K   [2,2] warm-weak - reference

The centre is the interaction, [2,1] - [0,1] (= [1,2] - [1,0]): how much the AMOC
effect changes with 3 K of warming. It is identically zero for the additive set 5,
so set 5 leaves it blank. Global-mean tas is held fixed along each row, so the AMOC
effects are pure spatial redistributions of tas (global mean ~0), though not of
precipitation. All four states sit inside the sampled predictor space (between
the 1x-4xCO2 runs at ~20 Sv and the +0.3 Sv runs at ~5.5 Sv), so these are
interpolations, not extrapolations.

The predicted change between two states is coef . (predictor(X) - predictor(R));
the intercept cancels. Set 5 uses raw predictors; set 10 uses the centered columns
(q_Tglob, q_AMOC, q_Tglob.AMOC), evaluated with the pooled centering means read
from the coef file's ``centering_mean_*`` attributes, with the interaction term
formed per state. All panels for a predictand, in both sets, share one symmetric
color scale, so set 5 and set 10 compare directly. Each map page is followed by a
page of the same panels' zonal statistics over longitude (mean, median, 5-95% band,
min/max against sine of latitude), again on one y range shared by both sets. One
PDF per predictand, two pages per set: ``data/output/scenarios/<predictand>_scenarios.pdf``.

    python scripts/predict_scenarios.py [--variables minimal | tas pr ...]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import regression as reg
from output import label_with_units, plot_map_grid, symmetric_bound, value_with_units, zonal_range

REG_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")
OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "scenarios")

BASELINE_CASE = "1xCO2"
WARMING_K = 3.0
AMOC_STRONG_SV = 20.0
AMOC_WEAK_SV = 6.0

SET_NUMBERS = [5, 10]

WARM = f"+{WARMING_K:g} K"
AMOC_DROP = f"AMOC {AMOC_STRONG_SV:g}→{AMOC_WEAK_SV:g} Sv"

# Corners of the 3 x 3 grid: (row, col) -> (panel title, state). Each corner maps
# that state's predicted change from the reference.
CORNERS = {
    (0, 0): (f"+0 K, {AMOC_STRONG_SV:g} Sv (reference)", "reference"),
    (0, 2): (f"+0 K, {AMOC_WEAK_SV:g} Sv − reference", "weak"),
    (2, 0): (f"{WARM}, {AMOC_STRONG_SV:g} Sv − reference", "warm"),
    (2, 2): (f"{WARM}, {AMOC_WEAK_SV:g} Sv − reference", "warm-weak"),
}
# Edges: (row, col) -> (panel title, minuend cell, subtrahend cell).
EDGES = {
    (0, 1): (f"{AMOC_DROP} effect at +0 K", (0, 2), (0, 0)),
    (2, 1): (f"{AMOC_DROP} effect at {WARM}", (2, 2), (2, 0)),
    (1, 0): (f"{WARM} warming effect at {AMOC_STRONG_SV:g} Sv", (2, 0), (0, 0)),
    (1, 2): (f"{WARM} warming effect at {AMOC_WEAK_SV:g} Sv", (2, 2), (0, 2)),
}
CENTER_TITLE = f"interaction: AMOC effect at {WARM} − at +0 K"


def baseline_tglob():
    """Global-mean tas of the baseline case over the years it contributes to the
    pooled fit (years with every predictor present)."""
    scal = xr.open_dataset(
        os.path.join(dl.PROCESSED_DIR, dl.scalar_file(BASELINE_CASE))
    )[reg.PREDICTOR_UNION]
    valid = scal.to_dataframe().dropna().index
    return float(scal["tas_global_mean"].sel(year=valid).mean())


def states():
    """(Tglob [K], AMOC [Sv]) per state."""
    t0 = baseline_tglob()
    return {
        "reference": (t0, AMOC_STRONG_SV),
        "weak": (t0, AMOC_WEAK_SV),
        "warm": (t0 + WARMING_K, AMOC_STRONG_SV),
        "warm-weak": (t0 + WARMING_K, AMOC_WEAK_SV),
    }


def predicted_change(coef, set_num, state_x, state_r, mT, mA):
    """Predicted field change coef . (predictor(X) - predictor(R)) for a set;
    ``state_x`` and ``state_r`` are (Tglob, AMOC) pairs."""
    Tx, Ax = state_x
    Tr, Ar = state_r
    if set_num == 5:  # raw predictors
        return (coef.sel(param="tas_global_mean") * (Tx - Tr)
                + coef.sel(param="amoc_strength") * (Ax - Ar))
    # set 10: centered columns (q_Tglob, q_AMOC, q_Tglob.AMOC); means from the fit.
    qx = (Tx - mT, Ax - mA, (Tx - mT) * (Ax - mA))
    qr = (Tr - mT, Ar - mA, (Tr - mT) * (Ar - mA))
    return (coef.sel(param="q_Tglob") * (qx[0] - qr[0])
            + coef.sel(param="q_AMOC") * (qx[1] - qr[1])
            + coef.sel(param="q_Tglob.AMOC") * (qx[2] - qr[2]))


def grid_panels(coef, set_num, state, mT, mA):
    """The 3 x 3 grid for one set: (row, col) -> (panel title, change field)."""
    fields = {cell: predicted_change(coef, set_num, state[s], state["reference"], mT, mA)
              for cell, (_, s) in CORNERS.items()}
    fields.update({cell: fields[a] - fields[b] for cell, (_, a, b) in EDGES.items()})
    titles = {cell: spec[0] for cell, spec in {**CORNERS, **EDGES}.items()}
    if set_num == 10:
        fields[1, 1] = fields[2, 1] - fields[0, 1]
        titles[1, 1] = CENTER_TITLE
    return {cell: (titles[cell], fields[cell]) for cell in fields}


def run_for_predictand(name, state):
    predictand = reg.PREDICTANDS[name]
    units, cmap = predictand["units"], predictand["cmap"]
    coef_path = os.path.join(REG_DIR, f"{name}_coef.nc")
    dsets = {s: xr.open_dataset(coef_path, group=reg.set_group(s)) for s in SET_NUMBERS}
    mT = float(dsets[10].attrs["centering_mean_Tglob_K"])
    mA = float(dsets[10].attrs["centering_mean_AMOC_Sv"])
    print(f"[{name}] set-10 centering: Tglob={mT:.3f} K, AMOC={mA:.3f} Sv")

    panels = {s: grid_panels(dsets[s]["coef"], s, state, mT, mA) for s in SET_NUMBERS}
    fields = [field for grid in panels.values() for _, field in grid.values()]
    bound = symmetric_bound(np.concatenate([field.values.ravel() for field in fields]))
    zonal_ylim = zonal_range(fields)
    out_path = os.path.join(OUT_DIR, f"{name}_scenarios.pdf")
    with PdfPages(out_path) as pdf:
        for set_num, grid in panels.items():
            for cell, (title, field) in sorted(grid.items()):
                print(f"[{name}] set {set_num} {cell}: {title}: global mean "
                      + value_with_units(f"{float(dl.global_mean(field)):.4g}", units))
            plot_map_grid(
                grid, (3, 3),
                title=(f"Predicted decadal-mean Δ{name}, set {set_num} "
                       f"({'Tglob + AMOC' if set_num == 5 else 'Tglob + AMOC + Tglob·AMOC'}); "
                       f"rows: warming, columns: AMOC decline"),
                units=label_with_units(f"Δ{name}", units), cmap=cmap, bound=bound,
                zonal_ylim=zonal_ylim, pdf=pdf,
            )
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    dl.add_variables_argument(parser)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    state = states()
    for label, (t, a) in state.items():
        print(f"{label:10s}: Tglob={t:.3f} K, AMOC={a:.2f} Sv")
    for name in dl.resolve_variables(args.variables):
        run_for_predictand(name, state)


if __name__ == "__main__":
    main()
