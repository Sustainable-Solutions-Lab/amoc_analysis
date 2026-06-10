"""Predicted decadal-mean field changes for end-of-century scenarios.

Uses the decadal (10-year block-mean) pooled regressions in
``data/output/regression/<predictand>/decadal10/`` to map the predicted change in
the gridded tas and prc (convective precip) fields for two end-of-century SSP5-8.5 conditions,
relative to the piControl baseline, plus their difference (which isolates the
AMOC-slowdown fingerprint). Done for set 5 (Tglob + AMOC) and set 10
(Tglob + AMOC + Tglob*AMOC interaction).

Addresses: where does AMOC slowdown exacerbate vs. ameliorate the CO2-induced
changes in surface temperature and precipitation? We compare the high-CO2 world
WITH AMOC slowdown (SSP585) against two counterfactuals WITHOUT slowdown (AMOC
restored to the control value), which bracket the unknown global-mean-temperature
effect of the slowdown:

    piControl    : Tglob = 287.207, AMOC = 17.44
    SSP585       : Tglob = 293.090, AMOC =  7.34
    SSP585-adj1  : Tglob = 294.665, AMOC = 17.44  (assm. 1: slowdown cooled by
                   0.1558 K/Sv * 10.10 Sv ~= 1.575 K, so add it back)
    SSP585-adj2  : Tglob = 293.090, AMOC = 17.44  (assm. 2: slowdown had no global
                   temperature effect; only AMOC is restored)

The predicted change between two conditions is coef . (predictor(X) - predictor(R));
the intercept cancels. Set 5 uses raw predictors; set 10 uses the centered columns
(q_Tglob, q_AMOC, q_Tglob*AMOC) from add_quadratic_columns, so its columns are
evaluated with the decadal pooled centering means and the interaction term is
formed per condition. SSP585 - adjN isolates the AMOC-slowdown effect under each
assumption (under adj2, global-mean tas is held fixed, so it is pure pattern).

    python scripts/predict_scenarios.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import regression as reg
from output import PROJECTION, plot_coefficient_map

REG_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression")
OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "scenarios")

# (Tglob [K], AMOC [Sv]) per condition.
COND = {
    "piControl": (287.207, 17.44),
    "SSP585": (293.090, 7.34),
    "SSP585-adj1": (294.665, 17.44),  # assm.1: add back AMOC-implied 1.575 K cooling
    "SSP585-adj2": (293.090, 17.44),  # assm.2: no global-T effect; only AMOC restored
}

# Columns to map: (title, condition X, reference R).
SCENARIOS = [
    ("SSP585 − piControl", "SSP585", "piControl"),
    ("SSP585-adj1 − piControl", "SSP585-adj1", "piControl"),
    ("SSP585 − adj1  (AMOC effect, assm.1)", "SSP585", "SSP585-adj1"),
    ("SSP585-adj2 − piControl", "SSP585-adj2", "piControl"),
    ("SSP585 − adj2  (AMOC effect, assm.2)", "SSP585", "SSP585-adj2"),
]

# Page layout. A column is either an int (an absolute map of SCENARIOS[i]) or a
# tuple (title, numerator_idx, denominator_idx) giving a percentage ratio
# 100 * change(num) / change(den). Page 1 = change relative to piControl (cols
# 1,2,4); page 2 = AMOC-slowdown effect (cols 3,5); page 3 = that effect as a
# percentage of the no-slowdown CO2-only change, i.e. (SSP585-adjN)/(adjN-piControl)
# -- the fractional increase(+)/decrease(-) in the response caused by the slowdown.
PAGES = [
    ("change relative to piControl (with / without AMOC slowdown)", [0, 1, 3]),
    ("AMOC-slowdown effect (SSP585 − adjN)", [2, 4]),
    ("AMOC-slowdown effect as % of the no-slowdown CO₂ change  [(SSP585−adjN)/(adjN−piControl)]",
     [("SSP585 − adj1  ÷ (adj1 − piControl)", 2, 1),
      ("SSP585 − adj2  ÷ (adj2 − piControl)", 4, 3)]),
]

# The page-3 ratio explodes where the denominator (adjN−piControl) crosses zero.
# Fix every page-3 panel to a common +/-100% scale so panels are directly
# comparable (values beyond +/-100% saturate).
RATIO_PCT_BOUND = 100.0

SET_FILES = {
    5: "coef_set5_Tglob-AMOC.nc",
    10: "coef_set10_Tglob-AMOC-TglobxAMOC.nc",
}


def predicted_change(coef, set_num, condX, condR, mT, mA):
    """Predicted field change coef . (predictor(X) - predictor(R)) for a set."""
    Tx, Ax = COND[condX]
    Tr, Ar = COND[condR]
    if set_num == 5:  # raw predictors
        return (coef.sel(param="tas_global_mean") * (Tx - Tr)
                + coef.sel(param="amoc_strength") * (Ax - Ar))
    # set 10: centered columns (q_Tglob, q_AMOC, q_Tglob.AMOC); means from the fit.
    qx = (Tx - mT, Ax - mA, (Tx - mT) * (Ax - mA))
    qr = (Tr - mT, Ar - mA, (Tr - mT) * (Ar - mA))
    return (coef.sel(param="q_Tglob") * (qx[0] - qr[0])
            + coef.sel(param="q_AMOC") * (qx[1] - qr[1])
            + coef.sel(param="q_Tglob.AMOC") * (qx[2] - qr[2]))


def run_for_predictand(name, mT, mA):
    predictand = reg.PREDICTANDS[name]
    units, cmap = predictand["units"], predictand.get("cmap", "RdBu_r")
    sets = sorted(SET_FILES)
    coefs = {
        s: xr.open_dataset(os.path.join(REG_BASE, name, "decadal10", SET_FILES[s]))["coef"]
        for s in sets
    }

    ratio_bound = RATIO_PCT_BOUND  # common ±100% scale on page 3 for all panels
    out_path = os.path.join(OUT_DIR, f"predicted_change_{name}.pdf")
    with PdfPages(out_path) as pdf:
        for page_title, cols in PAGES:
            fig, axes = plt.subplots(
                len(sets), len(cols),
                figsize=(5.2 * len(cols), 3.4 * len(sets)),
                squeeze=False, subplot_kw={"projection": PROJECTION},
            )
            for i, set_num in enumerate(sets):
                coef = coefs[set_num]
                for j, col in enumerate(cols):
                    if isinstance(col, tuple):  # ratio column: (title, num_idx, den_idx)
                        title, ni, di = col
                        num = predicted_change(coef, set_num, SCENARIOS[ni][1], SCENARIOS[ni][2], mT, mA)
                        den = predicted_change(coef, set_num, SCENARIOS[di][1], SCENARIOS[di][2], mT, mA)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio = 100.0 * num / den
                        change = ratio.where(np.isfinite(ratio))
                        u, bnd = "% of CO₂-only change", ratio_bound
                    else:  # absolute map of SCENARIOS[col]
                        title, condX, condR = SCENARIOS[col]
                        change = predicted_change(coef, set_num, condX, condR, mT, mA)
                        u, bnd = f"Δ{name} ({units})", None
                    plot_coefficient_map(
                        change, xr.zeros_like(change),  # zeros -> no stippling
                        title=f"set {set_num}: {title}", units=u,
                        ax=axes[i, j], cmap=cmap, bound=bnd,
                    )
            fig.suptitle(
                f"Predicted decadal-mean {name} change (decadal10 regressions)\n"
                f"{page_title}", fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {out_path}  ({len(PAGES)} pages)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # Decadal pooled centering means (exactly those add_quadratic_columns used).
    pooled, _ = reg.build_pooled(block=10)
    mT = float(pooled["tas_global_mean"].mean())
    mA = float(pooled["amoc_strength"].mean())
    print(f"decadal pooled means: Tglob={mT:.3f} K, AMOC={mA:.3f} Sv  (set-10 centering)")
    for name in ("tas", "prc"):
        run_for_predictand(name, mT, mA)


if __name__ == "__main__":
    main()
