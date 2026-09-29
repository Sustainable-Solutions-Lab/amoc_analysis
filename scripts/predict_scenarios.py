"""Predicted decadal-mean field changes for 3 K of warming with and without AMOC decline.

Uses the decadal (10-year block-mean) pooled regressions in
``data/output/regression/<predictand>/decadal10/`` to map the predicted change in
the gridded tas, prc (convective precip), and pr (total precip) fields relative to
a control baseline, for set 5 (Tglob + AMOC) and set 10 (Tglob + AMOC +
Tglob*AMOC interaction).

Addresses: where does an AMOC decline exacerbate vs. ameliorate the response to
global warming? Three (Tglob, AMOC) states:

    baseline   : Tglob = T0,       AMOC = 20 Sv
    warm       : Tglob = T0 + 3 K, AMOC = 20 Sv  (warming, AMOC unchanged)
    warm-weak  : Tglob = T0 + 3 K, AMOC =  6 Sv  (warming, AMOC 20 -> 6 Sv)

T0 is the 1xCO2 control's global-mean tas over its AMOC-present years (the years
it contributes to the pooled fit). Global-mean warming is the same 3 K in both
warm states, so warm-weak - warm is the AMOC-decline effect at fixed global-mean
temperature: a pure spatial redistribution in tas (global mean ~0), though not
necessarily in precipitation. Both warm states sit inside the sampled predictor
space: 4xCO2_m03Sv (~291.5 K, ~20 Sv) and 4xCO2_p03Sv (~290.4 K, ~5.5 Sv) are
close neighbours, so these are interpolations, not extrapolations.

The predicted change between two conditions is coef . (predictor(X) - predictor(R));
the intercept cancels. Set 5 uses raw predictors; set 10 uses the centered columns
(q_Tglob, q_AMOC, q_Tglob.AMOC), evaluated with the pooled centering means read
from the coef file's ``centering_mean_*`` attributes, with the interaction term
formed per condition.

    python scripts/predict_scenarios.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import data_loader as dl
import regression as reg
from output import PROJECTION, PdfBook, plot_coefficient_map

REG_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression")
OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "scenarios")

BASELINE_CASE = "1xCO2"
WARMING_K = 3.0
AMOC_STRONG_SV = 20.0
AMOC_WEAK_SV = 6.0

# Columns to map: (title, condition X, reference R).
SCENARIOS = [
    (f"+{WARMING_K:g} K, AMOC {AMOC_STRONG_SV:g} Sv − baseline", "warm", "baseline"),
    (f"+{WARMING_K:g} K, AMOC {AMOC_WEAK_SV:g} Sv − baseline", "warm-weak", "baseline"),
    (f"AMOC {AMOC_STRONG_SV:g}→{AMOC_WEAK_SV:g} Sv effect at +{WARMING_K:g} K", "warm-weak", "warm"),
]

# Page layout. A column is either an int (an absolute map of SCENARIOS[i]) or a
# tuple (title, numerator_idx, denominator_idx) giving a percentage ratio
# 100 * change(num) / change(den). Page 1 = the three changes; page 2 = the AMOC
# effect as a percentage of the warming-only change -- the fractional increase(+)
# or decrease(-) in the warming response caused by the AMOC decline.
PAGES = [
    (f"{WARMING_K:g} K warming with / without AMOC {AMOC_STRONG_SV:g}→{AMOC_WEAK_SV:g} Sv",
     [0, 1, 2]),
    ("AMOC-decline effect as % of the warming-only change",
     [("AMOC effect ÷ warming-only change", 2, 0)]),
]

# The page-2 ratio explodes where the warming-only change crosses zero; fix every
# ratio panel to a common +/-100% scale (values beyond saturate).
RATIO_PCT_BOUND = 100.0

SET_FILES = {
    5: "coef_set5_Tglob-AMOC.nc",
    10: "coef_set10_Tglob-AMOC-TglobxAMOC.nc",
}


def baseline_tglob():
    """Global-mean tas of the baseline case over the years it contributes to the
    pooled fit (years with every predictor present)."""
    scal = xr.open_dataset(
        os.path.join(dl.PROCESSED_DIR, dl.scalar_file(BASELINE_CASE))
    )[reg.PREDICTOR_UNION]
    valid = scal.to_dataframe().dropna().index
    return float(scal["tas_global_mean"].sel(year=valid).mean())


def conditions():
    """(Tglob [K], AMOC [Sv]) per condition."""
    t0 = baseline_tglob()
    return {
        "baseline": (t0, AMOC_STRONG_SV),
        "warm": (t0 + WARMING_K, AMOC_STRONG_SV),
        "warm-weak": (t0 + WARMING_K, AMOC_WEAK_SV),
    }


def predicted_change(coef, set_num, condX, condR, cond, mT, mA):
    """Predicted field change coef . (predictor(X) - predictor(R)) for a set."""
    Tx, Ax = cond[condX]
    Tr, Ar = cond[condR]
    if set_num == 5:  # raw predictors
        return (coef.sel(param="tas_global_mean") * (Tx - Tr)
                + coef.sel(param="amoc_strength") * (Ax - Ar))
    # set 10: centered columns (q_Tglob, q_AMOC, q_Tglob.AMOC); means from the fit.
    qx = (Tx - mT, Ax - mA, (Tx - mT) * (Ax - mA))
    qr = (Tr - mT, Ar - mA, (Tr - mT) * (Ar - mA))
    return (coef.sel(param="q_Tglob") * (qx[0] - qr[0])
            + coef.sel(param="q_AMOC") * (qx[1] - qr[1])
            + coef.sel(param="q_Tglob.AMOC") * (qx[2] - qr[2]))


def run_for_predictand(name, cond):
    predictand = reg.PREDICTANDS[name]
    units, cmap = predictand["units"], predictand["cmap"]
    sets = sorted(SET_FILES)
    dsets = {
        s: xr.open_dataset(os.path.join(REG_BASE, name, "decadal10", SET_FILES[s]))
        for s in sets
    }
    coefs = {s: dsets[s]["coef"] for s in sets}
    mT = float(dsets[10].attrs["centering_mean_Tglob_K"])
    mA = float(dsets[10].attrs["centering_mean_AMOC_Sv"])
    print(f"[{name}] set-10 centering: Tglob={mT:.3f} K, AMOC={mA:.3f} Sv")

    out_path = os.path.join(OUT_DIR, f"predicted_change_{name}.pdf")
    with PdfBook(out_path) as pdf:
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
                        num = predicted_change(coef, set_num, SCENARIOS[ni][1], SCENARIOS[ni][2], cond, mT, mA)
                        den = predicted_change(coef, set_num, SCENARIOS[di][1], SCENARIOS[di][2], cond, mT, mA)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio = 100.0 * num / den
                        change = ratio.where(np.isfinite(ratio))
                        u, bnd = "% of warming-only change", RATIO_PCT_BOUND
                    else:  # absolute map of SCENARIOS[col]
                        title, condX, condR = SCENARIOS[col]
                        change = predicted_change(coef, set_num, condX, condR, cond, mT, mA)
                        u, bnd = f"Δ{name} ({units})", None
                        print(f"[{name}] set {set_num}: {title}: global mean "
                              f"{float(dl.global_mean(change)):.4g} {units}")
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
    cond = conditions()
    for label, (t, a) in cond.items():
        print(f"{label:10s}: Tglob={t:.3f} K, AMOC={a:.2f} Sv")
    for name in ("tas", "prc", "pr"):
        run_for_predictand(name, cond)


if __name__ == "__main__":
    main()
