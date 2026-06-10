"""EOF / principal-component analysis of pooled gridded fields (additive path).

For each predictand (tas, prc): compute area-weighted covariance EOFs of the
grand-mean anomalies over the pooled AMOC-complete sample, plot the leading EOF
patterns, plot the principal-component (EOF weighting) time series per simulation,
and regress the leading PCs (>=95% variance) on each predictor set, saving the
PC-space regression coefficients. Complements scripts/run_regressions.py.

Two smoothing variants are produced: 'annual' (interannual, <predictand>/) and
'decadal10' (10-year block means, <predictand>/decadal10/).

    python scripts/run_eof_regressions.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import eof
import regression as reg
from output import (
    plot_eof_patterns,
    plot_pc_prediction,
    plot_pc_regression,
    plot_pc_timeseries,
)

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "eof")
PREDICTAND_NAMES = ["tas", "prc"]
# Retain leading EOFs until cumulative variance reaches VARIANCE_THRESHOLD, but
# never keep a mode explaining less than MIN_VARIANCE_FRACTION (drops the noise
# tail; the more restrictive rule wins).
VARIANCE_THRESHOLD = 0.95
MIN_VARIANCE_FRACTION = 0.01

# Smoothing variants (see scripts/run_regressions.py): 'annual' is the
# interannual analysis; 'decadal10' low-passes to slow timescales via 10-year
# block means applied per run/segment to predictors and predictand before pooling.
SMOOTHINGS = [
    {"tag": "annual", "block": None, "subdir": ""},
    {"tag": "decadal10", "block": 10, "subdir": "decadal10"},
]

CAVEATS = """EOF / principal-component analysis outputs (additive to the direct maps).

- Area-weighted covariance EOFs of grand-mean anomalies over the pooled
  AMOC-complete sample (4 simulations). Modes are truncated by two rules (more
  restrictive wins): keep until cumulative variance >= 95%, but never keep a mode
  explaining < 1% of variance (drops the low-variance noise tail).
- Outputs (all variants): eof_patterns.pdf (leading EOF maps + scree),
  pc_timeseries.pdf (the EOF weightings / PCs over time, one panel per simulation),
  and pc_regression_set{N}_*.nc (OLS of the PCs on each predictor set:
  coef/se/t/p/r2 in PC space). The spatial fingerprint maps (Sum_k beta_k * EOF_k)
  are intentionally NOT produced.
- DECADAL variant only also gets figures of the PC-on-scalar regression -- the EOF
  analog of the 2D coefficient maps with the mode index replacing (lat, lon):
    * pc_regression.pdf -- one PAGE per predictor set; each page has one panel per
      EOF mode, a bar per predictor showing the STANDARDIZED coefficient
      beta*sigma(x)/sigma(PC) with +/-SE; faded bars are not significant (p>0.05);
      panel title shows R^2 and % var. Standardizing makes bars comparable across
      modes (raw coefs scale with each PC's amplitude). t/p are scale-invariant
      (same as the per-set pc_regression_set{N}_*.nc files).
    * pc_prediction.pdf -- one PAGE per set (6, 9, 10): fitted (X*beta) vs actual
      PC over time per simulation, a direct view of how well the scalars predict
      each weighting.
- Two smoothing variants: 'annual' (interannual, this directory) and 'decadal10'
  (decadal10/ subdir) = 10-year block means per run/segment before pooling.
- Annual tas is strongly low-rank (2 modes >=95%); annual prc is not
  (237 modes); decadal smoothing removes the high-frequency noise and lowers the
  retained-mode count (reported at run time).
- p-values are nominal OLS; 'annual' is autocorrelation-optimistic, 'decadal10'
  (~independent decadal samples) is far more trustworthy.
"""


def run_for_predictand(name, smoothing):
    predictand = reg.PREDICTANDS[name]
    tag = smoothing["tag"]
    out_dir = os.path.join(OUT_BASE, name, smoothing["subdir"])
    os.makedirs(out_dir, exist_ok=True)

    predictors, response = reg.build_pooled(predictand=predictand, block=smoothing["block"])
    predictors = reg.add_orthogonalized_columns(predictors)
    predictors = reg.add_quadratic_columns(predictors)

    eof_ds = eof.compute_eofs(
        response, variance_threshold=VARIANCE_THRESHOLD,
        min_variance_fraction=MIN_VARIANCE_FRACTION,
    )
    n = eof_ds.attrs["n_modes"]
    print(f"\n[{name}/{tag}] pooled n={predictors.sizes['sample']}; {n} EOF modes retained "
          f"(cum var {eof_ds.attrs['total_variance_fraction'] * 100:.1f}%); "
          f"leading % = {(eof_ds['variance_fraction'].values[:6] * 100).round(1)}")

    pat_units = "K" if name == "tas" else "kg m-2 s-1"
    plot_eof_patterns(
        eof_ds, f"EOF patterns: {name} ({tag} anomalies)", pat_units,
        os.path.join(out_dir, "eof_patterns.pdf"),
        cmap=predictand.get("cmap", "RdBu_r"),
    )
    plot_pc_timeseries(
        eof_ds, f"EOF weightings (PCs) over time: {name} ({tag})",
        os.path.join(out_dir, "pc_timeseries.pdf"),
    )

    decadal = smoothing["block"] is not None  # the focus variant gets figures
    var_frac = eof_ds["variance_fraction"].values
    # One multi-page PDF collects every set's regression figure (one page per set).
    reg_pdf = PdfPages(os.path.join(out_dir, "pc_regression.pdf")) if decadal else None
    fits = {}
    for set_def in reg.PREDICTOR_SETS:
        num = set_def["number"]
        names = set_def["predictors"]
        pc_fit = eof.fit_pcs(predictors[names], eof_ds["pcs"])
        fits[num] = (pc_fit, names)
        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        plabels = ", ".join(reg.PREDICTORS[p]["label"] for p in names)
        pc_fit[["coef", "se", "tstat", "pvalue", "r2"]].to_netcdf(
            os.path.join(out_dir, f"pc_regression_set{num}_{labels}.nc"))
        if decadal:
            plot_pc_regression(
                pc_fit, predictors[names], eof_ds["pcs"],
                f"PC regression: {name} ({tag}) — set {num}: {plabels}",
                variance_fraction=var_frac, pdf=reg_pdf,
            )
        print(f"[{name}/{tag}] set {num} ({labels}) regressed")

    # Direct "predict the weightings" view for the richer sets (full 3-index,
    # quadratic, and Tglob×AMOC interaction): fitted vs actual PC over time, one
    # page per set, decadal only.
    if decadal:
        reg_pdf.close()
        with PdfPages(os.path.join(out_dir, "pc_prediction.pdf")) as pred_pdf:
            for num in (6, 9, 10):
                pc_fit, names = fits[num]
                plot_pc_prediction(
                    eof_ds, pc_fit, predictors[names],
                    f"PC fitted vs actual: {name} ({tag}) — set {num}",
                    pdf=pred_pdf,
                )

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(CAVEATS)


def main():
    for name in PREDICTAND_NAMES:
        for smoothing in SMOOTHINGS:
            run_for_predictand(name, smoothing)
    print(f"\nDone. Outputs in {OUT_BASE}/<predictand>/[decadal10/]")


if __name__ == "__main__":
    main()
