"""EOF / principal-component analysis of pooled gridded fields (additive path).

For each predictand (tas, precip): compute area-weighted covariance EOFs of the
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

import data_loader as dl
import eof
import regression as reg
from output import plot_eof_patterns, plot_pc_timeseries

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "eof")
PREDICTAND_NAMES = ["tas", "precip"]
VARIANCE_THRESHOLD = 0.95

# Smoothing variants (see scripts/run_regressions.py): 'annual' is the
# interannual analysis; 'decadal10' low-passes to slow timescales via 10-year
# block means applied per run/segment to predictors and predictand before pooling.
SMOOTHINGS = [
    {"tag": "annual", "block": None, "subdir": ""},
    {"tag": "decadal10", "block": 10, "subdir": "decadal10"},
]

CAVEATS = """EOF / principal-component analysis outputs (additive to the direct maps).

- Area-weighted covariance EOFs of grand-mean anomalies over the pooled
  AMOC-complete sample (4 simulations); leading modes to >=95% variance.
- Outputs: eof_patterns.pdf (leading EOF maps + scree), pc_timeseries.pdf (the
  EOF weightings / PCs over time, one panel per simulation), and
  pc_regression_set{N}_*.nc (OLS of the PCs on each predictor set: coef/se/t/p in
  PC space). The spatial fingerprint maps (Sum_k beta_k * EOF_k) are intentionally
  NOT produced -- only the EOFs and PC weightings are wanted.
- Two smoothing variants: 'annual' (interannual, this directory) and 'decadal10'
  (decadal10/ subdir) = 10-year block means per run/segment before pooling.
- Annual tas is strongly low-rank (2 modes >=95%); annual precip is not
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

    eof_ds = eof.compute_eofs(response, variance_threshold=VARIANCE_THRESHOLD)
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

    for set_def in reg.PREDICTOR_SETS:
        names = set_def["predictors"]
        pc_fit = eof.fit_pcs(predictors[names], eof_ds["pcs"])
        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        pc_fit[["coef", "se", "tstat", "pvalue"]].to_netcdf(
            os.path.join(out_dir, f"pc_regression_set{set_def['number']}_{labels}.nc"))
        print(f"[{name}/{tag}] set {set_def['number']} ({labels}) regressed")

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(CAVEATS)


def main():
    for name in PREDICTAND_NAMES:
        for smoothing in SMOOTHINGS:
            run_for_predictand(name, smoothing)
    print(f"\nDone. Outputs in {OUT_BASE}/<predictand>/[decadal10/]")


if __name__ == "__main__":
    main()
