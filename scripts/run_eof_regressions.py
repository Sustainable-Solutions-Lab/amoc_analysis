"""EOF / principal-component regression of pooled gridded fields (additive path).

For each predictand (tas, precip): compute area-weighted covariance EOFs of the
grand-mean anomalies over the pooled AMOC-complete sample, regress the leading
PCs (≥95% variance) on each predictor set, and map the coefficients back to space
(Σ_k β_k·EOF_k). Complements — does not replace — scripts/run_regressions.py.

    python scripts/run_eof_regressions.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import eof
import regression as reg
from output import plot_eof_patterns, plot_pc_timeseries, plot_set

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "eof")
PREDICTAND_NAMES = ["tas", "precip"]
VARIANCE_THRESHOLD = 0.95

CAVEATS = """EOF / principal-component regression outputs (additive to the direct maps).

- Area-weighted covariance EOFs of grand-mean anomalies over the pooled
  AMOC-complete 500-yr sample (4 simulations); leading modes to >=95% variance.
- PC time series are regressed on the same predictor sets as the direct analysis;
  fingerprint maps are Sum_k beta_k * EOF_k (EOF-filtered coefficient maps).
- With all modes retained this equals the direct field regression exactly;
  truncation denoises it.
- tas is strongly low-rank (2 modes >=95%); precip is not (237 modes >=95%), so
  precip fingerprints at this threshold ~= the direct result (little denoising).
  eof_patterns.pdf maps only the leading 9 modes; the regression uses all retained.
- p-values are nominal OLS (autocorrelation not corrected); fingerprint
  significance treats cross-mode coefficients as independent (PCs are orthogonal).
"""


def run_for_predictand(name):
    predictand = reg.PREDICTANDS[name]
    out_dir = os.path.join(OUT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)

    predictors, response = reg.build_pooled(predictand=predictand)
    predictors = reg.add_orthogonalized_columns(predictors)
    predictors = reg.add_quadratic_columns(predictors)

    eof_ds = eof.compute_eofs(response, variance_threshold=VARIANCE_THRESHOLD)
    n = eof_ds.attrs["n_modes"]
    print(f"\n[{name}] {n} EOF modes retained "
          f"(cum var {eof_ds.attrs['total_variance_fraction'] * 100:.1f}%); "
          f"per-mode % = {(eof_ds['variance_fraction'].values * 100).round(1)}")

    pat_units = "K" if name == "tas" else "kg m-2 s-1"
    plot_eof_patterns(
        eof_ds, f"EOF patterns: {name} (pooled anomalies)", pat_units,
        os.path.join(out_dir, "eof_patterns.pdf"),
        cmap=predictand.get("cmap", "RdBu_r"),
    )
    plot_pc_timeseries(
        eof_ds, f"EOF weightings (PCs) over time: {name}",
        os.path.join(out_dir, "pc_timeseries.pdf"),
    )

    run_label = f"predictand={name}; EOF-PC regression ({n} modes, >=95% var)"
    for set_def in reg.PREDICTOR_SETS:
        names = set_def["predictors"]
        pc_fit = eof.fit_pcs(predictors[names], eof_ds["pcs"])
        fingerprint = eof.reconstruct_fingerprint(eof_ds, pc_fit)

        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        plot_set(fingerprint, set_def, run_label,
                 os.path.join(out_dir, f"fingerprint_set{set_def['number']}_{labels}.pdf"),
                 predictand)
        pc_fit[["coef", "se", "tstat", "pvalue"]].to_netcdf(
            os.path.join(out_dir, f"pc_regression_set{set_def['number']}_{labels}.nc"))
        print(f"[{name}] set {set_def['number']} ({labels}) done")

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(CAVEATS)


def main():
    for name in PREDICTAND_NAMES:
        run_for_predictand(name)
    print(f"\nDone. Outputs in {OUT_BASE}/<predictand>/")


if __name__ == "__main__":
    main()
