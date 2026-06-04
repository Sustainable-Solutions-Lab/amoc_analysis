"""Pooled per-grid-point regressions of gridded tas on scalar indices.

Builds one pooled sample (years with all predictors present, across all four
simulations), then for each of the six predictor sets fits a per-grid-point OLS
and writes stippled coefficient maps (PDF) plus the coefficient fields (NetCDF)
to ``data/output/regression/``.

    python scripts/run_regressions.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import plot_set

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")

CAVEATS = """Regression outputs: pooled per-grid-point OLS of gridded annual-mean tas.

- One regression per grid cell; the years of all four simulations
  (historical-ssp585, abrupt-4xCO2, piControl, u03-hos) are POOLED into a single
  fit with a common intercept and no per-run fixed effects. Pooling exploits
  between-run differences in the index relationships to reduce collinearity.
- Common sample: years with all predictors (Tglob, dT_NS, AMOC) present, per run
  (= AMOC-present years), identical for all six sets.
- p-values are nominal OLS (independent residuals). Within-run autocorrelation of
  annual data makes them OPTIMISTIC; treat significance as indicative.
- Set 6 (three predictors) retains high collinearity (VIF ~ 22); its partial
  coefficients are poorly constrained. See per-set VIF printed at build time.
- Coefficient units: tas on Tglob and dT_NS are K/K; tas on AMOC is K/Sv.
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    predictors, tas = reg.build_pooled()
    per_run = predictors["run"].to_series().value_counts().to_dict()
    print(f"pooled sample: n={predictors.sizes['sample']}  per-run={per_run}")
    vif = reg.variance_inflation_factors(predictors)
    print("VIF (3-predictor union):", {k: round(v, 2) for k, v in vif.items()})

    run_label = "pooled: " + ", ".join(reg.RUNS)
    for set_def in reg.PREDICTOR_SETS:
        names = set_def["predictors"]
        fit = reg.fit_grid_ols(predictors[names], tas)

        labels = "-".join(reg.PREDICTORS[p]["label"] for p in names)
        pdf = os.path.join(OUT_DIR, f"coef_set{set_def['number']}_{labels}.pdf")
        nc = os.path.join(OUT_DIR, f"coef_set{set_def['number']}_{labels}.nc")
        plot_set(fit, set_def, run_label, pdf)
        fit.to_netcdf(nc)

        gmean_tglob = (
            float(dl.global_mean(fit["coef"].sel(param="tas_global_mean")))
            if "tas_global_mean" in names
            else float("nan")
        )
        print(
            f"set {set_def['number']} ({labels}): nobs={fit.attrs['nobs']} "
            f"-> {os.path.basename(pdf)}; global-mean Tglob coef={gmean_tglob:.3f}"
        )

    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(CAVEATS)
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
