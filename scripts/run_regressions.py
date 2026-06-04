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

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression")

# Predictands to analyze (each writes to its own subdirectory).
PREDICTAND_NAMES = ["tas", "precip"]

CAVEATS = """Regression outputs: pooled per-grid-point OLS of a gridded predictand.

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
- Coefficient units are [predictand units] / [predictor units] (predictor units:
  Tglob, dT_NS in K; AMOC in Sv).
- The 'precip' predictand POOLS total pr (historical-ssp585, abrupt-4xCO2) with
  CONVECTIVE prc (piControl, u03-hos) -- different quantities, prototype only.
"""


def run_for_predictand(name):
    predictand = reg.PREDICTANDS[name]
    out_dir = os.path.join(OUT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)

    predictors, response = reg.build_pooled(predictand=predictand)
    predictors = reg.add_orthogonalized_columns(predictors)  # for sets 7-8
    predictors = reg.add_quadratic_columns(predictors)  # for set 9
    per_run = predictors["run"].to_series().value_counts().to_dict()
    vif = reg.variance_inflation_factors(predictors[reg.PREDICTOR_UNION])
    print(f"\n[{name}] pooled sample: n={predictors.sizes['sample']}  per-run={per_run}")
    print(f"[{name}] VIF (3-predictor union):", {k: round(v, 2) for k, v in vif.items()})

    run_label = f"predictand={name}; pooled: " + ", ".join(reg.RUNS)
    for set_def in reg.PREDICTOR_SETS:
        names = set_def["predictors"]
        fit = reg.fit_grid_ols(predictors[names], response)

        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        pdf = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.pdf")
        nc = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.nc")
        plot_set(fit, set_def, run_label, pdf, predictand)
        fit.to_netcdf(nc)
        print(
            f"[{name}] set {set_def['number']} ({labels}): nobs={fit.attrs['nobs']} "
            f"-> {os.path.relpath(pdf, OUT_BASE)}"
        )

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(CAVEATS)


def main():
    for name in PREDICTAND_NAMES:
        run_for_predictand(name)
    print(f"\nDone. Outputs in {OUT_BASE}/<predictand>/")


if __name__ == "__main__":
    main()
