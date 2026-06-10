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
PREDICTAND_NAMES = ["tas", "prc"]

# Smoothing variants. "annual" is the interannual analysis (existing outputs,
# under <predictand>/); "decadal10" low-passes to slower timescales via 10-year
# block means and writes to <predictand>/decadal10/.
SMOOTHINGS = [
    {"tag": "annual", "block": None, "subdir": ""},
    {"tag": "decadal10", "block": 10, "subdir": "decadal10"},
]

CAVEATS = """Regression outputs: pooled per-grid-point OLS of a gridded predictand.

- One regression per grid cell; the years of all four simulations
  (historical-ssp585, abrupt-4xCO2, piControl, u03-hos) are POOLED into a single
  fit with a common intercept and no per-run fixed effects. Pooling exploits
  between-run differences in the index relationships to reduce collinearity.
- Common sample: years with all predictors (Tglob, dT_NS, AMOC) present, per run
  (= AMOC-present years).
- Two smoothing variants: 'annual' (interannual, this directory) and 'decadal10'
  (slow timescales, decadal10/ subdir) = non-overlapping 10-year block means
  applied per run/segment to BOTH predictors and predictand before pooling.
- p-values are nominal OLS (independent residuals). For 'annual' the within-run
  autocorrelation of annual data makes them OPTIMISTIC. The 'decadal10' block
  means decimate to ~independent decadal samples, so its degrees of freedom (and
  thus p-values) are far more trustworthy -- at the cost of n (~50 vs 500).
- Set 6 (three predictors) retains high collinearity (annual VIF ~ 22); its
  partial coefficients are poorly constrained. See per-set VIF printed at build.
- Coefficient units are [predictand units] / [predictor units] (predictor units:
  Tglob, dT_NS in K; AMOC in Sv).
- The 'prc' predictand is CONVECTIVE precipitation uniformly for all runs. For
  historical-ssp585 the r1->r4 member splice (historical r1i1p1f1, ssp585 r4i1p1f1)
  is identical for prc, tas, and AMOC, so predictand and predictors stay consistent.
"""


def run_for_predictand(name, smoothing):
    predictand = reg.PREDICTANDS[name]
    tag = smoothing["tag"]
    out_dir = os.path.join(OUT_BASE, name, smoothing["subdir"])
    os.makedirs(out_dir, exist_ok=True)

    predictors, response = reg.build_pooled(predictand=predictand, block=smoothing["block"])
    predictors = reg.add_orthogonalized_columns(predictors)  # for sets 7-8
    predictors = reg.add_quadratic_columns(predictors)  # for set 9
    per_run = predictors["run"].to_series().value_counts().to_dict()
    vif = reg.variance_inflation_factors(predictors[reg.PREDICTOR_UNION])
    print(f"\n[{name}/{tag}] pooled sample: n={predictors.sizes['sample']}  per-run={per_run}")
    print(f"[{name}/{tag}] VIF (3-predictor union):", {k: round(v, 2) for k, v in vif.items()})

    run_label = f"predictand={name}; smoothing={tag}; pooled: " + ", ".join(reg.RUNS)
    for set_def in reg.PREDICTOR_SETS:
        names = set_def["predictors"]
        fit = reg.fit_grid_ols(predictors[names], response)

        centering = reg.centering_means_for_set(predictors, names)
        for tag, (mean, units) in centering.items():
            fit.attrs[f"centering_mean_{tag}_{units}"] = mean
        if centering:
            fit.attrs["centering_note"] = (
                "Centered (q_) predictors were demeaned by these pooled means before "
                "fitting; subtract them from raw values before applying the centered "
                "terms. The cross-product (interaction) coefficient is read relative "
                "to these means."
            )

        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        pdf = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.pdf")
        nc = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.nc")
        plot_set(fit, set_def, run_label, pdf, predictand, centering)
        fit.to_netcdf(nc)
        print(
            f"[{name}/{tag}] set {set_def['number']} ({labels}): nobs={fit.attrs['nobs']} "
            f"-> {os.path.relpath(pdf, OUT_BASE)}"
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
