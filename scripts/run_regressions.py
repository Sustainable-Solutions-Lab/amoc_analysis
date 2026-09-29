"""Pooled per-grid-point regressions of gridded tas on scalar indices.

Builds one pooled sample (years with all predictors present, across the nine CESM1.2
simulations), then for each selected predictor set fits a per-grid-point OLS and
writes the coefficient fields (NetCDF) and a PDF book of stippled coefficient
maps, one page per set (``coef_maps.pdf``), to
``data/output/regression/<predictand>/[decadal10/]``. By default only sets 5 & 10 are run (pass
``--all-sets`` for all ten) and only the decadal10 smoothing (pass ``--do-annuals``
to also run the annual variant). ``--variables`` picks the predictands: a set name
from ``data_loader.VARIABLE_SETS`` (``minimal`` = tas, pr; ``key``; ``all``,
the default) and/or individual variable names.

    python scripts/run_regressions.py [--all-sets] [--do-annuals] [--variables minimal]
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from matplotlib.backends.backend_pdf import PdfPages

from output import plot_set

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression")

# Predictands to analyze: every loader variable (each writes its NetCDF to its own
# subdirectory and one page per set book).
CAVEATS = """Regression outputs: pooled per-grid-point OLS of a gridded predictand.

- Model: CESM1.2 (B1850CN, f19g16) NAHosMIP runs, a 3x3 matrix of CO2 level
  (1x, 2x, 4x) x North Atlantic hosing (-0.3, 0, +0.3 Sv); case names
  [124]xCO2[_m03Sv|_p03Sv].
- One regression per grid cell; the years of all nine runs are POOLED into a
  single fit with a common intercept and no per-run fixed effects.
- Common sample: years with all predictors (Tglob, dT_NS, AMOC) present, per run
  = the AMOC years 2051-2150 (100 per run, 900 pooled).
- Smoothing: 'decadal10' (slow timescales, decadal10/ subdir) = non-overlapping
  10-year block means applied per run to BOTH predictors and predictand before
  pooling (90 pooled blocks), produced by default. The 'annual' (interannual, this
  directory) variant is produced only with --do-annuals.
- p-values are nominal OLS (independent residuals). For 'annual' the within-run
  autocorrelation of annual data makes them OPTIMISTIC. The 'decadal10' block
  means decimate to ~independent decadal samples, so its degrees of freedom (and
  thus p-values) are far more trustworthy.
- Coefficient units are [predictand units] / [predictor units] (predictor units:
  Tglob, dT_NS in K; AMOC in Sv).
- Predictands: every variable in data_loader.VARIABLES (all CAM fields in the
  input files plus derived fields such as pr_minus_evap); each variable's units
  and definition are in its NetCDF attributes. 'prc' is CONVECTIVE precipitation
  (CAM PRECC); 'pr' is TOTAL precipitation (CAM PRECT); water fluxes are
  mm/day.
"""


def set_labels(set_def):
    """File-name tag for a predictor set, e.g. ``Tglob-AMOC``."""
    return "-".join(reg.PREDICTORS[p]["tag"] for p in set_def["predictors"])


def run_for_predictand(name, smoothing, all_sets):
    """Fit every selected set for predictand ``name``; write each fit's NetCDF and
    its maps as one page of the predictand's book, ``coef_maps.pdf``."""
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
    book_path = os.path.join(out_dir, "coef_maps.pdf")
    book = PdfPages(book_path)
    for set_def in reg.select_predictor_sets(all_sets):
        names = set_def["predictors"]
        fit = reg.fit_grid_ols(predictors[names], response)

        centering = reg.centering_means_for_set(predictors, names)
        for base, (mean, units) in centering.items():
            fit.attrs[f"centering_mean_{base}_{units}"] = mean
        if centering:
            fit.attrs["centering_note"] = (
                "Centered (q_) predictors were demeaned by these pooled means before "
                "fitting; subtract them from raw values before applying the centered "
                "terms. The cross-product (interaction) coefficient is read relative "
                "to these means."
            )

        labels = set_labels(set_def)
        nc = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.nc")
        plot_set(fit, set_def, run_label, book, predictand, centering)
        fit.to_netcdf(nc)
        print(
            f"[{name}/{tag}] set {set_def['number']} ({labels}): nobs={fit.attrs['nobs']} "
            f"-> {os.path.relpath(nc, OUT_BASE)}"
        )

    book.close()
    print(f"[{name}/{tag}] wrote {os.path.relpath(book_path, OUT_BASE)}")

    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(CAVEATS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="fit all ten predictor sets (default: only sets 5 & 10)",
    )
    parser.add_argument(
        "--do-annuals", action="store_true",
        help="also run the annual (interannual) variant (default: decadal10 only)",
    )
    parser.add_argument(
        "--variables", nargs="+", default=["all"], choices=list(dl.VARIABLE_SETS),
        metavar="NAME",
        help="predictands: set names (minimal, key, all) and/or variable names "
             "(default: all)",
    )
    args = parser.parse_args()
    predictand_names = dl.resolve_variables(args.variables)
    for smoothing in reg.select_smoothings(args.do_annuals):
        for name in predictand_names:
            run_for_predictand(name, smoothing, args.all_sets)
    print(f"\nDone. Books and NetCDF in {OUT_BASE}/<predictand>/[decadal10/]")


if __name__ == "__main__":
    main()
