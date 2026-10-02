"""Pooled per-grid-point regressions of decadal-mean gridded fields on scalar indices.

Builds one pooled decadal-mean sample (10-year block means of the years with all
predictors present, across the nine CESM1.2 simulations), then for each selected
predictor set fits a per-grid-point OLS. Outputs, flat in ``data/output/regression/``:

- ``<var>_coef.pdf`` -- per set, a page of stippled coefficient maps followed by a
  page of their zonal statistics over longitude.
- ``<var>_coef.nc`` -- each set's fit in its own NetCDF group (``set5``, ``set10``,
  ...; ``regression.set_group``).
- ``README.txt`` -- caveats (shared by all variables).

By default only sets 5 & 10 are run (pass ``--all-sets`` for all ten).
``--variables`` picks the predictands: set names from ``data_loader.VARIABLE_SETS``
(``minimal`` = tas, pr; ``key``; ``all``, the default) and/or variable names.

    python scripts/run_regressions.py [--all-sets] [--variables minimal | tas pr ...]
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from matplotlib.backends.backend_pdf import PdfPages

from output import plot_set

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")

CAVEATS = """Regression outputs: pooled per-grid-point OLS of decadal-mean gridded fields.

- Model: CESM1.2 (B1850CN, f19g16) NAHosMIP runs, a 3x3 matrix of CO2 level
  (1x, 2x, 4x) x North Atlantic hosing (-0.3, 0, +0.3 Sv); case names
  [124]xCO2[_m03Sv|_p03Sv].
- One regression per grid cell; the decadal means of all nine runs are POOLED into
  a single fit with a common intercept and no per-run fixed effects.
- Sample: per run, the years with all predictors (Tglob, dT_NS, AMOC) present =
  the AMOC years 2051-2150, reduced to non-overlapping 10-year block means applied
  to BOTH predictors and predictand (9 runs x 10 decades = 90 pooled samples).
- p-values are nominal OLS (independent residuals). Decadal means are far less
  autocorrelated than annual values, but successive decades of a run still drift
  together toward equilibrium, so p-values remain somewhat optimistic.
- Coefficient units are [predictand units] / [predictor units] (predictor units:
  Tglob, dT_NS in K; AMOC in Sv).
- Files (flat in this directory, one pair per variable):
    * <var>_coef.pdf -- per predictor set, a page of stippled coefficient maps
      (p > 0.05 hatched) followed by a page of their zonal statistics over
      longitude (mean, median, 5-95% band, min/max vs sine of latitude; all cells).
    * <var>_coef.nc -- one NetCDF group per set (set5, set10, ...) holding coef, se,
      tstat, pvalue on (param, lat, lon) and r2 on (lat, lon); set 10's group
      attributes give the centering means to subtract before applying its
      centered (q_) terms. Read with xr.open_dataset(path, group="set10").
- Predictands: every variable in data_loader.VARIABLES (all CAM fields in the
  input files plus derived fields such as pr_minus_evap); each variable's units
  and definition are in its NetCDF attributes. 'prc' is CONVECTIVE precipitation
  (CAM PRECC); 'pr' is TOTAL precipitation (CAM PRECT); water fluxes are
  mm/day.
"""


def run_for_predictand(name, all_sets):
    """Fit every selected set for predictand ``name``; write the coefficient-map book
    (one page per set) and all fits as groups of one NetCDF file."""
    predictand = reg.PREDICTANDS[name]
    predictors, response = reg.build_pooled(predictand=predictand, block=reg.DECADAL_BLOCK)
    predictors = reg.add_orthogonalized_columns(predictors)  # for sets 7-8
    predictors = reg.add_quadratic_columns(predictors)  # for set 9
    per_run = predictors["run"].to_series().value_counts().to_dict()
    vif = reg.variance_inflation_factors(predictors[reg.PREDICTOR_UNION])
    print(f"\n[{name}] pooled sample: n={predictors.sizes['sample']}  per-run={per_run}")
    print(f"[{name}] VIF (3-predictor union):", {k: round(v, 2) for k, v in vif.items()})

    run_label = f"predictand={name}; decadal means; pooled: " + ", ".join(reg.RUNS)
    book_path = os.path.join(OUT_DIR, f"{name}_coef.pdf")
    fits = {}
    with PdfPages(book_path) as book:
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
            fit.attrs["predictors"] = ", ".join(names)
            plot_set(fit, set_def, run_label, book, predictand, centering)
            fits[set_def["number"]] = fit
            print(f"[{name}] set {set_def['number']} ({fit.attrs['predictors']}): "
                  f"nobs={fit.attrs['nobs']}")
    nc_path = os.path.join(OUT_DIR, f"{name}_coef.nc")
    reg.write_set_fits(nc_path, fits)
    print(f"[{name}] wrote {book_path}\n[{name}] wrote {nc_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="fit all ten predictor sets (default: only sets 5 & 10)",
    )
    dl.add_variables_argument(parser)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(CAVEATS)
    for name in dl.resolve_variables(args.variables):
        run_for_predictand(name, args.all_sets)


if __name__ == "__main__":
    main()
