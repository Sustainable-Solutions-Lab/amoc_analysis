"""Per-calendar-month pooled regressions of gridded tas/prc/pr on annual indices.

For each calendar month (Jan..Dec) separately, fits the same per-grid-point OLS as
``scripts/run_regressions.py`` but with that month's gridded field as the response
and the *annual* scalar indices (Tglob, dT_NS, AMOC) of the same year as predictors
-- i.e. how each month's field responds to annual warming and AMOC. The 12 monthly
fits are stacked along a ``month`` dimension into one NetCDF and one PDF per
predictand x set. The decadal variant uses the 10-year average of each calendar
month.

By default only sets 5 & 10 and only the decadal10 smoothing are produced; pass
``--all-sets`` for all ten, ``--do-annuals`` to add the annual (interannual) variant.
Coefficient-map fields are rasterized inside the PDF (small files); ``--vector``
writes full vector PDFs instead.

    python scripts/run_monthly_regressions.py [--all-sets] [--do-annuals] [--vector]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import xarray as xr

import data_loader as dl
import regression as reg
from output import plot_set_monthly

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression_monthly")

PREDICTAND_NAMES = ["tas", "prc", "pr"]
MONTHS = list(range(1, 13))

CAVEATS = """Monthly regression outputs: per-calendar-month pooled per-grid-point OLS.

- One regression per (grid cell, calendar month): the response is that month's
  gridded field; predictors are the ANNUAL scalar indices (Tglob, dT_NS, AMOC) of
  the same year. The 12 months are stacked along a 'month' dimension.
- The pooled run set, AMOC-present-year sampling, smoothing (decadal10 = 10-year mean
  of each calendar month), and predictor sets match scripts/run_regressions.py.
- NetCDF dims: coef/se/tstat/pvalue (month, param, lat, lon); r2 (month, lat, lon);
  nobs (month). For the cross-product set 10, centering_mean_{Tglob,AMOC} (month) give
  the per-month offsets to subtract before applying the centered interaction term.
"""


def run_for_predictand(name, smoothing, all_sets, vector):
    predictand = reg.PREDICTANDS[name]
    tag = smoothing["tag"]
    out_dir = os.path.join(OUT_BASE, name, smoothing["subdir"])
    os.makedirs(out_dir, exist_ok=True)
    run_label = f"predictand={name}; smoothing={tag}; pooled: " + ", ".join(predictand["by_run"])

    for set_def in reg.select_predictor_sets(all_sets):
        names = set_def["predictors"]
        fits, centerings = [], []
        for month in MONTHS:
            predictors, response = reg.build_pooled_monthly(
                predictand, month, block=smoothing["block"]
            )
            predictors = reg.add_orthogonalized_columns(predictors)  # sets 7-8
            predictors = reg.add_quadratic_columns(predictors)  # sets 9-10
            fit = reg.fit_grid_ols(predictors[names], response)
            fits.append(fit)
            centerings.append(reg.centering_means_for_set(predictors, names))

        stacked = xr.concat(fits, dim=xr.DataArray(MONTHS, dims="month", name="month"))
        stacked["nobs"] = ("month", [int(f.attrs["nobs"]) for f in fits])
        # Per-month centering means for the centered (q_) terms of this set (empty for raw sets).
        for ctag, (_, units) in (centerings[0] or {}).items():
            stacked[f"centering_mean_{ctag}"] = (
                "month", [c[ctag][0] for c in centerings],
            )
            stacked[f"centering_mean_{ctag}"].attrs["units"] = units
        stacked.attrs = {"predictand": name, "smoothing": tag}

        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        pdf = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.pdf")
        nc = os.path.join(out_dir, f"coef_set{set_def['number']}_{labels}.nc")

        centering_plot = {
            ctag: (float(np.mean([c[ctag][0] for c in centerings])), units)
            for ctag, (_, units) in (centerings[0] or {}).items()
        }
        plot_set_monthly(
            stacked, set_def, run_label, pdf, predictand,
            centering=centering_plot, rasterized=not vector,
        )
        stacked.to_netcdf(nc)
        print(
            f"[{name}/{tag}] set {set_def['number']} ({labels}): "
            f"nobs/month={int(stacked['nobs'].values[0])} -> "
            f"{os.path.relpath(pdf, OUT_BASE)}",
            flush=True,
        )

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
        "--vector", action="store_true",
        help="write full vector PDFs (default: rasterized map fields, smaller files)",
    )
    args = parser.parse_args()
    for name in PREDICTAND_NAMES:
        for smoothing in reg.select_smoothings(args.do_annuals):
            run_for_predictand(name, smoothing, args.all_sets, args.vector)
    print(f"\nDone. Outputs in {OUT_BASE}/<predictand>/[decadal10/]")


if __name__ == "__main__":
    main()
