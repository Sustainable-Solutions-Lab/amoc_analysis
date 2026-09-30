"""Plots for the ITCZ-latitude analysis: time series and decadal-mean regression figures.

For each tropical-band centroid (``band20`` = 20S-20N, ``band30`` = 30S-30N) writes
one book, ``data/output/itcz/<band>_itcz.pdf``, with pages:

1. the precip centroid latitude per simulation, annual (thin) with the decadal
   (10-year block-mean) values overlaid;
2. centroid latitude vs each single predictor (Tglob, ΔT_NS, AMOC) on the pooled
   decadal-mean sample, with the OLS line, 95% confidence band, and slope ± SE /
   R² / p annotated;
3. predicted vs observed centroid latitude for the multi-predictor sets (5 & 10 by
   default; 5, 6, 10 with ``--all-sets``), with the 1:1 line and R²;
4. partial-slope (coef ± SE) bar charts for the same sets.

    python scripts/plot_itcz_regressions.py [--all-sets]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import regression as reg
from output import (
    plot_itcz_coefficients,
    plot_itcz_predicted_vs_observed,
    plot_itcz_scatter,
    plot_itcz_timeseries,
)

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "itcz")
SINGLE_VARS = ["tas_global_mean", "tas_interhemispheric_diff", "amoc_strength"]
MULTI_SETS = [5, 6, 10]  # multi-predictor sets to visualize jointly


def plot_regression_pages(predictors, response, band, all_sets, pdf):
    """Append the scatter, predicted-vs-observed, and coefficient pages to ``pdf``.

    ``predictors``/``response`` are the pooled decadal-mean sample; ``band`` labels
    the figures ("20S-20N"). The multi-predictor pages cover the selected sets
    intersected with ``MULTI_SETS``.
    """
    fits = {v: reg.fit_scalar_ols(predictors[[v]], response) for v in SINGLE_VARS}
    plot_itcz_scatter(
        predictors, response, fits, SINGLE_VARS,
        f"ITCZ latitude (precip centroid, {band}) vs scalar predictors "
        f"(pooled decadal means, n={response.sizes['sample']}) — OLS line, 95% CI band",
        pdf=pdf,
    )

    # Multi-predictor joint fits (the selected sets within MULTI_SETS = {5, 6, 10};
    # 5 & 10 by default), with the orthogonalized + quadratic columns on the sample.
    full_p = reg.add_quadratic_columns(reg.add_orthogonalized_columns(predictors))
    pvo_panels, coef_panels = [], []
    for set_def in (s for s in reg.select_predictor_sets(all_sets) if s["number"] in MULTI_SETS):
        names = set_def["predictors"]
        fit = reg.fit_scalar_ols(full_p[names], response)
        disp = "+".join(reg.PREDICTORS[p]["label"] for p in names)
        label = f"set {set_def['number']}: {disp}"
        pvo_panels.append({
            "label": label,
            "predicted": reg.predict_scalar_ols(fit, full_p[names]),
            "r2": fit.attrs["r2"],
        })
        params = [p for p in fit["param"].values if p != "intercept"]
        coef_panels.append({
            "label": label,
            "names": [reg.PREDICTORS[p]["label"] for p in params],
            "coef": [float(fit["coef"].sel(param=p)) for p in params],
            "se": [float(fit["se"].sel(param=p)) for p in params],
            "pvalue": [float(fit["pvalue"].sel(param=p)) for p in params],
        })

    plot_itcz_predicted_vs_observed(
        response.values, full_p["run"].values, pvo_panels,
        f"ITCZ latitude (precip centroid, {band}): predicted vs observed "
        "(pooled decadal means)",
        pdf=pdf,
    )
    plot_itcz_coefficients(
        coef_panels,
        f"ITCZ latitude (precip centroid, {band}): partial slopes ± SE "
        "(pooled decadal means)",
        pdf=pdf,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="visualize all multi-predictor sets (5, 6, 10); default: sets 5 & 10",
    )
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for response in reg.ITCZ_RESPONSES:
        band = response["label"]
        annual_r = reg.build_pooled_scalar(response["var"])[1]
        predictors, decadal_r = reg.build_pooled_scalar(response["var"], block=reg.DECADAL_BLOCK)
        out_path = os.path.join(OUT_DIR, f"{response['tag']}_itcz.pdf")
        with PdfPages(out_path) as pdf:
            plot_itcz_timeseries(
                annual_r, decadal_r,
                f"ITCZ latitude (precip centroid, {band}) — annual (thin) + decadal "
                "means (markers)",
                pdf=pdf,
            )
            plot_regression_pages(predictors, decadal_r, band, args.all_sets, pdf)
        print(f"wrote {out_path}  (annual n={annual_r.sizes['sample']}, "
              f"decadal n={decadal_r.sizes['sample']})")


if __name__ == "__main__":
    main()
