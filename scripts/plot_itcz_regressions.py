"""Plots for the ITCZ-latitude analysis: time series and regression figures.

For each tropical-band centroid (20S-20N, 30S-30N) writes a time series to
``data/output/itcz/{band20,band30}/`` and the regression figures for both the
annual sample (same directory) and the decadal10 sample (``decadal10/`` subdir):

- ``itcz_timeseries.pdf`` -- the precip centroid latitude per simulation, annual
  (thin) with the decadal (10-year block-mean) values overlaid (band level only).
- ``itcz_scatter.pdf`` -- centroid latitude vs each single predictor (Tglob, ΔT_NS,
  AMOC), with the OLS line, 95% confidence band, and slope ± SE / R² / p annotated.
- ``itcz_predicted_vs_observed.pdf`` -- predicted vs observed centroid latitude for
  the multi-predictor sets (5 & 10 by default; 5, 6, 10 with ``--all-sets``), with
  the 1:1 line and R².
- ``itcz_coefficients.pdf`` -- partial-slope (coef ± SE) bar charts for the same sets.

    python scripts/plot_itcz_regressions.py [--all-sets]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import (
    plot_itcz_coefficients,
    plot_itcz_predicted_vs_observed,
    plot_itcz_scatter,
    plot_itcz_timeseries,
)

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "itcz")
SINGLE_VARS = ["tas_global_mean", "tas_interhemispheric_diff", "amoc_strength"]
MULTI_SETS = [5, 6, 10]  # multi-predictor sets to visualize jointly

RESPONSES = [
    {"tag": "band20", "var": "precip_centroid_lat_20", "label": "20S-20N"},
    {"tag": "band30", "var": "precip_centroid_lat_30", "label": "30S-30N"},
]

# Smoothing variants for the regression figures: annual at the band level,
# decadal10 in its own subdir (mirrors the run_itcz_regressions.py layout).
SMOOTHINGS = [
    {"tag": "annual", "block": None, "subdir": ""},
    {"tag": "decadal10", "block": 10, "subdir": "decadal10"},
]


def make_regression_figures(predictors, response, out_dir, band, sample_tag, all_sets):
    """Write the scatter, predicted-vs-observed, and coefficient figures.

    ``predictors``/``response`` are a pooled sample (annual or decadal); ``band``
    and ``sample_tag`` label the figures ("20S-20N", "annual"/"decadal"). The
    multi-predictor panels cover the selected sets intersected with ``MULTI_SETS``.
    """
    os.makedirs(out_dir, exist_ok=True)

    fits = {v: reg.fit_scalar_ols(predictors[[v]], response) for v in SINGLE_VARS}
    sc_path = os.path.join(out_dir, "itcz_scatter.pdf")
    plot_itcz_scatter(
        predictors, response, fits, SINGLE_VARS,
        f"ITCZ latitude (precip centroid, {band}) vs scalar predictors "
        f"(pooled {sample_tag} sample) — OLS line, 95% CI band",
        sc_path,
    )
    print(f"wrote {sc_path}  (n={response.sizes['sample']})")

    # Multi-predictor joint fits (the selected sets within MULTI_SETS = {5, 6, 10};
    # 5 & 10 by default), with the orthogonalized + quadratic columns on the sample.
    full_p = reg.add_quadratic_columns(reg.add_orthogonalized_columns(predictors))
    run_of = full_p["run"].values
    observed = response.values
    pvo_panels, coef_panels = [], []
    for set_def in reg.select_predictor_sets(all_sets):
        if set_def["number"] not in MULTI_SETS:
            continue
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

    pvo_path = os.path.join(out_dir, "itcz_predicted_vs_observed.pdf")
    plot_itcz_predicted_vs_observed(
        observed, run_of, pvo_panels,
        f"ITCZ latitude (precip centroid, {band}): predicted vs observed "
        f"(pooled {sample_tag} sample)",
        pvo_path,
    )
    print(f"wrote {pvo_path}")

    coef_path = os.path.join(out_dir, "itcz_coefficients.pdf")
    plot_itcz_coefficients(
        coef_panels,
        f"ITCZ latitude (precip centroid, {band}): partial slopes ± SE "
        f"(pooled {sample_tag} sample)",
        coef_path,
    )
    print(f"wrote {coef_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="visualize all multi-predictor sets (5, 6, 10); default: sets 5 & 10",
    )
    args = parser.parse_args()
    for response in RESPONSES:
        band_dir = os.path.join(OUT_BASE, response["tag"])
        os.makedirs(band_dir, exist_ok=True)
        band = response["label"]

        annual_r = reg.build_pooled_scalar(response["var"])[1]
        decadal_r = reg.build_pooled_scalar(response["var"], block=10)[1]
        ts_path = os.path.join(band_dir, "itcz_timeseries.pdf")
        plot_itcz_timeseries(
            annual_r, decadal_r,
            f"ITCZ latitude (precip centroid, {band}) — annual (thin) + decadal "
            "means (markers)",
            ts_path,
        )
        print(f"wrote {ts_path}  (annual n={annual_r.sizes['sample']}, "
              f"decadal n={decadal_r.sizes['sample']})")

        for smoothing in SMOOTHINGS:
            predictors, resp = reg.build_pooled_scalar(
                response["var"], block=smoothing["block"]
            )
            out_dir = os.path.join(band_dir, smoothing["subdir"])
            make_regression_figures(
                predictors, resp, out_dir, band, smoothing["tag"], args.all_sets
            )


if __name__ == "__main__":
    main()
