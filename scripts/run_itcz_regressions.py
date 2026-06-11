"""Pooled scalar regressions of the ITCZ latitude on the scalar indices.

The response is the precipitation-mass centroid latitude (the area- and
precip-weighted mean latitude of the zonal-mean precipitation in a tropical band,
an ITCZ-position index that varies continuously through double-ITCZ states),
computed for two bands -- ``precip_centroid_lat_20`` (20S-20N) and
``precip_centroid_lat_30`` (30S-30N) -- read from the per-simulation
``scalars_annual_CESM2_{run}.nc`` files. Each is a single value per
simulation-year. The predictors are the same scalar indices used elsewhere
(Tglob, dT_NS, AMOC) and the same predictor sets and two smoothing variants
(annual, decadal10). By default only sets 5 & 10 are run; pass ``--all-sets`` to
run all ten.

For each band x set a closed-form OLS is fit (``regression.fit_scalar_ols``);
per-set coefficient Datasets (NetCDF) and a combined coefficient table (CSV) are
written to ``data/output/itcz/{band20,band30}/[decadal10/]``.

    python scripts/run_itcz_regressions.py [--all-sets]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

import data_loader as dl
import regression as reg

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "itcz")

# ITCZ response variants: one per tropical band used for the precip centroid.
RESPONSES = [
    {"tag": "band20", "var": "precip_centroid_lat_20", "label": "20S-20N"},
    {"tag": "band30", "var": "precip_centroid_lat_30", "label": "30S-30N"},
]

SMOOTHINGS = [
    {"tag": "annual", "block": None, "subdir": ""},
    {"tag": "decadal10", "block": 10, "subdir": "decadal10"},
]

CAVEATS = """ITCZ regressions: pooled OLS of a scalar ITCZ-latitude response.

- Response: precip_centroid_lat = the precipitation-mass centroid latitude (deg N)
  = area- and precip-weighted mean latitude of the zonal-mean precipitation within
  the tropical band {band}. It integrates over both branches of a double ITCZ, so
  it varies continuously (unlike the bare argmax, which jumps between branches).
- The years of all four simulations (historical-ssp585, abrupt-4xCO2, piControl,
  u03-hos) are POOLED into a single fit with a common intercept and no per-run
  fixed effects, on years where all predictors (Tglob, dT_NS, AMOC) AND the
  response are present.
- Two smoothing variants: 'annual' (interannual, this directory) and 'decadal10'
  (slow timescales, decadal10/ subdir) = non-overlapping 10-year block means
  applied per run/segment to BOTH predictors and response before pooling.
- p-values are nominal OLS (independent residuals). For 'annual', within-run
  autocorrelation makes them OPTIMISTIC; 'decadal10' decimates to ~independent
  decadal samples with far more trustworthy degrees of freedom (at lower n).
- Coefficient units are deg latitude per predictor unit (Tglob, dT_NS in K;
  AMOC in Sv).
- The precip-centroid source is CONVECTIVE prc uniformly for all runs. For
  historical-ssp585 the r1->r4 member splice (historical r1i1p1f1, ssp585 r4i1p1f1)
  is identical for prc, tas, and AMOC, so predictand and predictors stay consistent.
"""


def run_for(response, smoothing, all_sets):
    tag = smoothing["tag"]
    out_dir = os.path.join(OUT_BASE, response["tag"], smoothing["subdir"])
    os.makedirs(out_dir, exist_ok=True)

    predictors, resp = reg.build_pooled_scalar(response["var"], block=smoothing["block"])
    predictors = reg.add_orthogonalized_columns(predictors)  # for sets 7-8
    predictors = reg.add_quadratic_columns(predictors)  # for sets 9-10
    per_run = predictors["run"].to_series().value_counts().to_dict()
    vif = reg.variance_inflation_factors(predictors[reg.PREDICTOR_UNION])
    head = f"itcz/{response['tag']}/{tag}"
    print(f"\n[{head}] pooled sample: n={predictors.sizes['sample']}  per-run={per_run}")
    print(f"[{head}] VIF (3-predictor union):", {k: round(v, 2) for k, v in vif.items()})

    rows = []
    for set_def in reg.select_predictor_sets(all_sets):
        names = set_def["predictors"]
        fit = reg.fit_scalar_ols(predictors[names], resp)

        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
        nc = os.path.join(out_dir, f"itcz_fit_set{set_def['number']}_{labels}.nc")
        fit.to_netcdf(nc)

        for param in fit["param"].values:
            rows.append({
                "set": set_def["number"],
                "predictors": labels,
                "param": param,
                "coef": float(fit["coef"].sel(param=param)),
                "se": float(fit["se"].sel(param=param)),
                "tstat": float(fit["tstat"].sel(param=param)),
                "pvalue": float(fit["pvalue"].sel(param=param)),
                "conf_int_lo": float(fit["conf_int"].sel(param=param, bound="lo")),
                "conf_int_hi": float(fit["conf_int"].sel(param=param, bound="hi")),
                "r2": fit.attrs["r2"],
                "nobs": fit.attrs["nobs"],
            })
        print(f"[{head}] set {set_def['number']} ({labels}): "
              f"R²={fit.attrs['r2']:.3f}, nobs={fit.attrs['nobs']}")

    table_path = os.path.join(out_dir, f"coef_table_{tag}.csv")
    pd.DataFrame(rows).to_csv(table_path, index=False)
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(CAVEATS.format(band=response["label"]))
    print(f"[{head}] wrote {os.path.relpath(table_path, OUT_BASE)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="fit all ten predictor sets (default: only sets 5 & 10)",
    )
    args = parser.parse_args()
    for response in RESPONSES:
        for smoothing in SMOOTHINGS:
            run_for(response, smoothing, args.all_sets)
    print(f"\nDone. Outputs in {OUT_BASE}/{{band20,band30}}/[decadal10/]")


if __name__ == "__main__":
    main()
