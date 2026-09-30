"""Pooled scalar regressions of the decadal-mean ITCZ latitude on the scalar indices.

The response is the precipitation-mass centroid latitude (the area- and
precip-weighted mean latitude of the zonal-mean precipitation in a tropical band,
an ITCZ-position index that varies continuously through double-ITCZ states),
computed for two bands -- ``precip_centroid_lat_20`` (20S-20N) and
``precip_centroid_lat_30`` (30S-30N) -- read from the per-simulation
``scalars_annual_CESM1_{run}.nc`` files and reduced to decadal means. The
predictors are the same scalar indices used elsewhere (Tglob, dT_NS, AMOC) and the
same predictor sets. By default only sets 5 & 10 are run (pass ``--all-sets`` for
all ten).

For each band x set a closed-form OLS is fit (``regression.fit_scalar_ols``); all
sets' coefficients go in one table per band, ``data/output/itcz/<band>_coef_table.csv``
(``band20``, ``band30``), with a shared caveats ``README.txt``.

    python scripts/run_itcz_regressions.py [--all-sets]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

import data_loader as dl
import regression as reg

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "itcz")

CAVEATS = """ITCZ regressions: pooled OLS of a decadal-mean scalar ITCZ-latitude response.

- Response: precip_centroid_lat = the precipitation-mass centroid latitude (deg N)
  = area- and precip-weighted mean latitude of the zonal-mean precipitation within
  a tropical band: band20 = 20S-20N, band30 = 30S-30N. It integrates over both
  branches of a double ITCZ, so it varies continuously (unlike the bare argmax,
  which jumps between branches).
- The decadal means of all nine simulations (the 3x3 CO2 x hosing matrix) are
  POOLED into a single fit with a common intercept and no per-run fixed effects.
- Sample: per run, the years where all predictors (Tglob, dT_NS, AMOC) AND the
  response are present, reduced to non-overlapping 10-year block means applied to
  BOTH predictors and response (9 runs x 10 decades = 90 pooled samples).
- p-values are nominal OLS (independent residuals). Successive decades of a run
  still drift together toward equilibrium, so they remain somewhat optimistic.
- Coefficient units are deg latitude per predictor unit (Tglob, dT_NS in K;
  AMOC in Sv).
- The precip-centroid source is TOTAL precipitation pr (CAM PRECT) for all runs.
- Files (flat in this directory, one pair per band):
    * <band>_coef_table.csv (scripts/run_itcz_regressions.py) -- every set's
      coef, se, tstat, pvalue, 95% conf_int, r2 and nobs, one row per parameter.
    * <band>_itcz.pdf (scripts/plot_itcz_regressions.py) -- time series (annual
      thin, decadal means marked), scatter against each single predictor with
      the OLS line and 95% band, predicted vs observed and partial slopes +/- SE
      for the multi-predictor sets.
"""


def run_for(response, all_sets):
    predictors, resp = reg.build_pooled_scalar(response["var"], block=reg.DECADAL_BLOCK)
    predictors = reg.add_orthogonalized_columns(predictors)  # for sets 7-8
    predictors = reg.add_quadratic_columns(predictors)  # for sets 9-10
    per_run = predictors["run"].to_series().value_counts().to_dict()
    vif = reg.variance_inflation_factors(predictors[reg.PREDICTOR_UNION])
    head = f"itcz/{response['tag']}"
    print(f"\n[{head}] pooled sample: n={predictors.sizes['sample']}  per-run={per_run}")
    print(f"[{head}] VIF (3-predictor union):", {k: round(v, 2) for k, v in vif.items()})

    rows = []
    for set_def in reg.select_predictor_sets(all_sets):
        names = set_def["predictors"]
        fit = reg.fit_scalar_ols(predictors[names], resp)
        labels = "-".join(reg.PREDICTORS[p]["tag"] for p in names)
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

    table_path = os.path.join(OUT_DIR, f"{response['tag']}_coef_table.csv")
    pd.DataFrame(rows).to_csv(table_path, index=False)
    print(f"[{head}] wrote {table_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="fit all ten predictor sets (default: only sets 5 & 10)",
    )
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(CAVEATS)
    for response in reg.ITCZ_RESPONSES:
        run_for(response, args.all_sets)


if __name__ == "__main__":
    main()
