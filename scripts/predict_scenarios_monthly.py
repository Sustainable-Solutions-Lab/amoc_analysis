"""Predicted monthly-mean field changes for the end-of-century scenarios.

The monthly analog of ``scripts/predict_scenarios.py``: uses the per-calendar-month
decadal regressions in ``data/output/regression_monthly/<predictand>/decadal10/`` to
map the predicted change in each calendar month's gridded tas/prc/pr field for the
SSP5-8.5 scenarios, relative to the preindustrial baseline. The scenario states
(Tglob, AMOC), the per-predictand baseline (pr uses the historical 1850-1900 mean),
and the ``coef . (predictor(X) - predictor(R))`` formula are reused unchanged from
predict_scenarios.py; the only difference is that ``coef`` (and, for set 10, the
centering means) carry a ``month`` dimension, so each predicted change is
(month, lat, lon) and is drawn as a 3x4 twelve-month grid.

Map fields are rasterized inside the PDF by default; ``--vector`` writes full vector.

    python scripts/predict_scenarios_monthly.py [--vector]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import predict_scenarios as ps  # reuse COND, SCENARIOS, conditions_for, predicted_change
import regression as reg
from output import MONTH_NAMES, PROJECTION, plot_coefficient_map

REG_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression_monthly")
OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "scenarios_monthly")


def run_for_predictand(name, vector):
    predictand = reg.PREDICTANDS[name]
    units, cmap = predictand["units"], predictand.get("cmap", "RdBu_r")
    cond = ps.conditions_for(name)
    sets = sorted(ps.SET_FILES)
    dsets = {
        s: xr.open_dataset(os.path.join(REG_BASE, name, "decadal10", ps.SET_FILES[s]))
        for s in sets
    }
    coefs = {s: dsets[s]["coef"] for s in sets}
    # Per-month set-10 centering means (DataArrays on 'month'); broadcast in predicted_change.
    mT = dsets[10]["centering_mean_Tglob"]
    mA = dsets[10]["centering_mean_AMOC"]
    print(f"[{name}] baseline (piControl ref) Tglob={cond['piControl'][0]:.3f} K, "
          f"AMOC={cond['piControl'][1]:.3f} Sv; set-10 centering month-mean "
          f"Tglob={float(mT.mean()):.3f} K, AMOC={float(mA.mean()):.3f} Sv")

    out_path = os.path.join(OUT_DIR, f"predicted_change_{name}.pdf")
    with PdfPages(out_path) as pdf:
        for set_num in sets:
            coef = coefs[set_num]
            for title, condX, condR in ps.SCENARIOS:
                change = ps.predicted_change(coef, set_num, condX, condR, cond, mT, mA)
                bound = float(np.nanpercentile(np.abs(change.values), 99))
                fig, axes = plt.subplots(
                    3, 4, figsize=(22, 12), squeeze=False,
                    subplot_kw={"projection": PROJECTION},
                )
                for ax, month in zip(axes.flat, change["month"].values):
                    cm = change.sel(month=month)
                    plot_coefficient_map(
                        cm, xr.zeros_like(cm),  # zeros -> no stippling
                        title=MONTH_NAMES[int(month) - 1],
                        units=f"Δ{name} ({units})",
                        ax=ax, cmap=cmap, bound=bound, rasterized=not vector,
                    )
                fig.suptitle(
                    f"Predicted monthly-mean {name} change (monthly decadal10 "
                    f"regressions)\nset {set_num}: {title}",
                    fontsize=12,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig, dpi=150, bbox_inches="tight")
                plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vector", action="store_true",
        help="write full vector PDFs (default: rasterized map fields, smaller files)",
    )
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in ("tas", "prc", "pr"):
        run_for_predictand(name, args.vector)


if __name__ == "__main__":
    main()
