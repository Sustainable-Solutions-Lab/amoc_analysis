"""Scatter plots of the pooled regression predictors, colored by simulation.

Four panels in one PDF: AMOC strength and interhemispheric tas difference vs
global-mean tas (top row), then global-mean tas and interhemispheric tas
difference vs AMOC strength (bottom row). Uses the same pooled, AMOC-complete
500-row sample as the regressions.

    python scripts/plot_predictor_scatter.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import plot_predictor_scatter

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    predictors, _ = reg.build_pooled()  # predictand is irrelevant for the scalars
    out_path = os.path.join(OUT_DIR, "predictor_scatter.pdf")
    plot_predictor_scatter(predictors, out_path)
    print(f"wrote {out_path}  (n={predictors.sizes['sample']})")


if __name__ == "__main__":
    main()
