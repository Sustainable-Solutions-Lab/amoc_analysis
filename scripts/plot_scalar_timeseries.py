"""Time series of the scalar regression predictors, per simulation.

One panel per simulation; Tglob and ΔT_NS as anomalies from their pooled means on
the left axis (K), AMOC strength in absolute Sv on the right axis. Annual values
are thin lines and the decadal (10-year block-mean) values are overlaid as marked
lines on the same plot. Uses the same pooled, AMOC-complete sample as the
regressions (annual = block=None, decadal = block=10).

    python scripts/plot_scalar_timeseries.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import plot_scalar_timeseries

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    annual, _ = reg.build_pooled()  # predictand irrelevant for the scalar predictors
    decadal, _ = reg.build_pooled(block=10)
    out_path = os.path.join(OUT_DIR, "predictor_timeseries.pdf")
    plot_scalar_timeseries(
        annual, decadal,
        "Scalar predictor indices over time — annual (thin) + decadal means "
        "(markers)\nTglob, ΔT_NS as anomalies from pooled mean; AMOC absolute",
        out_path,
    )
    print(f"wrote {out_path}  (annual n={annual.sizes['sample']}, "
          f"decadal n={decadal.sizes['sample']})")


if __name__ == "__main__":
    main()
