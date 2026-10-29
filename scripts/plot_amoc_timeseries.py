"""Time series of AMOC strength for the nine CESM1.2 cases on one plot.

Uses each case's AMOC years (2051-2150): thin lines through the annual means and
bold lines through the 10-year block means, with the project's case line
conventions (CO2 level -> line style, hosing -> color).

    python scripts/plot_amoc_timeseries.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import plot_amoc_timeseries

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    annual, _ = reg.build_pooled()  # predictand is irrelevant for the scalars
    decadal, _ = reg.build_pooled(block=reg.DECADAL_BLOCK)
    out_path = os.path.join(OUT_DIR, "amoc_timeseries.pdf")
    plot_amoc_timeseries(annual, decadal, out_path)
    print(f"wrote {out_path}  (annual n={annual.sizes['sample']}, "
          f"decadal n={decadal.sizes['sample']})")


if __name__ == "__main__":
    main()
