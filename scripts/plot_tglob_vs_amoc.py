"""x-y line plot of AMOC strength vs global-mean tas for the nine CESM1.2 cases.

Uses each case's AMOC years (2051-2150): thin lines through the annual means and
bold lines through the 10-year block means, with the project's case line
conventions (CO2 level -> line style, hosing -> color).

    python scripts/plot_tglob_vs_amoc.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import plot_tglob_vs_amoc

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "regression")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    annual, _ = reg.build_pooled()  # predictand is irrelevant for the scalars
    decadal, _ = reg.build_pooled(block=reg.DECADAL_BLOCK)
    out_path = os.path.join(OUT_DIR, "tglob_vs_amoc.pdf")
    plot_tglob_vs_amoc(annual, decadal, out_path)
    print(f"wrote {out_path}  (annual n={annual.sizes['sample']}, "
          f"decadal n={decadal.sizes['sample']})")


if __name__ == "__main__":
    main()
