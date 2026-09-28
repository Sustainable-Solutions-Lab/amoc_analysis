"""PDF book of 3 x 3 CO2 x hosing map grids for every analysis variable.

Each variable in ``data_loader.VARIABLES`` gets four pages (see
``output.CASE_GRID_PAGES``), each a 3 x 3 grid of time-mean maps: rows 1x, 2x,
4xCO2 (top to bottom), columns -0.3, 0, +0.3 Sv hosing (left to right).

  1. raw field
  2. minus piControl (1xCO2, 0 Sv)
  3. minus the 1xCO2 run at the same hosing   -> effect of CO2
  4. minus the no-hosing run at the same CO2  -> effect of hosing

Maps are means over 2101-2150, the last 50 years common to all nine runs.
By default the filled fields are embedded as raster images to keep the PDF small;
``--vector`` makes them vector graphics.

    python scripts/plot_case_grid_book.py [--vector] [--variables tas pr ...]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
from output import plot_case_grid_book

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "case_grid")
FIRST_YEAR, LAST_YEAR = 2101, 2150


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--vector", action="store_true",
                        help="draw map fields as vector graphics (large PDF)")
    parser.add_argument("--variables", nargs="+", default=list(dl.VARIABLES),
                        choices=list(dl.VARIABLES), metavar="VAR",
                        help="variables to include (default: all)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_vector" if args.vector else ""
    out_path = os.path.join(OUT_DIR, f"case_grid_{FIRST_YEAR}-{LAST_YEAR}{suffix}.pdf")
    with PdfPages(out_path) as pdf:
        for var in args.variables:
            print(f"  {var}")
            plot_case_grid_book(dl.case_grid_time_mean(var, FIRST_YEAR, LAST_YEAR),
                                pdf, rasterized=not args.vector)
    print(f"wrote {out_path}  ({4 * len(args.variables)} pages)")


if __name__ == "__main__":
    main()
