"""PDF books of 3 x 3 CO2 x hosing map grids, one book per analysis variable.

Each variable gets its own book, ``data/output/case_grid/<var>_2101-2150.pdf``,
with four pages (see
``output.CASE_GRID_PAGES``), each a 3 x 3 grid of time-mean maps: rows 1x, 2x,
4xCO2 (top to bottom), columns -0.3, 0, +0.3 Sv hosing (left to right).

  1. raw field
  2. minus piControl (1xCO2, 0 Sv)
  3. minus the 1xCO2 run at the same hosing   -> effect of CO2
  4. minus the no-hosing run at the same CO2  -> effect of hosing

Maps are means over 2101-2150, the last 50 years common to all nine runs.
By default the filled fields are embedded as raster images to keep the PDF small;
``--vector`` makes them vector graphics. ``--variables`` takes set names from
``data_loader.VARIABLE_SETS`` (``minimal`` = tas, pr; ``key``; ``all``, the
default) and/or individual variable names.

    python scripts/plot_case_grid_book.py [--vector] [--variables minimal | tas pr ...]
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
    dl.add_variables_argument(parser)
    args = parser.parse_args()
    variables = dl.resolve_variables(args.variables)

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_vector" if args.vector else ""
    for var in variables:
        out_path = os.path.join(OUT_DIR, f"{var}_{FIRST_YEAR}-{LAST_YEAR}{suffix}.pdf")
        with PdfPages(out_path) as pdf:
            plot_case_grid_book(dl.case_grid_time_mean(var, FIRST_YEAR, LAST_YEAR),
                                pdf, rasterized=not args.vector)
        print(f"wrote {out_path}  (4 pages)")


if __name__ == "__main__":
    main()
