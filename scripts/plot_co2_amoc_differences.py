"""2 x 2 maps separating the 4xCO2 response with and without AMOC weakening.

For each variable, one page of 2101-2150 mean differences
(``output.plot_co2_amoc_differences``, cases ``data_loader.CO2_AMOC_CASES``):

    4xCO2 - 1xCO2 (total response)        4xCO2 - 4xCO2_m03Sv (AMOC-weakening effect)
    4xCO2_m03Sv - 1xCO2 (CO2 effect)      AMOC effect / CO2 effect

4xCO2_m03Sv holds AMOC near the control's ~20 Sv while 4xCO2 weakens to ~10 Sv.
Each map has a strip on its right with the zonal mean and 5-95% band over
longitude (thin green / blue lines over land / ocean), aligned with the map's
latitudes. Each difference map is contoured at its colorbar ticks, black where that
difference is significant (Welch t-test of the five 10-year means of each run,
p <= 0.05) and light grey where not. The
ratio is masked where the CO2 effect is not significant; its contours show the AMOC
effect's significance. The ratio strip shows ratios of
zonal means, shaded with their 95% confidence interval (delta method on the
10-year means of the zonal means, ``regression.difference_ratio_conf_int``). 2101-2150 is the last 50 years common to the runs (4xCO2 ends in 2150).
One single-page PDF per variable, ``data/output/co2_amoc_diff/<var>_2101-2150.pdf``,
then merged (pages copied, not re-rendered, so memory stays per-variable) into
``data/output/co2_amoc_diff/book_2101-2150.pdf``. ``--variables`` takes set names
from ``data_loader.VARIABLE_SETS`` and/or variable names (default ``all``).

    python scripts/plot_co2_amoc_differences.py [--variables tas | minimal | key | all]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pypdf import PdfWriter

import data_loader as dl
import regression as reg
from output import RATIO_CONFIDENCE, plot_co2_amoc_differences

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "co2_amoc_diff")
FIRST_YEAR, LAST_YEAR = 2101, 2150

# Panel name -> (minuend, subtrahend) case, as named in ``plot_co2_amoc_differences``.
DIFFERENCES = {
    "total": ("4xCO2", "1xCO2"),
    "amoc_effect": ("4xCO2", "4xCO2_m03Sv"),
    "co2_effect": ("4xCO2_m03Sv", "1xCO2"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    dl.add_variables_argument(parser)
    args = parser.parse_args()
    variables = dl.resolve_variables(args.variables)

    os.makedirs(OUT_DIR, exist_ok=True)
    page_paths = []
    for var in variables:
        out_path = os.path.join(OUT_DIR, f"{var}_{FIRST_YEAR}-{LAST_YEAR}.pdf")
        fields = dl.case_annual_fields(var, dl.CO2_AMOC_CASES, FIRST_YEAR, LAST_YEAR)
        runs = {case: fields.sel(case=case) for case in dl.CO2_AMOC_CASES}
        pvalues = {name: reg.difference_pvalue(runs[minuend], runs[subtrahend], reg.DECADAL_BLOCK)
                   for name, (minuend, subtrahend) in DIFFERENCES.items()}
        ratio_conf_int = reg.difference_ratio_conf_int(
            *(runs[case].mean("lon") for case in ("4xCO2", "4xCO2_m03Sv", "1xCO2")),
            reg.DECADAL_BLOCK, RATIO_CONFIDENCE)
        plot_co2_amoc_differences(fields, pvalues, ratio_conf_int, out_path)
        page_paths.append(out_path)
        print(f"wrote {out_path}")

    book_path = os.path.join(OUT_DIR, f"book_{FIRST_YEAR}-{LAST_YEAR}.pdf")
    writer = PdfWriter()
    for path in page_paths:
        writer.append(path)
    writer.write(book_path)
    print(f"wrote {book_path}  ({len(page_paths)} pages)")


if __name__ == "__main__":
    main()
