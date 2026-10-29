"""Decompose the 2x and 4xCO2 responses into CO2, AMOC and nonlinear parts.

For each analysis variable, one OLS per grid cell is fit over the nine runs'
2101-2150 means (the last 50 years with AMOC in every run; one sample per run,
n = 9, df = 5):

    field = a + b*C + c*A + d*C*A,   C = log2(CO2 multiple),   A = T(AMOC) - T(AMOC(1xCO2))

with AMOC the run's 2101-2150 mean at 26.5N (Sv) and T its Box-Cox transform of
power ``--amoc-power`` (``regression.amoc_transform``; default 0 = ln AMOC, 1 =
linear). Both predictors are zero at the 1xCO2 control (``regression.fit_co2_amoc``). The decomposition page is a 5 x 2 map
grid: columns 2xCO2 and 4xCO2 (no hosing), each with dC = log2(n) and dA = that
run's transformed AMOC minus the control's; rows

  1. a           fitted 1xCO2 field (same in both columns)
  2. b * dC      CO2 effect at the control's AMOC
  3. c * dA      effect of the run's AMOC change, at 1xCO2
  4. d * dC * dA nonlinear (interaction) effect
  5. sum of rows 1-4: the reconstructed (fitted) nxCO2 field

Rows 1 and 5 share one absolute color scale; rows 2-4 share one symmetric scale
and are contoured at the colorbar ticks where their coefficient's p <= 0.05
(color alone: not significant). The check page is a 3 x 3
grid: rows 1x, 2x, 4xCO2 (no hosing), columns simulated, reconstructed, and
simulated minus reconstructed. Each map page is followed by its zonal-statistics
page. Outputs, in ``data/output/co2_amoc_decomposition/``:

- ``<var>_2101-2150_<T>.pdf`` -- the decomposition and check map pages, each
  followed by its zonal-statistics page (4 pages).
- ``<var>_<T>_fit.nc`` -- the per-cell fit (coef, se, tstat, pvalue, r2).

``<T>`` names the AMOC transform (``amoc_file_tag``): ``lnAMOC``, ``AMOC`` (linear)
or ``AMOCpow<p>``, so runs with different powers sit side by side.

``--variables`` takes set names from ``data_loader.VARIABLE_SETS`` (``minimal`` =
tas, pr; ``key``; ``all``, the default) and/or variable names.

    python scripts/plot_co2_amoc_decomposition.py [--amoc-power P]
        [--variables minimal | tas pr_minus_evap ...]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import regression as reg
from output import (
    SIGNIFICANCE_P,
    label_with_units,
    plot_co2_amoc_decomposition,
    plot_co2_amoc_reconstruction_check,
    value_with_units,
)

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "co2_amoc_decomposition")
FIRST_YEAR, LAST_YEAR = 2101, 2150


def amoc_file_tag(power):
    """File-name tag for the AMOC transform: ``lnAMOC``, ``AMOC`` or ``AMOCpow<p>``."""
    return {0: "lnAMOC", 1: "AMOC"}.get(power, f"AMOCpow{power:g}")


def run_for_variable(name, amoc_grid, amoc_power):
    """Fit, decompose and plot variable ``name``; write its PDF and fit NetCDF."""
    grid = dl.case_grid_time_mean(name, FIRST_YEAR, LAST_YEAR)
    fit = reg.fit_co2_amoc(grid, amoc_grid, amoc_power)
    transform = fit.attrs["amoc_transform"]
    tag = amoc_file_tag(amoc_power)
    decomposition = reg.co2_amoc_decomposition(fit, amoc_grid)
    units = grid.attrs["units"]
    reference = fit.attrs["amoc_reference_Sv"]
    actual = grid.sel(hosing=0.0, drop=True)
    cmap = reg.PREDICTANDS[name]["cmap"]
    header = (
        f"{name}: {label_with_units(grid.attrs['long_name'], units)}, CESM1.2 means "
        f"{grid.attrs['time_mean']}\n"
        f"per-cell OLS over the 9 runs (n = {fit.attrs['nobs']}, df = {fit.attrs['df']}): "
        f"{name} = a + b·log2(CO2) + c·A + d·log2(CO2)·A,  "
        f"A = {transform} − {transform} at 1xCO2 (AMOC = {reference:.2f} Sv)"
    )
    print(f"\n[{name}] n = {fit.attrs['nobs']}, df = {fit.attrs['df']}")
    for co2 in decomposition["co2"].values:
        for param in fit["param"].values:
            field = decomposition["contribution"].sel(co2=co2, param=param)
            print(f"[{name}] {co2}xCO2 {param}: global mean "
                  + value_with_units(f"{float(dl.global_mean(field)):.4g}", units))
        residual = actual.sel(co2=co2) - decomposition["fitted"].sel(co2=co2)
        print(f"[{name}] {co2}xCO2 simulated - reconstructed: global mean "
              + value_with_units(f"{float(dl.global_mean(residual)):.4g}", units))

    pdf_path = os.path.join(OUT_DIR, f"{name}_{grid.attrs['time_mean']}_{tag}.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_co2_amoc_decomposition(
            decomposition, name, units, cmap,
            f"{header}\ndecomposition of the nxCO2 (no hosing) response; "
            f"contours: coefficient p ≤ {SIGNIFICANCE_P}; color only: not significant", pdf)
        plot_co2_amoc_reconstruction_check(
            actual, decomposition["fitted"], name, units, cmap,
            f"{header}\nno-hosing runs: simulated vs reconstructed from the regression", pdf)
    nc_path = os.path.join(OUT_DIR, f"{name}_{tag}_fit.nc")
    fit.to_netcdf(nc_path)
    print(f"[{name}] wrote {pdf_path}\n[{name}] wrote {nc_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--amoc-power", type=float, default=reg.DEFAULT_AMOC_POWER, metavar="P",
        help="Box-Cox power of the AMOC transform (0 = ln AMOC, the default; 1 = linear)",
    )
    dl.add_variables_argument(parser)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    amoc_grid = dl.case_grid_amoc_mean(FIRST_YEAR, LAST_YEAR)
    print(f"AMOC means {FIRST_YEAR}-{LAST_YEAR} (Sv):\n{amoc_grid.to_pandas().round(2)}")
    for name in dl.resolve_variables(args.variables):
        run_for_variable(name, amoc_grid, args.amoc_power)


if __name__ == "__main__":
    main()
