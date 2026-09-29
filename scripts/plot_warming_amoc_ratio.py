"""Map the ratio of two per-grid-point response coefficients, per predictand.

From each set-5 pooled regression ``Y ~ Tglob + AMOC`` (Y = tas, prc, or pr) this
builds three maps:

  A = ∂Y/∂Tglob                  local Y change per K of global-mean warming  [Yunit/K]
  B = ∂Y/∂(AMOC slowdown) = −∂Y/∂AMOC   local Y change per Sv of slowdown      [Yunit/Sv]
  ratio = A / B                                                               [Sv/K]

The predictand units cancel in the ratio, so the ratio is always **Sv/K** -- the
number of Sv of AMOC slowdown that produce the same local change in Y as 1 K of
global-mean warming: small magnitude = strongly AMOC-sensitive grid point, large
= AMOC-insensitive. B uses the slowdown sign convention (positive B = Y increases
when AMOC weakens), i.e. the AMOC coefficient with its sign flipped.

The AMOC coefficient crosses zero across the map, so the ratio has singularities
where the denominator vanishes. Per the chosen convention no points are masked;
the diverging color scale is clipped at a fixed ±RATIO_BOUND Sv/K so the
singularities saturate rather than dominate. Negative/positive integer contours
are drawn at ±1..±5 Sv/K.

By default every predictand × smoothing variant is processed in one run:

    python scripts/plot_warming_amoc_ratio.py
    python scripts/plot_warming_amoc_ratio.py --predictand tas --smoothing decadal10

Writes ``ratio_warming_over_slowdown_set5.{nc,pdf}`` to each regression output dir.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl
import regression as reg
from output import DATA_CRS, PROJECTION, centered_lon, draw_coastlines, plot_coefficient_map

import matplotlib.pyplot as plt

OUT_BASE = os.path.join(dl._REPO_ROOT, "data", "output", "regression")

PREDICTANDS = ["tas", "prc", "pr"]
SMOOTHINGS = ["", "decadal10"]  # "" = annual variant (predictand root dir)

# Color-scale half-range for the ratio panel (Sv/K). The ratio is Sv of slowdown
# per K of global warming; a realistic slowdown is well under ~10 Sv per K (AMOC
# will not weaken 15 Sv by 1.5 K), so ratios beyond ±10 saturate at the scale ends.
RATIO_BOUND = 10.0
# Diverging colormap for the ratio panel -- fixed across predictands (the ratio is
# Sv/K regardless of Y) so the ratio maps stay directly comparable.
RATIO_CMAP = "RdBu_r"
RATIO_LEVELS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def build_ratio(coef_path):
    """Return (A, B, ratio) DataArrays plus A/B p-values from a set-5 coef file."""
    ds = xr.open_dataset(coef_path)
    a = ds["coef"].sel(param="tas_global_mean")           # ∂Y/∂Tglob    [Yunit/K]
    a_p = ds["pvalue"].sel(param="tas_global_mean")
    b = -ds["coef"].sel(param="amoc_strength")            # ∂Y/∂slowdown [Yunit/Sv]
    b_p = ds["pvalue"].sel(param="amoc_strength")         # sign flip leaves p unchanged
    ratio = (a / b).rename("ratio_warming_over_slowdown")
    return a, a_p, b, b_p, ratio


def plot(a, a_p, b, b_p, ratio, predictand, out_pdf):
    """Three-panel figure: warming response, slowdown response, and their ratio."""
    label, units, cmap = predictand["label"], predictand["units"], predictand["cmap"]
    fig, axes = plt.subplots(
        3, 1, figsize=(9, 12), subplot_kw={"projection": PROJECTION},
    )
    plot_coefficient_map(
        a, a_p, title=f"∂{label}/∂Tglob  (warming response)",
        units=f"({units}) / K", ax=axes[0], cmap=cmap,
    )
    plot_coefficient_map(
        b, b_p, title=f"∂{label}/∂(AMOC slowdown)  (slowdown response)",
        units=f"({units}) / Sv", ax=axes[1], cmap=cmap,
    )
    # Ratio panel: no significance stippling; color scale fixed at ±RATIO_BOUND so a
    # realistic slowdown-per-K is on scale and AMOC-insensitive cells saturate.
    bound = RATIO_BOUND
    ax = axes[2]
    ratio = centered_lon(ratio)
    mesh = ax.pcolormesh(
        ratio["lon"], ratio["lat"], ratio, cmap=RATIO_CMAP,
        vmin=-bound, vmax=bound, shading="auto", transform=DATA_CRS,
    )
    # Contour lines at ±1..±5 Sv/K (where a small slowdown matches 1 K of warming).
    cs = ax.contour(
        ratio["lon"], ratio["lat"], ratio, levels=RATIO_LEVELS,
        colors="black", linewidths=0.6, transform=DATA_CRS,
    )
    ax.clabel(cs, fmt="%d", fontsize=6)
    draw_coastlines(ax)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title("ratio = warming response / slowdown response", fontsize=10)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
    cbar.set_label("Sv per K  (Sv of slowdown ≡ 1 K global warming)")

    fig.suptitle(
        f"Set 5: {label} ~ Tglob + AMOC  |  ratio of per-grid {label} response coefficients\n"
        f"ratio color scale saturated at ±{bound:.3g} Sv/K (realistic slowdown per K)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def process(predictand_name, smoothing):
    """Build and write the ratio NetCDF + figure for one predictand/smoothing."""
    out_dir = os.path.join(OUT_BASE, predictand_name, smoothing)
    coef_path = os.path.join(out_dir, "coef_set5_Tglob-AMOC.nc")
    if not os.path.exists(coef_path):
        print(f"skip {predictand_name}/{smoothing or 'annual'}: no {os.path.basename(coef_path)}")
        return
    a, a_p, b, b_p, ratio = build_ratio(coef_path)

    out_nc = os.path.join(out_dir, "ratio_warming_over_slowdown_set5.nc")
    out_pdf = os.path.join(out_dir, "ratio_warming_over_slowdown_set5.pdf")
    ratio.attrs = {
        "long_name": "warming response over AMOC-slowdown response",
        "units": "Sv K-1",
        "description": ("(dY/dTglob) / (-dY/dAMOC): Sv of AMOC slowdown producing "
                        "the local change in Y of 1 K global-mean warming"),
        "predictand": predictand_name,
        "source": os.path.basename(coef_path),
    }
    ratio.to_dataset().to_netcdf(out_nc)
    plot(a, a_p, b, b_p, ratio, reg.PREDICTANDS[predictand_name], out_pdf)
    print(f"wrote {out_nc}\nwrote {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictand", choices=PREDICTANDS, default=None,
                        help="default: all of %s" % PREDICTANDS)
    parser.add_argument("--smoothing", choices=SMOOTHINGS, default=None,
                        help="'' = annual, 'decadal10' = decadal; default: both")
    args = parser.parse_args()

    predictands = [args.predictand] if args.predictand else PREDICTANDS
    smoothings = [args.smoothing] if args.smoothing is not None else SMOOTHINGS
    for predictand_name, smoothing in itertools.product(predictands, smoothings):
        process(predictand_name, smoothing)


if __name__ == "__main__":
    main()
