"""Plot regression-coefficient maps with significance stippling.

Coefficient maps use a diverging colormap with symmetric bounds (white = 0, per
the project plotting conventions) and Cartopy coastlines. Cells where the
coefficient is not significant at p < 0.05 are stippled with hatching.
"""

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

SIGNIFICANCE_P = 0.05
PROJECTION = ccrs.PlateCarree(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()


def _symmetric_bound(values):
    """Robust symmetric color bound: the 99th percentile of |values| (NaN-safe)."""
    return float(np.nanpercentile(np.abs(values), 99))


def plot_coefficient_map(coef, pvalue, title, units, ax, cmap="RdBu_r"):
    """Draw one coefficient map on a Cartopy ``ax``: filled field + p>0.05 stippling.

    ``coef`` and ``pvalue`` are 2-D (lat, lon) DataArrays. ``cmap`` is a diverging
    colormap (white = 0); use ``RdBu_r`` for temperature (warm = red) and ``RdBu``
    for precipitation (wet = blue). Returns the mappable for colorbar creation.
    """
    lon, lat = coef["lon"], coef["lat"]
    bound = _symmetric_bound(coef.values)
    mesh = ax.pcolormesh(
        lon, lat, coef, cmap=cmap, vmin=-bound, vmax=bound,
        shading="auto", transform=DATA_CRS,
    )
    # Stipple where NOT significant (p > 0.05).
    ax.contourf(
        lon,
        lat,
        (pvalue > SIGNIFICANCE_P).astype(float),
        levels=[0.5, 1.5],
        colors="none",
        hatches=["...."],
        transform=DATA_CRS,
    )
    ax.coastlines(linewidth=0.5)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title(title, fontsize=10)
    cbar = ax.figure.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
    cbar.set_label(units)
    return mesh


def plot_set(fit, set_def, run_label, out_path, predictand):
    """Render all predictor coefficient maps for one regression set to ``out_path``.

    One panel per predictor (the intercept is omitted). Stippling marks p > 0.05.
    ``predictand`` is a ``regression.PREDICTANDS`` entry (label + units), used for
    titles and to form coefficient units ([predictand units] / [predictor units]).
    """
    from regression import PREDICTORS  # local import to avoid a cycle at import time

    plabel, punits = predictand["label"], predictand["units"]
    cmap = predictand.get("cmap", "RdBu_r")
    predictors = set_def["predictors"]
    n = len(predictors)
    fig, axes = plt.subplots(
        n, 1, figsize=(9, 4.0 * n), squeeze=False,
        subplot_kw={"projection": PROJECTION},
    )
    for ax, name in zip(axes[:, 0], predictors):
        meta = PREDICTORS[name]
        plot_coefficient_map(
            fit["coef"].sel(param=name),
            fit["pvalue"].sel(param=name),
            title=f"∂{plabel}/∂{meta['label']}  ({meta['label']} coefficient)",
            units=f"({punits}) / {meta['units']}",
            ax=ax,
            cmap=cmap,
        )
    preds = ", ".join(PREDICTORS[p]["label"] for p in predictors)
    fig.suptitle(
        f"Set {set_def['number']}: {plabel} ~ {preds}  |  {run_label}\n"
        f"stippling: p > {SIGNIFICANCE_P} (nominal OLS; autocorrelation not corrected)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
