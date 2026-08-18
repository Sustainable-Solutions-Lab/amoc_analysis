"""The 3x3 panel figure: rows are CO2 levels, columns are hosing levels.

This module knows nothing about which analysis produced a field. It takes a
dict of ``{(co2, hosing): DataArray}`` on the model grid and draws a page, so
the quasi-steady-state and time-dependent workflows share it unchanged.
"""

from __future__ import annotations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import ticker

from .config import CO2_LEVELS, HOSING_LEVELS, get_case
from .io import global_mean

# Hosing is an ordered variable centred on zero, so its three curves take a
# diverging assignment: cool for a strengthened AMOC, neutral for the control,
# warm for a weakened one. The neutral middle is deliberate here (a categorical
# palette check flags it as low-chroma, which is the wrong check for an ordered
# set), and the dash patterns carry the same information without color.
HOSING_COLOR = {-0.3: "#2a78d6", 0.0: "#52514e", 0.3: "#eb6834"}
HOSING_DASH = {-0.3: (0, (5, 1.5)), 0.0: "solid", 0.3: (0, (1.4, 1.2))}

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
COASTLINE = "#3a3a38"
MISSING_FACE = "#f2f1ee"
# Land, on pages whose field is ocean-only and so NaN over the continents.
LAND_FACE = "#e6e5e1"

MAP_PROJECTION = ccrs.Robinson(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()

# Height/width of a global Robinson map. Placeholder panels take the same box
# shape so that empty cells line up with the maps beside them.
ROBINSON_BOX_ASPECT = 0.5072


def tick_levels(vmin: float, vmax: float) -> np.ndarray:
    """The values the colorbar will label, chosen once and used for both.

    Contours and colorbar ticks are drawn from this same array, so a contour
    line always sits exactly on a labeled color-scale value rather than at some
    nearby number of matplotlib's choosing.
    """
    locator = ticker.MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10])
    ticks = locator.tick_values(vmin, vmax)
    return ticks[(ticks > vmin) & (ticks < vmax)]


def hosing_label(hosing: float) -> str:
    """'-0.3 Sv', '0 Sv', '+0.3 Sv' — no '+0.0'."""
    return "0 Sv" if hosing == 0.0 else f"{hosing:+.1f} Sv"


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Cell boundaries midway between centers, extrapolated at both ends.

    The CAM FV grid has half-width cells at the poles, and this reproduces them:
    the first and last centers sit on the boundary, so their cells extend only
    half a spacing inward.
    """
    mid = (centers[:-1] + centers[1:]) / 2
    return np.concatenate([
        [centers[0] - (mid[0] - centers[0])], mid,
        [centers[-1] + (centers[-1] - mid[-1])],
    ])


_MESH_CACHE: dict[bytes, tuple] = {}


def _projected_mesh(lon: np.ndarray, lat: np.ndarray) -> tuple:
    """Project the grid into map coordinates once and reuse it for every panel.

    Passing ``transform=`` to pcolormesh makes cartopy re-project the mesh on
    every call — 1.4 s a panel, which dominated the whole book. The grid is
    identical for every panel of every page, so it is projected once here and
    the panels draw in native coordinates. Verified pixel-identical to the
    transform path everywhere except antialiasing on the map outline.

    Longitudes are rotated to -180..180 so no cell straddles the projection
    boundary, and the dateline column is repeated at -180 so both trimmed
    half-cells there carry the data that genuinely spans the seam.

    Returns projected edge coordinates plus the column index to apply to data.
    """
    key = lon.tobytes() + lat.tobytes()
    if key not in _MESH_CACHE:
        shifted = np.where(lon > 180, lon - 360, lon)
        order = np.argsort(shifted)
        reindex = np.concatenate([[order[-1]], order])
        lon_centers = np.concatenate([[shifted[order][-1] - 360], shifted[order]])

        lon_e = np.clip(_cell_edges(lon_centers), -180, 180)
        lat_e = np.clip(_cell_edges(lat), -90, 90)
        xyz = MAP_PROJECTION.transform_points(DATA_CRS, *np.meshgrid(lon_e, lat_e))
        _MESH_CACHE[key] = (xyz[..., 0], xyz[..., 1], reindex, lon_centers)
    return _MESH_CACHE[key]


def _draw_map(
    ax,
    da: xr.DataArray,
    cmap: str,
    vmin: float,
    vmax: float,
    mask: xr.DataArray | None,
    significance_style: str,
    contour_levels: np.ndarray | None,
    mean_fn=global_mean,
    footnote: str | None = None,
    nan_face: str | None = None,
) -> object:
    x, y, reindex, _ = _projected_mesh(da["lon"].values, da["lat"].values)
    ax.set_global()
    if nan_face is not None:
        # An ocean field is NaN over land; the axes background shows through the
        # holes in the mesh, so land is drawn by not drawing it.
        ax.set_facecolor(nan_face)
    mesh = ax.pcolormesh(
        x, y, da.values[:, reindex],
        cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor=COASTLINE)

    if contour_levels is not None:
        _draw_contours(ax, da, mask, significance_style, contour_levels)

    ax.text(
        0.5, -0.09, f"mean {float(mean_fn(da)):.4g}",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7, color=TEXT_SECONDARY,
    )
    if footnote is not None:
        ax.text(
            0.5, -0.165, footnote,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6.5, color=TEXT_SECONDARY,
        )
    return mesh


def _draw_contours(ax, da, mask, style, contour_levels) -> None:
    """Black contours at the colorbar's own tick values.

    With no ``mask`` the lines are drawn across the whole panel — that is the
    absolute pages, where no statistical test is involved and a contour is just
    an isoline. With a mask, they are clipped to where the difference is
    significant, so a line means both "this value" and "trust it".

    ``style="outline"`` instead traces the boundary of the significant region,
    saying where the signal is trustworthy but nothing about its magnitude.

    All contours are solid, overriding matplotlib's dashed-for-negative default:
    the color already carries sign, and the dashes only added visual noise.
    Either way the color field covers the whole map — contours add information,
    they never mask, stipple, or hide data.
    """
    x, y, reindex, _ = _projected_mesh(da["lon"].values, da["lat"].values)
    xc, yc = _cell_centers(x), _cell_centers(y)

    if style == "outline" and mask is not None:
        ax.contour(
            xc, yc, mask.values[:, reindex].astype(float), levels=[0.5],
            colors="black", linewidths=0.5,
        )
        return

    values = da.where(mask) if mask is not None else da
    ax.contour(
        xc, yc, values.values[:, reindex], levels=contour_levels,
        colors="black", linewidths=0.4, linestyles="solid",
    )


def _cell_centers(edges: np.ndarray) -> np.ndarray:
    """Centers of the projected cells, for contouring on the same mesh."""
    return (edges[:-1, :-1] + edges[1:, 1:]) / 2


def _draw_missing(ax, note: str) -> None:
    ax.set_box_aspect(ROBINSON_BOX_ASPECT)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(MISSING_FACE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#c9c8c3")
    ax.add_patch(
        plt.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            facecolor="none", edgecolor="#dcdbd6", hatch="///", linewidth=0,
        )
    )
    ax.text(
        0.5, 0.5, note,
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8, color=TEXT_SECONDARY, linespacing=1.5,
    )


def _draw_zonal(ax, row_fields: dict[float, xr.DataArray], show_legend: bool) -> None:
    for hosing, da in sorted(row_fields.items()):
        ax.plot(
            da.mean("lon"), da["lat"],
            color=HOSING_COLOR[hosing], linestyle=HOSING_DASH[hosing],
            linewidth=1.4, label=hosing_label(hosing),
        )
    ax.set_ylim(-90, 90)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_yticklabels(["60S", "30S", "EQ", "30N", "60N"], fontsize=7)
    ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
    ax.grid(True, color="#e8e7e3", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    if show_legend:
        # "best" rather than a fixed corner: the curves' shape varies by field,
        # and a pinned legend lands on top of them for some (CLDTOT, PRECT).
        ax.legend(fontsize=6.5, frameon=False, loc="best", labelcolor=TEXT_SECONDARY,
                  handlelength=2.2, borderpad=0.2, labelspacing=0.25)


def grid_3x3(
    fields: dict[tuple[int, float], xr.DataArray],
    *,
    title: str,
    subtitle: str,
    units: str,
    cmap: str,
    vmin: float,
    vmax: float,
    significance: dict[tuple[int, float], xr.DataArray] | None = None,
    significance_style: str = "field",
    drop_zero_contour: bool = True,
    annotations: dict[tuple[int, float], str] | None = None,
    footnotes: dict[tuple[int, float], str] | None = None,
    missing_notes: dict[tuple[int, float], str] | None = None,
    mean_fn=global_mean,
    nan_face: str | None = None,
    extend: str = "neither",
    figsize: tuple[float, float] = (13.5, 7.6),
    bottom: float = 0.115,
    hspace: float = 0.16,
) -> plt.Figure:
    """Draw one page: 3x3 maps plus a zonal-mean profile beside each row.

    ``fields`` is keyed by ``(co2, hosing)``; absent keys render as "not yet
    run" placeholders so the grid keeps its shape as runs complete. Color limits
    are passed in rather than computed here, so a caller can hold them fixed
    across pages. ``annotations`` labels individual panels above the map, e.g.
    marking the reference case on an anomaly page; ``footnotes`` labels them
    below, under the panel mean, which is where a per-panel averaging window
    goes when the cases do not share one. ``mean_fn`` overrides the cos(lat)
    global mean in that footer — ocean fields want their own area weights.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3, 4, width_ratios=[1, 1, 1, 0.62],
        left=0.05, right=0.975, top=0.885, bottom=bottom,
        wspace=0.09, hspace=hspace,
    )

    ticks = tick_levels(vmin, vmax)
    # On a difference field the zero contour traces where the sign changes
    # rather than a magnitude, and would ring every panel wherever values merely
    # cross zero. On an absolute field zero is a genuine isoline, so it is kept.
    contour_levels = ticks[ticks != 0] if drop_zero_contour else ticks
    mesh = None
    map_axes: list = []
    zonal_axes: list = []

    for r, co2 in enumerate(CO2_LEVELS):
        row_fields: dict[float, xr.DataArray] = {}
        for c, hosing in enumerate(HOSING_LEVELS):
            key = (co2, hosing)
            if key in fields:
                ax = fig.add_subplot(gs[r, c], projection=MAP_PROJECTION)
                mask = significance.get(key) if significance is not None else None
                mesh = _draw_map(
                    ax, fields[key], cmap, vmin, vmax, mask,
                    significance_style, contour_levels,
                    mean_fn=mean_fn,
                    footnote=(footnotes or {}).get(key),
                    nan_face=nan_face,
                )
                row_fields[hosing] = fields[key]
                if annotations is not None and key in annotations:
                    ax.text(
                        0.5, 1.015, annotations[key], transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=7, style="italic",
                        color=TEXT_SECONDARY,
                    )
            else:
                ax = fig.add_subplot(gs[r, c])
                default_note = f"not yet run\n{get_case(co2, hosing).name}"
                note = (missing_notes or {}).get(key, default_note)
                _draw_missing(ax, note)

            map_axes.append(ax)
            if r == 0:
                ax.set_title(f"{hosing_label(hosing)} hosing", fontsize=10,
                             color=TEXT_PRIMARY, pad=8)
            if c == 0:
                ax.text(
                    -0.06, 0.5, f"{co2}$\\times$CO$_2$",
                    transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=10, color=TEXT_PRIMARY,
                )

        zonal_ax = fig.add_subplot(gs[r, 3])
        zonal_axes.append(zonal_ax)
        _draw_zonal(zonal_ax, row_fields, show_legend=(r == 0))
        if r == 0:
            zonal_ax.set_title("zonal mean", fontsize=8.5, color=TEXT_SECONDARY, pad=8)
        if r == 2:
            zonal_ax.set_xlabel(units, fontsize=7.5, color=TEXT_SECONDARY)

    _share_zonal_limits(zonal_axes)

    fig.canvas.draw()  # settle the aspect-adjusted map positions before placing the colorbar
    first, third = map_axes[0].get_position(), map_axes[2].get_position()
    cax = fig.add_axes([first.x0, 0.055, third.x1 - first.x0, 0.018])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend=extend, ticks=ticks)
    cbar.set_label(units, fontsize=8.5, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=7.5, colors=TEXT_SECONDARY, length=2)
    cbar.outline.set_edgecolor("#c9c8c3")

    fig.suptitle(title, fontsize=13, color=TEXT_PRIMARY, x=0.05, ha="left", y=0.965)
    fig.text(0.05, 0.925, subtitle, fontsize=9, color=TEXT_SECONDARY, ha="left")
    return fig


def _share_zonal_limits(panels: list) -> None:
    """Give every zonal panel the same x-range so rows compare directly.

    The zero line is drawn only when the data actually straddle zero — on an
    absolute field it would otherwise drag the axis down to zero and squash the
    curves into a corner.
    """
    lo = min(ax.get_xlim()[0] for ax in panels)
    hi = max(ax.get_xlim()[1] for ax in panels)
    for ax in panels:
        ax.set_xlim(lo, hi)
        if lo < 0 < hi:
            ax.axvline(0, color="#c9c8c3", linewidth=0.6, zorder=0)
