"""Plot regression-coefficient maps with significance stippling.

Coefficient maps use a diverging colormap with symmetric bounds (white = 0, per
the project plotting conventions) and simplified Natural Earth coastlines
(``draw_coastlines``). Cells where the coefficient is not significant at
p < 0.05 are stippled with hatching.
"""

import matplotlib

matplotlib.use("Agg")
import functools

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats

import data_loader as dl

SIGNIFICANCE_P = 0.05
PROJECTION = ccrs.EqualEarth()  # default for all maps (UN guidance)
DATA_CRS = ccrs.PlateCarree()

# Coarse coastline: Natural Earth 110m coastline lines, Douglas-Peucker simplified
# with a tolerance of COASTLINE_TOLERANCE degrees (lon/lat), dropping lines shorter
# than COASTLINE_MIN_LENGTH degrees (small islands). Still finer than the
# 1.9 x 2.5 deg model grid, with ~1/5 the vertices of the 110m set, so vector
# coastlines stay small in many-panel PDFs.
COASTLINE_TOLERANCE = 1.0
COASTLINE_MIN_LENGTH = 5.0

# CF-convention units string for a dimensionless quantity (e.g. planetary albedo).
# It stays "1" in the data attributes but is left off plot labels.
DIMENSIONLESS_UNITS = "1"


def label_with_units(label, units):
    """``"label (units)"``, or bare ``label`` for a dimensionless quantity."""
    return label if units == DIMENSIONLESS_UNITS else f"{label} ({units})"


def value_with_units(value_text, units):
    """``"value_text units"``, or bare ``value_text`` for a dimensionless quantity."""
    return value_text if units == DIMENSIONLESS_UNITS else f"{value_text} {units}"


@functools.cache
def coarse_coastline():
    """Simplified global coastline as an array of Shapely LineStrings (lon/lat)."""
    lines = shapely.get_parts(np.array(list(
        cfeature.NaturalEarthFeature("physical", "coastline", "110m").geometries())))
    return shapely.simplify(lines[shapely.length(lines) >= COASTLINE_MIN_LENGTH],
                            COASTLINE_TOLERANCE)


def draw_coastlines(ax):
    """Draw ``coarse_coastline`` on a Cartopy map ``ax``."""
    ax.add_geometries(coarse_coastline(), crs=DATA_CRS, facecolor="none",
                      edgecolor="black", linewidth=0.5)


def centered_lon(da):
    """``da`` with longitude relabeled to [-180, 180) and sorted, for mapping.

    The CAM grid runs 0..357.5 E, so drawn as is its wrap point lands on the
    central meridian of the Greenwich-centered map projection, leaving a thin
    unfilled stripe at 0 deg. Relabeled, the wrap point moves to +/-180 deg, the
    map's outer edge. ``draw_field`` applies it; contour and stippling calls on a
    map need it too.
    """
    return da.assign_coords(lon=(da["lon"] + 180) % 360 - 180).sortby("lon")


def draw_field(ax, da, **kwargs):
    """``pcolormesh`` of a ``(lat, lon)`` field on map ``ax``, returning the mesh.

    The cell corners are projected here, once, and drawn in the map's own
    coordinates. Cartopy's ``pcolormesh(transform=...)`` does the same job but
    spends ~1.3 s per call checking for cells that wrap around the map edge,
    which made a 9-panel page take ~12 s. So that no cell straddles the edge,
    longitudes are relabeled to [-180, 180), the -180 column is repeated at +180,
    and the outer cell edges are clipped to +/-180 deg lon and +/-90 deg lat.
    ``kwargs`` go to ``pcolormesh`` (``cmap``, ``vmin``, ``vmax``, ``rasterized``).
    """
    da = centered_lon(da)
    lon, lat = da["lon"].values, da["lat"].values
    values = np.concatenate([da.values, da.values[:, :1]], axis=1)
    lon_edges = np.concatenate([[-180.0], (lon[1:] + lon[:-1]) / 2,
                                [lon[-1] + (lon[-1] - lon[-2]) / 2, 180.0]])
    lat_edges = np.concatenate([[-90.0], (lat[1:] + lat[:-1]) / 2, [90.0]])
    corners = ax.projection.transform_points(DATA_CRS, *np.meshgrid(lon_edges, lat_edges))
    return ax.pcolormesh(corners[..., 0], corners[..., 1], values, shading="flat", **kwargs)

# Line-plot convention for cases: CO2 level sets the line style (1x solid,
# 2x dashed, 4x dotted) and hosing sets the color (-0.3 Sv red, 0 black,
# +0.3 Sv blue).
CO2_LINESTYLE = {1: "-", 2: "--", 4: ":"}
HOSING_COLOR = {-0.3: "red", 0.0: "black", 0.3: "blue"}


def case_line_style(case):
    """Matplotlib ``color``/``linestyle`` kwargs for a case's line."""
    spec = dl.EXPERIMENTS[case]
    return {"color": HOSING_COLOR[spec["hosing_sv"]],
            "linestyle": CO2_LINESTYLE[spec["co2_multiple"]]}


# Scatter-marker convention: CO2 level sets the marker shape -- 1x circle,
# 2x triangle, 4x square; hosing sets the color, as for lines. The triangle has the
# least ink at a given size, so the least-emphasized 2xCO2 level draws least attention.
# Markers are filled when each case contributes at most FILLED_MARKER_MAX_POINTS
# points (e.g. 10 decadal means) and open when it contributes more (e.g. ~100
# annual values), so dense clouds of overlapping points stay readable.
CO2_MARKER = {1: "o", 2: "^", 4: "s"}
FILLED_MARKER_MAX_POINTS = 10


def case_marker_style(case, points_per_case):
    """``ax.scatter`` kwargs for a case's ``points_per_case`` points: marker shape by
    CO2, color by hosing, filled or open per ``FILLED_MARKER_MAX_POINTS``."""
    spec = dl.EXPERIMENTS[case]
    color = HOSING_COLOR[spec["hosing_sv"]]
    filled = points_per_case <= FILLED_MARKER_MAX_POINTS
    return {"marker": CO2_MARKER[spec["co2_multiple"]],
            "facecolors": color if filled else "none",
            "edgecolors": "none" if filled else color,
            "linewidths": 0.7}

# Axis labels for the scalar predictors (used by the scatter plot).
SCALAR_AXIS_LABELS = {
    "tas_global_mean": "global-mean tas (K)",
    "amoc_strength": "AMOC strength (Sv)",
    "tas_interhemispheric_diff": "interhemispheric tas diff, NH−SH (K)",
}


def symmetric_bound(values):
    """Robust symmetric color bound: the 99th percentile of |values| (NaN-safe)."""
    return float(np.nanpercentile(np.abs(values), 99))


def _save_figure(fig, out_path=None, pdf=None, dpi=300):
    """Write ``fig`` as a standalone PDF (``out_path``) or one page of ``pdf``.

    ``pdf`` is a ``matplotlib.backends.backend_pdf.PdfPages`` handle; when given,
    the figure is appended as a page (so many figures collect into one file).
    """
    if pdf is not None:
        pdf.savefig(fig, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_coefficient_map(coef, pvalue, title, units, ax, cmap="RdBu_r", bound=None,
                         rasterized=False):
    """Draw one coefficient map on a Cartopy ``ax``: filled field + p>0.05 stippling.

    ``coef`` and ``pvalue`` are 2-D (lat, lon) DataArrays. ``cmap`` is a diverging
    colormap (white = 0); use ``RdBu_r`` for temperature (warm = red) and ``RdBu``
    for precipitation (wet = blue). The symmetric color scale is fixed to ±``bound``
    if given, else the 99th percentile of |coef| (values beyond saturate). With
    ``rasterized`` the filled field (the heavy artist) is embedded as raster while
    axes/text stay vector -- keeps many-panel PDFs small. Returns the mappable.
    """
    coef, pvalue = centered_lon(coef), centered_lon(pvalue)
    lon, lat = coef["lon"], coef["lat"]
    b = bound if bound is not None else symmetric_bound(coef.values)
    mesh = draw_field(ax, coef, cmap=cmap, vmin=-b, vmax=b, rasterized=rasterized)
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
    draw_coastlines(ax)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title(title, fontsize=10)
    cbar = ax.figure.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.03)
    cbar.set_label(units)
    return mesh


def plot_set(fit, set_def, run_label, pdf, predictand, centering=None):
    """Render all predictor coefficient maps for one regression set as one page of
    ``pdf`` (an open ``PdfPages`` book, one page per set).

    One panel per predictor (the intercept is omitted). Stippling marks p > 0.05.
    Panels whose coefficients carry the same units share one symmetric color scale.
    ``predictand`` is a ``regression.PREDICTANDS`` entry (label + units), used for
    titles and to form coefficient units ([predictand units] / [predictor units]).
    ``centering`` is the ``tag -> (mean, units)`` dict from
    ``regression.centering_means_for_set`` (empty/None for raw-predictor sets); when
    present its means are annotated, since they are the offsets a user must subtract
    before applying the centered cross-product terms.
    """
    from regression import PREDICTORS  # local import to avoid a cycle at import time

    plabel, punits = predictand["label"], predictand["units"]
    cmap = predictand.get("cmap", "RdBu_r")
    predictors = set_def["predictors"]
    n = len(predictors)
    # Stack few panels in a single column; lay many (e.g. the 9-term set) on a grid.
    ncols = 3 if n > 4 else 1
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9 if ncols == 1 else 6.0 * ncols, (4.0 if ncols == 1 else 3.4) * nrows),
        squeeze=False,
        subplot_kw={"projection": PROJECTION},
    )
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    units_of = {name: f"({punits}) / {PREDICTORS[name]['units']}" for name in predictors}
    bound_of = {units: symmetric_bound(np.concatenate(
                    [fit["coef"].sel(param=name).values.ravel()
                     for name in predictors if units_of[name] == units]))
                for units in set(units_of.values())}
    for ax, name in zip(flat, predictors):
        meta = PREDICTORS[name]
        plot_coefficient_map(
            fit["coef"].sel(param=name),
            fit["pvalue"].sel(param=name),
            title=f"∂{plabel}/∂{meta['label']}  ({meta['label']} coefficient)",
            units=units_of[name],
            ax=ax,
            cmap=cmap,
            bound=bound_of[units_of[name]],
        )
    preds = ", ".join(PREDICTORS[p]["label"] for p in predictors)
    centering_line = ""
    if centering:
        parts = ", ".join(f"{tag} = {mean:.4g} {units}" for tag, (mean, units) in centering.items())
        centering_line = f"\ncentering means (subtract before applying centered terms): {parts}"
    fig.suptitle(
        f"Set {set_def['number']}: {plabel} ~ {preds}  |  {run_label}\n"
        f"stippling: p > {SIGNIFICANCE_P} (nominal OLS; autocorrelation not corrected)"
        f"{centering_line}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, pdf=pdf)


def plot_eof_patterns(eof_ds, title, out_path=None, cmap="RdBu_r", max_patterns=9,
                      pdf=None):
    """Map the leading EOF spatial patterns plus a scree panel of variance explained.

    ``eof_ds`` is the Dataset from ``eof.compute_eofs``, whose EOFs are dimensionless
    patterns of area-weighted RMS 1 (the PCs carry the physical amplitude). All
    mapped patterns share one symmetric diverging color scale. Only the leading ``max_patterns`` modes are mapped (fields
    such as precipitation are not low-rank and can retain hundreds of modes at the
    95% threshold — mapping them all is unreadable and the regression uses every
    retained mode regardless). The final panel is a scree: a per-mode bar when the
    modes are few, otherwise a cumulative-variance curve marking the retained count.
    Written to ``out_path`` or appended as a page of ``pdf`` (see ``_save_figure``).
    """
    eofs = eof_ds["eofs"]
    n = eofs.sizes["mode"]
    var = eof_ds["variance_fraction"].values
    n_plot = min(n, max_patterns)
    ncols = 2
    nrows = -(-(n_plot + 1) // ncols)  # +1 for the scree panel
    fig = plt.figure(figsize=(6.0 * ncols, 3.4 * nrows))
    bound = symmetric_bound(eofs.isel(mode=slice(0, n_plot)).values)
    for i in range(n_plot):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=PROJECTION)
        mesh = draw_field(ax, eofs.isel(mode=i), cmap=cmap, vmin=-bound, vmax=bound)
        draw_coastlines(ax)
        ax.set_global()
        ax.set_title(f"EOF {i + 1}  ({var[i] * 100:.1f}% var)", fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("dimensionless (RMS = 1)")
    ax = fig.add_subplot(nrows, ncols, n_plot + 1)
    if n <= 20:
        ax.bar(np.arange(1, n + 1), var * 100)
        ax.set_ylabel("variance explained (%)")
    else:
        ax.plot(np.arange(1, n + 1), np.cumsum(var) * 100, lw=1.2)
        ax.set_ylabel("cumulative variance (%)")
    ax.set_xlabel("EOF mode")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"scree: {n} modes retained (Σ = {var.sum() * 100:.1f}%); "
                 f"mapped leading {n_plot}", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path=out_path, pdf=pdf)


def plot_pc_regression(pc_fit, predictors, pcs, title, out_path=None,
                       variance_fraction=None, pdf=None):
    """Per-mode bar charts of the PC-on-predictor regression (standardized).

    The EOF analog of the 2D coefficient maps: one panel per retained EOF mode,
    with one bar per predictor. Bar height is the **standardized** coefficient
    β·σ(xⱼ)/σ(PCₘ) (z-scoring predictors and the PC) so bars are comparable across
    modes — raw coefficients scale with each PC's amplitude. The whisker is the
    matching standardized SE. Bars significant at p < ``SIGNIFICANCE_P`` are drawn
    solid, non-significant ones faded; significance/p come straight from ``pc_fit``
    (scale-invariant). Panel titles report R² and (if given) the mode's variance
    fraction. ``predictors`` must contain the regressed columns; ``pcs`` is the
    (sample, mode) PC array used in the fit.
    """
    from regression import PREDICTORS  # local import to avoid an import cycle

    names = [p for p in pc_fit["param"].values if p != "intercept"]
    labels = [PREDICTORS[p]["label"] for p in names]
    sigma_x = np.array([predictors[p].values.std() for p in names])  # (k,)
    sigma_y = pcs.std("sample").values  # (mode,)
    modes = pc_fit["mode"].values
    n = modes.size

    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False,
                             sharey=True)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    x = np.arange(len(names))
    for i, mode in enumerate(modes):
        ax = flat[i]
        sel = pc_fit.sel(mode=mode)
        scale = sigma_x / sigma_y[i]  # standardize each predictor's coefficient
        beta = sel["coef"].sel(param=names).values * scale
        err = sel["se"].sel(param=names).values * scale
        sig = sel["pvalue"].sel(param=names).values < SIGNIFICANCE_P
        base = np.where(beta >= 0, "#c0392b", "#2c5fa8")  # warm +, cool -
        rgba = [mcolors.to_rgba(c, 1.0 if s else 0.35) for c, s in zip(base, sig)]
        ax.bar(x, beta, yerr=err, color=rgba, edgecolor="k", linewidth=0.5, capsize=3)
        ax.axhline(0, color="k", lw=0.6)
        vtxt = f", {variance_fraction[i] * 100:.0f}% var" if variance_fraction is not None else ""
        ax.set_title(f"EOF {int(mode)}  (R²={float(sel['r2'].values):.2f}{vtxt})", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("standardized coef (β·σx/σy)")
    fig.suptitle(
        f"{title}\nbars = standardized coefficients; faded = not significant "
        f"(p > {SIGNIFICANCE_P})", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, out_path, pdf)


def plot_pc_prediction(eof_ds, pc_fit, predictors, title, units, out_path=None,
                       pdf=None, max_modes=3):
    """Overlay the fitted (X·β) PC against the actual PC over time, per simulation.

    A direct view of how well the scalar predictors reproduce each EOF weighting:
    the fitted PC is ``intercept + Σⱼ βⱼ xⱼ`` in raw PC units. The leading
    ``max_modes`` modes are drawn (solid = actual, dashed = fitted) on one panel
    per run; lines break across genuine year gaps but stay connected across
    regular decadal steps. ``predictors`` must contain the regressed columns.
    ``units`` are the field's units, which the PCs carry (``eof.compute_eofs``).
    """
    pcs = eof_ds["pcs"]
    years = eof_ds["sample"].values
    run_of = eof_ds["run"].values
    runs = list(dict.fromkeys(run_of))
    params = list(pc_fit["param"].values)
    names = [p for p in params if p != "intercept"]

    X = np.column_stack([np.ones(pcs.sizes["sample"])] + [predictors[p].values for p in names])
    coef = pc_fit["coef"].sel(param=params).values  # (k, mode), intercept first
    fitted = X @ coef  # (sample, mode), aligned with pcs' sample axis

    n_modes = min(pcs.sizes["mode"], max_modes)
    colors = plt.cm.tab10(np.arange(n_modes))
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), squeeze=False,
                             sharex=True, sharey=True)
    for ax, run in zip(axes[:, 0], runs):
        m = run_of == run
        order = np.argsort(years[m])
        yr = years[m][order].astype(float)
        d = np.diff(yr)
        thresh = 1.5 * np.median(d) if d.size else np.inf
        gaps = np.where(d > thresh)[0] + 1
        yr_b = np.insert(yr, gaps, np.nan)
        for k in range(n_modes):
            act = pcs.isel(mode=k).values[m][order]
            fit_run = fitted[:, k][m][order]
            ax.plot(yr_b, np.insert(act, gaps, np.nan), lw=1.3, color=colors[k],
                    label=f"PC{k + 1} actual")
            ax.plot(yr_b, np.insert(fit_run, gaps, np.nan), lw=1.0, color=colors[k],
                    ls="--", label=f"PC{k + 1} fitted")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel(label_with_units("PC", units))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=n_modes, loc="best")
    axes[-1, 0].set_xlabel("year")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, pdf)


def plot_scalar_timeseries(annual, decadal, title, out_path):
    """Per-simulation time series of the scalar predictors, annual + decadal overlay.

    One panel per simulation. Tglob and ΔT_NS share the left axis as **anomalies
    from their pooled means** (they cannot share a raw axis — Tglob is ~287 K,
    ΔT_NS only a few K); AMOC strength is on a right axis in **absolute Sv** (its
    own axis, so real magnitudes and any collapse stay visible). Annual values are
    thin lines; decadal block means are overlaid as marked lines (centered on the
    same annual baseline, so they sit on the annual curves). Lines break across
    genuine year gaps (a discontinuity left by dropping NaN years). ``annual`` and
    ``decadal`` are pooled predictor Datasets (variables on ``sample`` with a
    ``run`` coord) from ``regression.build_pooled`` (block=None and block=10).
    """
    left_vars = [("tas_global_mean", "Tglob", "C0"),
                 ("tas_interhemispheric_diff", "ΔT_NS", "C1")]
    ref = {v: float(annual[v].values.mean()) for v, _, _ in left_vars}  # shared baseline
    runs = list(dict.fromkeys(annual["run"].values))

    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.9 * len(runs)), squeeze=False,
                             sharex=True, sharey=True)
    twins = [ax.twinx() for ax in axes[:, 0]]
    for twin in twins[1:]:  # one AMOC (Sv) range on every panel, like the K axes
        twin.sharey(twins[0])
    for ax, ax2, run in zip(axes[:, 0], twins, runs):
        for ds, style in [(annual, dict(lw=0.9, alpha=0.65)),
                          (decadal, dict(lw=1.8, marker="o", ms=3))]:
            m = ds["run"].values == run
            yr = ds["sample"].values[m].astype(float)
            order = np.argsort(yr)
            yr = yr[order]
            d = np.diff(yr)
            thresh = 1.5 * np.median(d) if d.size else np.inf  # break only real gaps
            gaps = np.where(d > thresh)[0] + 1
            yr_b = np.insert(yr, gaps, np.nan)
            for var, _lbl, c in left_vars:
                v = ds[var].values[m][order] - ref[var]
                ax.plot(yr_b, np.insert(v, gaps, np.nan), color=c, **style)
            a = ds["amoc_strength"].values[m][order]
            ax2.plot(yr_b, np.insert(a, gaps, np.nan), color="C3", **style)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel("Tglob, ΔT_NS anomaly (K)")
        ax2.set_ylabel("AMOC (Sv)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("year")

    handles = [
        Line2D([], [], color="C0", label="Tglob (left, K anomaly)"),
        Line2D([], [], color="C1", label="ΔT_NS (left, K anomaly)"),
        Line2D([], [], color="C3", label="AMOC (right, Sv)"),
        Line2D([], [], color="0.4", lw=0.9, label="annual"),
        Line2D([], [], color="0.4", lw=1.8, marker="o", ms=3, label="decadal mean"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=5, loc="best", framealpha=0.9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_scatter(predictors, out_path):
    """Four-panel scatter of the pooled predictors, points styled by case
    (``case_marker_style``).

    Top row uses global-mean tas on the x-axis (AMOC and ΔT_NS on y); bottom row
    uses AMOC strength on the x-axis (global-mean tas and ΔT_NS on y). The pooled
    Pearson r is annotated per panel. ``predictors`` is the pooled Dataset from
    ``regression.build_pooled`` (variables on ``sample``, with a ``run`` coord).
    """
    panels = [
        ("tas_global_mean", "amoc_strength"),
        ("tas_global_mean", "tas_interhemispheric_diff"),
        ("amoc_strength", "tas_global_mean"),
        ("amoc_strength", "tas_interhemispheric_diff"),
    ]
    runs = [case for case in dl.EXPERIMENTS if case in set(predictors["run"].values)]
    run_of = predictors["run"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (xv, yv) in zip(axes.flat, panels):
        x, y = predictors[xv].values, predictors[yv].values
        for run in runs:
            m = run_of == run
            ax.scatter(x[m], y[m], s=14, alpha=0.7, label=run,
                       **case_marker_style(run, m.sum()))
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel(SCALAR_AXIS_LABELS[xv])
        ax.set_ylabel(SCALAR_AXIS_LABELS[yv])
        ax.set_title(f"pooled r = {r:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(
        f"Pooled regression predictors (AMOC-complete sample, {len(runs)} simulations)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itcz_timeseries(annual, decadal, title, out_path=None, pdf=None):
    """Per-simulation time series of the ITCZ latitude (``precip_max_lat``).

    One panel per simulation; annual values as a thin line and decadal block means
    overlaid as a marked line. Lines break across genuine year gaps (a
    discontinuity left by dropping NaN years). ``annual``/``decadal`` are the pooled response
    DataArrays (on ``sample`` with a ``run`` coord) from
    ``regression.build_pooled_scalar`` (block=None and block=10).
    """
    runs = list(dict.fromkeys(annual["run"].values))
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs)), squeeze=False,
                             sharex=True, sharey=True)
    for ax, run in zip(axes[:, 0], runs):
        for da, style in [(annual, dict(lw=0.9, alpha=0.65)),
                          (decadal, dict(lw=1.8, marker="o", ms=3))]:
            m = da["run"].values == run
            yr = da["sample"].values[m].astype(float)
            order = np.argsort(yr)
            yr = yr[order]
            d = np.diff(yr)
            thresh = 1.5 * np.median(d) if d.size else np.inf  # break only real gaps
            gaps = np.where(d > thresh)[0] + 1
            yr_b = np.insert(yr, gaps, np.nan)
            v = da.values[m][order]
            ax.plot(yr_b, np.insert(v, gaps, np.nan), color="C2", **style)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel("ITCZ lat (°N)")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("year")
    handles = [
        Line2D([], [], color="C2", lw=0.9, label="annual"),
        Line2D([], [], color="C2", lw=1.8, marker="o", ms=3, label="decadal mean"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8, ncol=2, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_path, pdf)


def plot_itcz_predicted_vs_observed(observed, run_of, panels, title, out_path=None, pdf=None):
    """Predicted vs observed ITCZ latitude for one or more multi-predictor fits.

    One subplot per entry of ``panels`` (each a dict with ``label``, ``predicted``
    -- the fitted values on ``sample`` -- and ``r2``). Points are colored by
    simulation; the dashed 1:1 line and R² / n are drawn on shared, equal-aspect
    axes so departures from the 1:1 line read directly as prediction error.
    ``observed`` is the response array and ``run_of`` the per-sample run labels.
    """
    runs = [case for case in dl.EXPERIMENTS if case in set(run_of)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.8),
                             squeeze=False)
    # One range for both axes of every panel, spanning all observed and predicted values.
    values = np.concatenate([observed] + [np.asarray(p["predicted"]) for p in panels])
    pad = 0.05 * (values.max() - values.min())
    lim = (float(values.min() - pad), float(values.max() + pad))
    for ax, panel in zip(axes[0], panels):
        pred = panel["predicted"]
        for run in runs:
            m = run_of == run
            ax.scatter(observed[m], pred[m], s=14, alpha=0.7,
                       label=run, **case_marker_style(run, m.sum()))
        ax.plot(lim, lim, color="k", lw=1.0, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("observed ITCZ lat (°N)")
        ax.set_ylabel("predicted ITCZ lat (°N)")
        ax.set_title(f"{panel['label']}\nR² = {panel['r2']:.2f}, n = {observed.size}",
                     fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, out_path, pdf)


def plot_itcz_coefficients(panels, title, out_path=None, pdf=None):
    """Partial-slope bar charts (coef ± SE) for one or more multi-predictor fits.

    One subplot per entry of ``panels`` (each a dict with ``label`` and equal-length
    ``names`` / ``coef`` / ``se`` / ``pvalue`` lists, intercept excluded). Bars are
    the partial slopes with ±SE error bars, blue for positive and red for negative,
    hatched where not significant at p < 0.05. Note the slopes carry mixed units
    (° lat per K, per Sv, or per K·Sv for the interaction), so compare sign and
    significance rather than bar heights across different predictors.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4),
                             squeeze=False, sharey=True)
    for ax, panel in zip(axes[0], panels):
        x = np.arange(len(panel["names"]))
        coef = np.asarray(panel["coef"])
        se = np.asarray(panel["se"])
        sig = np.asarray(panel["pvalue"]) < SIGNIFICANCE_P
        ax.bar(x, coef, yerr=se, capsize=4,
               color=["C0" if c >= 0 else "C3" for c in coef],
               hatch=["" if s else "//" for s in sig],
               edgecolor="k", linewidth=0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(panel["names"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("partial slope (° lat / predictor unit)")
        ax.set_title(panel["label"], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    handles = [
        Line2D([], [], marker="s", ls="", color="C0", label="positive"),
        Line2D([], [], marker="s", ls="", color="C3", label="negative"),
        Line2D([], [], marker="s", ls="", color="0.8", mec="k", label="hatched: p≥0.05"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7, loc="best")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, out_path, pdf)


def plot_itcz_scatter(predictors, response, fits, single_vars, title, out_path=None, pdf=None):
    """ITCZ latitude vs each single predictor: scatter, OLS line, 95% CI band.

    One panel per variable in ``single_vars`` (the single-predictor sets: Tglob,
    ΔT_NS, AMOC). Points are colored by simulation. The pooled simple-OLS line is
    drawn with a shaded 95% mean-response confidence band, and the slope ± SE, R²
    and p-value (from the corresponding ``fits[var]`` Dataset, i.e.
    ``regression.fit_scalar_ols`` on that single predictor) are annotated.
    ``predictors`` and ``response`` are the pooled Datasets/DataArray on ``sample``.
    """
    runs = [case for case in dl.EXPERIMENTS if case in set(predictors["run"].values)]
    run_of = predictors["run"].values
    y = response.values

    fig, axes = plt.subplots(1, len(single_vars), figsize=(5.2 * len(single_vars), 4.6),
                             squeeze=False, sharey=True)
    for ax, var in zip(axes[0], single_vars):
        x = predictors[var].values
        for run in runs:
            m = run_of == run
            ax.scatter(x[m], y[m], s=14, alpha=0.7, label=run,
                       **case_marker_style(run, m.sum()))

        fit = fits[var]
        b0 = float(fit["coef"].sel(param="intercept"))
        b1 = float(fit["coef"].sel(param=var))
        slope_se = float(fit["se"].sel(param=var))
        pval = float(fit["pvalue"].sel(param=var))
        n, df = fit.attrs["nobs"], fit.attrs["df"]

        # 95% mean-response band: ŷ ± t* · s · sqrt(1/n + (x-x̄)²/Sxx).
        resid = y - (b0 + b1 * x)
        sigma2 = (resid**2).sum() / df
        xbar = x.mean()
        sxx = ((x - xbar) ** 2).sum()
        tcrit = stats.t.ppf(0.975, df)
        xx = np.linspace(x.min(), x.max(), 200)
        yhat = b0 + b1 * xx
        band = tcrit * np.sqrt(sigma2 * (1.0 / n + (xx - xbar) ** 2 / sxx))
        ax.plot(xx, yhat, color="k", lw=1.6)
        ax.fill_between(xx, yhat - band, yhat + band, color="k", alpha=0.15, lw=0)

        ax.set_xlabel(SCALAR_AXIS_LABELS[var])
        ax.set_ylabel("ITCZ lat (°N)")
        ax.set_title(
            f"slope = {b1:+.3g} ± {slope_se:.2g} °/[{SCALAR_AXIS_LABELS[var].split('(')[-1].rstrip(') ')}]\n"
            f"R² = {fit.attrs['r2']:.2f}, p = {pval:.1e}, n = {n}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, out_path, pdf)


def plot_tglob_vs_amoc(annual, decadal, out_path):
    """x-y line plot of AMOC strength against global-mean tas, one line per case.

    ``annual`` and ``decadal`` are pooled predictor Datasets from
    ``regression.build_pooled`` (block=None and block=10): each case is drawn as a
    thin line through its annual values and a bold line through its 10-year block
    means, both in time order, with a circle at the first decadal mean. Line style
    and color follow the case convention (``case_line_style``).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for case in dl.EXPERIMENTS:
        style = case_line_style(case)
        a = annual.isel(sample=annual["run"].values == case)
        d = decadal.isel(sample=decadal["run"].values == case)
        ax.plot(a["tas_global_mean"], a["amoc_strength"], lw=0.6, alpha=0.35,
                color=style["color"])
        ax.plot(d["tas_global_mean"], d["amoc_strength"], lw=2.0, label=case, **style)
        ax.plot(d["tas_global_mean"][0], d["amoc_strength"][0], "o", ms=5,
                color=style["color"])
    ax.set_xlabel("global-mean near-surface air temperature, tas (K)")
    ax.set_ylabel("AMOC strength at 26.5°N (Sv)")
    ax.set_title(
        "CESM1.2 AMOC vs global-mean temperature, 2051–2150\n"
        "bold: 10-year block means (○ = first decade); thin: annual means",
        fontsize=10,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, handlelength=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# The four pages drawn per variable by ``plot_case_grid_book``: (page title, map
# of the (co2, hosing) grid to the plotted field). Differences broadcast over the
# grid, so the reference panels of each difference page are identically zero.
CASE_GRID_PAGES = [
    ("raw", lambda grid: grid),
    ("minus piControl (1xCO2, 0 Sv)", lambda grid: grid - grid.sel(co2=1, hosing=0.0)),
    ("minus 1xCO2 at the same hosing (CO2 effect)", lambda grid: grid - grid.sel(co2=1)),
    ("minus no hosing at the same CO2 (hosing effect)", lambda grid: grid - grid.sel(hosing=0.0)),
]


# Resolution of the rasterized case-grid map fields (text and coastlines stay
# vector). The nine small panels need no more, and PdfPages holds every page's
# images until the book closes: at 300 dpi a 4-page book peaked at ~5 GB.
CASE_GRID_RASTER_DPI = 150


def plot_case_grid_page(field, title, cmap, vmin, vmax, pdf, rasterized):
    """One page of ``pdf``: a 3 x 3 map grid of ``field`` (``(co2, hosing, lat,
    lon)``, laid out like ``data_loader.CASE_GRID``) on a shared color scale.

    Each panel is titled with its case name and area-weighted global mean. With
    ``rasterized`` the filled fields are embedded as raster images (small PDF);
    otherwise everything is vector.
    """
    fig, axes = plt.subplots(
        len(dl.CO2_LEVELS), len(dl.HOSING_LEVELS), figsize=(15, 9.5),
        subplot_kw={"projection": PROJECTION}, layout="constrained",
    )
    units = field.attrs["units"]
    for i, co2 in enumerate(dl.CO2_LEVELS):
        for j, hosing in enumerate(dl.HOSING_LEVELS):
            ax, panel = axes[i, j], field.sel(co2=co2, hosing=hosing)
            mesh = draw_field(ax, panel, cmap=cmap, vmin=vmin, vmax=vmax,
                              rasterized=rasterized)
            draw_coastlines(ax)
            ax.set_global()
            ax.set_title(f"{dl.CASE_GRID[i][j]}   global mean = "
                         + value_with_units(f"{float(dl.global_mean(panel)):.4g}", units),
                         fontsize=10)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.5,
                        pad=0.02, aspect=40)
    cbar.set_label(label_with_units(field.name, units))
    fig.suptitle(title, fontsize=12)
    _save_figure(fig, pdf=pdf, dpi=CASE_GRID_RASTER_DPI)


def plot_map_grid(panels, shape, title, units, cmap, bound, out_path=None, pdf=None):
    """One page of maps on a ``shape`` (rows, cols) grid sharing one color scale,
    written to ``out_path`` or appended to ``pdf`` (see ``_save_figure``).

    ``panels`` maps ``(row, col)`` to ``(panel title, (lat, lon) DataArray)``;
    grid cells not in ``panels`` are left blank. The symmetric scale is
    ±``bound`` (white = 0). Each panel title ends with its area-weighted global
    mean. Map fields are rasterized at ``CASE_GRID_RASTER_DPI``; text and
    coastlines stay vector.
    """
    fig, axes = plt.subplots(*shape, figsize=(5.0 * shape[1], 3.2 * shape[0]),
                             subplot_kw={"projection": PROJECTION}, layout="constrained")
    for ax in axes.flat:
        ax.set_visible(False)
    for (i, j), (panel_title, field) in panels.items():
        ax = axes[i, j]
        ax.set_visible(True)
        mesh = draw_field(ax, field, cmap=cmap, vmin=-bound, vmax=bound, rasterized=True)
        draw_coastlines(ax)
        ax.set_global()
        ax.set_title(f"{panel_title}\nglobal mean = {float(dl.global_mean(field)):+.3f} {units}",
                     fontsize=9)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", shrink=0.5,
                        pad=0.02, aspect=40)
    cbar.set_label(units)
    fig.suptitle(title, fontsize=12)
    _save_figure(fig, out_path=out_path, pdf=pdf, dpi=CASE_GRID_RASTER_DPI)


def plot_case_grid_book(grid, pdf, rasterized):
    """Append the four ``CASE_GRID_PAGES`` for one variable to ``pdf``.

    ``grid`` is a ``data_loader.case_grid_time_mean`` result. The raw page uses
    ``viridis`` scaled to the 1st-99th percentile over all nine panels; the
    difference pages use a diverging map with symmetric bounds (±99th percentile
    of |difference| over the page; white = 0): ``RdBu`` (wet = blue) for water
    fluxes (``data_loader.WATER_FLUX_UNITS``), else ``RdBu_r``.
    """
    header = (f"{grid.name}: {label_with_units(grid.attrs['long_name'], grid.attrs['units'])}, "
              f"CESM1.2 mean {grid.attrs['time_mean']}")
    diverging = "RdBu" if grid.attrs["units"] == dl.WATER_FLUX_UNITS else "RdBu_r"
    (raw_label, _), *difference_pages = CASE_GRID_PAGES
    low, high = np.nanpercentile(grid.values, [1, 99])
    plot_case_grid_page(grid, f"{header}\n{raw_label}", "viridis", low, high,
                        pdf, rasterized)
    for label, transform in difference_pages:
        field = transform(grid).assign_attrs(grid.attrs).rename(grid.name)
        bound = symmetric_bound(field.values)
        plot_case_grid_page(field, f"{header}\n{label}", diverging, -bound, bound,
                            pdf, rasterized)
