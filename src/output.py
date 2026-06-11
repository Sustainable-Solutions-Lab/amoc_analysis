"""Plot regression-coefficient maps with significance stippling.

Coefficient maps use a diverging colormap with symmetric bounds (white = 0, per
the project plotting conventions) and Cartopy coastlines. Cells where the
coefficient is not significant at p < 0.05 are stippled with hatching.
"""

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy import stats

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SIGNIFICANCE_P = 0.05
PROJECTION = ccrs.PlateCarree(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()

# Axis labels for the scalar predictors (used by the scatter plot).
SCALAR_AXIS_LABELS = {
    "tas_global_mean": "global-mean tas (K)",
    "amoc_strength": "AMOC strength (Sv)",
    "tas_interhemispheric_diff": "interhemispheric tas diff, NH−SH (K)",
}


def _symmetric_bound(values):
    """Robust symmetric color bound: the 99th percentile of |values| (NaN-safe)."""
    return float(np.nanpercentile(np.abs(values), 99))


def _save_figure(fig, out_path=None, pdf=None):
    """Write ``fig`` as a standalone PDF (``out_path``) or one page of ``pdf``.

    ``pdf`` is a ``matplotlib.backends.backend_pdf.PdfPages`` handle; when given,
    the figure is appended as a page (so many figures collect into one file).
    """
    if pdf is not None:
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
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
    lon, lat = coef["lon"], coef["lat"]
    b = bound if bound is not None else _symmetric_bound(coef.values)
    mesh = ax.pcolormesh(
        lon, lat, coef, cmap=cmap, vmin=-b, vmax=b,
        shading="auto", transform=DATA_CRS, rasterized=rasterized,
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


def plot_set(fit, set_def, run_label, out_path, predictand, centering=None):
    """Render all predictor coefficient maps for one regression set to ``out_path``.

    One panel per predictor (the intercept is omitted). Stippling marks p > 0.05.
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
    for ax, name in zip(flat, predictors):
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_set_monthly(fit, set_def, run_label, out_path, predictand, centering=None,
                     rasterized=True):
    """Render a monthly regression set to one PDF: a 3x4 twelve-month grid per predictor.

    ``fit`` is the stacked monthly result with ``coef``/``pvalue`` on dims
    ``(month, param, lat, lon)`` and a 1..12 ``month`` coordinate. One PDF page per
    predictor (intercept omitted); each page shows that predictor's coefficient map
    for all 12 calendar months on a shared symmetric color scale. ``centering`` is the
    optional ``tag -> (mean, units)`` dict (set-10 cross-product offsets, month-mean)
    annotated on the page. ``rasterized`` embeds the map fields as raster (small PDF).
    """
    from regression import PREDICTORS  # local import to avoid an import cycle

    plabel, punits = predictand["label"], predictand["units"]
    cmap = predictand.get("cmap", "RdBu_r")
    centering_line = ""
    if centering:
        parts = ", ".join(f"{tag} = {mean:.4g} {units}" for tag, (mean, units) in centering.items())
        centering_line = f"\ncentering means (month-mean; per-month values in the NetCDF): {parts}"

    with PdfPages(out_path) as pdf:
        for name in set_def["predictors"]:
            meta = PREDICTORS[name]
            coef_p = fit["coef"].sel(param=name)
            pval_p = fit["pvalue"].sel(param=name)
            bound = _symmetric_bound(coef_p.values)  # shared scale across all 12 months
            fig, axes = plt.subplots(
                3, 4, figsize=(22, 12), squeeze=False,
                subplot_kw={"projection": PROJECTION},
            )
            for ax, month in zip(axes.flat, fit["month"].values):
                plot_coefficient_map(
                    coef_p.sel(month=month), pval_p.sel(month=month),
                    title=MONTH_NAMES[int(month) - 1],
                    units=f"({punits}) / {meta['units']}",
                    ax=ax, cmap=cmap, bound=bound, rasterized=rasterized,
                )
            fig.suptitle(
                f"Set {set_def['number']}: ∂{plabel}/∂{meta['label']} by calendar month "
                f"|  {run_label}\n"
                f"stippling: p > {SIGNIFICANCE_P} (nominal OLS; autocorrelation not "
                f"corrected){centering_line}",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig, dpi=150, bbox_inches="tight")
            plt.close(fig)


def plot_eof_patterns(eof_ds, title, units, out_path, cmap="RdBu_r", max_patterns=9):
    """Map the leading EOF spatial patterns plus a scree panel of variance explained.

    ``eof_ds`` is the Dataset from ``eof.compute_eofs``. Patterns use a symmetric
    diverging colormap. Only the leading ``max_patterns`` modes are mapped (fields
    such as precipitation are not low-rank and can retain hundreds of modes at the
    95% threshold — mapping them all is unreadable and the regression uses every
    retained mode regardless). The final panel is a scree: a per-mode bar when the
    modes are few, otherwise a cumulative-variance curve marking the retained count.
    """
    eofs = eof_ds["eofs"]
    n = eofs.sizes["mode"]
    var = eof_ds["variance_fraction"].values
    n_plot = min(n, max_patterns)
    ncols = 2
    nrows = -(-(n_plot + 1) // ncols)  # +1 for the scree panel
    fig = plt.figure(figsize=(6.0 * ncols, 3.4 * nrows))
    for i in range(n_plot):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=PROJECTION)
        e = eofs.isel(mode=i)
        bound = _symmetric_bound(e.values)
        mesh = ax.pcolormesh(e["lon"], e["lat"], e, cmap=cmap, vmin=-bound,
                             vmax=bound, shading="auto", transform=DATA_CRS)
        ax.coastlines(linewidth=0.5)
        ax.set_global()
        ax.set_title(f"EOF {i + 1}  ({var[i] * 100:.1f}% var)", fontsize=10)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(units)
    ax = fig.add_subplot(nrows, ncols, n_plot + 1)
    if n <= 20:
        ax.bar(np.arange(1, n + 1), var * 100)
        ax.set_ylabel("variance explained (%)")
    else:
        ax.plot(np.arange(1, n + 1), np.cumsum(var) * 100, lw=1.2)
        ax.set_ylabel("cumulative variance (%)")
    ax.set_xlabel("EOF mode")
    ax.set_title(f"scree: {n} modes retained (Σ = {var.sum() * 100:.1f}%); "
                 f"mapped leading {n_plot}", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pc_timeseries(eof_ds, title, out_path, max_modes=4):
    """Time series of the EOF weightings (PCs) vs year, one panel per simulation.

    Each panel plots the leading ``max_modes`` principal components against that
    run's own years; line breaks are inserted across any year gaps (a
    discontinuity left by dropping NaN years) so segments aren't joined across them.
    """
    pcs = eof_ds["pcs"]
    years = eof_ds["sample"].values
    run_of = eof_ds["run"].values
    runs = list(dict.fromkeys(run_of))
    n_modes = min(pcs.sizes["mode"], max_modes)

    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        m = run_of == run
        order = np.argsort(years[m])
        yr = years[m][order].astype(float)
        # Break lines only across a genuine gap, scaled to the sampling interval
        # (annual: spacing 1, decadal blocks: ~10), so the historical-ssp585 gap
        # still breaks but regular decadal steps stay connected.
        d = np.diff(yr)
        thresh = 1.5 * np.median(d) if d.size else np.inf
        gaps = np.where(d > thresh)[0] + 1
        yr_b = np.insert(yr, gaps, np.nan)
        for k in range(n_modes):
            v = pcs.isel(mode=k).values[m][order]
            ax.plot(yr_b, np.insert(v, gaps, np.nan), lw=1.0, label=f"PC{k + 1}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(run, fontsize=10)
        ax.set_ylabel("PC amplitude")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, ncol=n_modes, loc="best")
    axes[-1, 0].set_xlabel("year")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False)
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


def plot_pc_prediction(eof_ds, pc_fit, predictors, title, out_path=None,
                       pdf=None, max_modes=3):
    """Overlay the fitted (X·β) PC against the actual PC over time, per simulation.

    A direct view of how well the scalar predictors reproduce each EOF weighting:
    the fitted PC is ``intercept + Σⱼ βⱼ xⱼ`` in raw PC units. The leading
    ``max_modes`` modes are drawn (solid = actual, dashed = fitted) on one panel
    per run; line breaks follow genuine year gaps exactly as in
    ``plot_pc_timeseries``. ``predictors`` must contain the regressed columns.
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
    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.6 * len(runs)), squeeze=False)
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
        ax.set_ylabel("PC amplitude")
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

    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.9 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs):
        ax2 = ax.twinx()
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
    """Four-panel scatter of the pooled predictors, points colored by simulation.

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
    runs = list(dict.fromkeys(predictors["run"].values))
    colors = plt.cm.tab10(np.arange(len(runs)))
    run_of = predictors["run"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (xv, yv) in zip(axes.flat, panels):
        x, y = predictors[xv].values, predictors[yv].values
        for color, run in zip(colors, runs):
            m = run_of == run
            ax.scatter(x[m], y[m], s=12, color=color, alpha=0.7,
                       edgecolors="none", label=run)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_xlabel(SCALAR_AXIS_LABELS[xv])
        ax.set_ylabel(SCALAR_AXIS_LABELS[yv])
        ax.set_title(f"pooled r = {r:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8, markerscale=1.6, title="simulation")
    fig.suptitle(
        "Pooled regression predictors (AMOC-complete sample, 4 simulations)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itcz_timeseries(annual, decadal, title, out_path):
    """Per-simulation time series of the ITCZ latitude (``precip_max_lat``).

    One panel per simulation; annual values as a thin line and decadal block means
    overlaid as a marked line. Lines break across genuine year gaps (a
    discontinuity left by dropping NaN years). ``annual``/``decadal`` are the pooled response
    DataArrays (on ``sample`` with a ``run`` coord) from
    ``regression.build_pooled_scalar`` (block=None and block=10).
    """
    runs = list(dict.fromkeys(annual["run"].values))
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs)), squeeze=False)
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itcz_predicted_vs_observed(observed, run_of, panels, title, out_path):
    """Predicted vs observed ITCZ latitude for one or more multi-predictor fits.

    One subplot per entry of ``panels`` (each a dict with ``label``, ``predicted``
    -- the fitted values on ``sample`` -- and ``r2``). Points are colored by
    simulation; the dashed 1:1 line and R² / n are drawn on shared, equal-aspect
    axes so departures from the 1:1 line read directly as prediction error.
    ``observed`` is the response array and ``run_of`` the per-sample run labels.
    """
    runs = list(dict.fromkeys(run_of))
    colors = plt.cm.tab10(np.arange(len(runs)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.8),
                             squeeze=False)
    for ax, panel in zip(axes[0], panels):
        pred = panel["predicted"]
        lo = float(min(observed.min(), pred.min()))
        hi = float(max(observed.max(), pred.max()))
        pad = 0.05 * (hi - lo)
        lim = (lo - pad, hi + pad)
        for color, run in zip(colors, runs):
            m = run_of == run
            ax.scatter(observed[m], pred[m], s=12, color=color, alpha=0.7,
                       edgecolors="none", label=run)
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itcz_coefficients(panels, title, out_path):
    """Partial-slope bar charts (coef ± SE) for one or more multi-predictor fits.

    One subplot per entry of ``panels`` (each a dict with ``label`` and equal-length
    ``names`` / ``coef`` / ``se`` / ``pvalue`` lists, intercept excluded). Bars are
    the partial slopes with ±SE error bars, blue for positive and red for negative,
    hatched where not significant at p < 0.05. Note the slopes carry mixed units
    (° lat per K, per Sv, or per K·Sv for the interaction), so compare sign and
    significance rather than bar heights across different predictors.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4),
                             squeeze=False)
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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_itcz_scatter(predictors, response, fits, single_vars, title, out_path):
    """ITCZ latitude vs each single predictor: scatter, OLS line, 95% CI band.

    One panel per variable in ``single_vars`` (the single-predictor sets: Tglob,
    ΔT_NS, AMOC). Points are colored by simulation. The pooled simple-OLS line is
    drawn with a shaded 95% mean-response confidence band, and the slope ± SE, R²
    and p-value (from the corresponding ``fits[var]`` Dataset, i.e.
    ``regression.fit_scalar_ols`` on that single predictor) are annotated.
    ``predictors`` and ``response`` are the pooled Datasets/DataArray on ``sample``.
    """
    runs = list(dict.fromkeys(predictors["run"].values))
    colors = plt.cm.tab10(np.arange(len(runs)))
    run_of = predictors["run"].values
    y = response.values

    fig, axes = plt.subplots(1, len(single_vars), figsize=(5.2 * len(single_vars), 4.6),
                             squeeze=False)
    for ax, var in zip(axes[0], single_vars):
        x = predictors[var].values
        for color, run in zip(colors, runs):
            m = run_of == run
            ax.scatter(x[m], y[m], s=12, color=color, alpha=0.7,
                       edgecolors="none", label=run)

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
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
