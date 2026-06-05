"""EOF / principal-component analysis of pooled annual fields, and regression of
the principal-component time series on the scalar predictors.

This is an additive alternative to the direct per-grid-point regression
(`regression.fit_grid_ols`): instead of regressing every grid cell, we decompose
the field into a few empirical orthogonal functions (EOFs), regress the leading
principal-component (PC) time series on the predictors, and map the coefficients
back to space as Σ_k β_k·EOF_k. With all modes retained this reproduces the
direct regression exactly; truncation (here, to ≥95% of variance) denoises it.

Conventions (see README / plan):
- Anomalies use the grand temporal mean over all pooled samples (keeps the
  between-run forced variability the predictors explain).
- Area-weighted covariance EOF: anomalies are multiplied by √(zonal-band area
  weight) before the SVD and patterns divided by it afterward (no per-cell std
  normalization).
"""

import numpy as np
import xarray as xr
from scipy import stats

import data_loader as dl


def _sqrt_area_weights(field):
    """√(area weight) per grid cell, broadcast to (lat, lon) and mean-normalized."""
    w = dl.latitude_band_weights(field["lat"]).values
    w = w / w.mean()
    wcell = np.repeat(w[:, None], field["lon"].size, axis=1)  # (lat, lon)
    return np.sqrt(wcell)


def compute_eofs(field, variance_threshold=0.95):
    """Area-weighted covariance EOFs of grand-mean anomalies of ``field``.

    ``field`` has dims (sample, lat, lon). Returns an xarray Dataset with
    ``eofs`` (mode, lat, lon; physical units, un-weighted), ``pcs`` (sample, mode;
    = U·S, physical amplitude), ``variance_fraction`` (mode), and ``mean_map``
    (lat, lon). Modes are truncated to the smallest count whose cumulative
    variance fraction reaches ``variance_threshold`` (use 1.0 to keep all).
    """
    nlat, nlon = field["lat"].size, field["lon"].size
    mean_map = field.mean("sample")
    anom = (field - mean_map).values.reshape(field.sizes["sample"], -1)  # (n, cells)

    sw = _sqrt_area_weights(field).reshape(-1)  # (cells,)
    A = anom * sw[None, :]

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    var_frac = S**2 / (S**2).sum()
    n_modes = int(np.searchsorted(np.cumsum(var_frac), variance_threshold) + 1)
    n_modes = min(n_modes, S.size)

    pcs = (U[:, :n_modes] * S[:n_modes])
    eofs = (Vt[:n_modes] / sw[None, :]).reshape(n_modes, nlat, nlon)

    modes = np.arange(1, n_modes + 1)
    ds = xr.Dataset(
        {
            "eofs": (("mode", "lat", "lon"), eofs),
            "pcs": (("sample", "mode"), pcs),
            "variance_fraction": (("mode",), var_frac[:n_modes]),
            "mean_map": (("lat", "lon"), mean_map.values),
        },
        coords={"mode": modes, "lat": field["lat"], "lon": field["lon"]},
        attrs={"n_modes": n_modes, "total_variance_fraction": float(var_frac[:n_modes].sum())},
    )
    # Preserve the year and source-simulation labels of each pooled sample so the
    # PCs can be plotted as time series per simulation.
    ds = ds.assign_coords(sample=("sample", field["sample"].values))
    if "run" in field.coords:
        ds = ds.assign_coords(run=("sample", field["run"].values))
    return ds


def fit_pcs(predictors, pcs):
    """Closed-form OLS of each PC (sample, mode) on ``predictors`` + intercept.

    Same normal-equations math as ``regression.fit_grid_ols``; returns a Dataset
    with ``coef``, ``se``, ``tstat``, ``pvalue`` indexed (param, mode), where
    ``param`` is ``["intercept", *predictor names]``.
    """
    names = list(predictors.data_vars)
    X = np.column_stack([np.ones(pcs.sizes["sample"])]
                        + [predictors[v].values for v in names])
    n, k = X.shape
    df = n - k
    Y = pcs.values  # (sample, mode)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ Y)  # (k, mode)
    resid = Y - X @ beta
    sigma2 = (resid**2).sum(axis=0) / df  # (mode,)
    se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))  # (k, mode)
    tstat = beta / se
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), df)

    param = ["intercept"] + names
    dims = ("param", "mode")
    return xr.Dataset(
        {
            "coef": (dims, beta),
            "se": (dims, se),
            "tstat": (dims, tstat),
            "pvalue": (dims, pvalue),
            # Retained for exact fingerprint-variance propagation (see reconstruct):
            "resid": (("sample", "mode"), resid),
            "xtx_inv": (("param", "param_b"), XtX_inv),
        },
        coords={"param": param, "param_b": param, "mode": pcs["mode"],
                "sample": pcs["sample"]},
        attrs={"nobs": n, "df": df},
    )


def reconstruct_fingerprint(eof_ds, pc_fit):
    """Map PC-regression coefficients back to physical space per predictor.

    fingerprint_j(x) = Σ_k β_{j,k}·EOF_k(x). Its variance is propagated exactly:
    the retained PCs are orthogonal but their regression *residuals* are not, so
    Var = (XᵀX)⁻¹_jj · e(x)ᵀ Σ_resid e(x), where Σ_resid is the cross-mode
    residual covariance and e(x) the EOF values at x. With all modes retained this
    reproduces the direct field regression's coef AND p-value exactly. Returns a
    Dataset with ``coef`` and ``pvalue`` (param, lat, lon) — the structure
    ``output.plot_set`` consumes (intercept dropped, as for the direct maps).
    """
    eofs = eof_ds["eofs"]  # (mode, lat, lon)
    nlat, nlon = eofs.sizes["lat"], eofs.sizes["lon"]
    df = pc_fit.attrs["df"]
    params = [p for p in pc_fit["param"].values if p != "intercept"]

    resid = pc_fit["resid"].values                      # (sample, mode)
    sigma = resid.T @ resid / df                         # (mode, mode)
    E = eofs.values.reshape(eofs.sizes["mode"], -1)      # (mode, ncell)
    quad = np.einsum("kx,kl,lx->x", E, sigma, E).reshape(nlat, nlon)  # e' Σ e

    coef_list, pval_list = [], []
    for p in params:
        cmap = (pc_fit["coef"].sel(param=p) * eofs).sum("mode")       # (lat, lon)
        var = float(pc_fit["xtx_inv"].sel(param=p, param_b=p)) * quad
        tstat = cmap.values / np.sqrt(var)
        pval = 2.0 * stats.t.sf(np.abs(tstat), df)
        coef_list.append(cmap)
        pval_list.append(xr.DataArray(pval, coords=cmap.coords, dims=cmap.dims))

    coef = xr.concat(coef_list, dim="param").assign_coords(param=params)
    pvalue = xr.concat(pval_list, dim="param").assign_coords(param=params)
    return xr.Dataset({"coef": coef, "pvalue": pvalue})
