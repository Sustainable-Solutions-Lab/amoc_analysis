"""Pooled per-grid-point OLS of gridded annual-mean ``tas`` on scalar indices.

The response is gridded annual-mean ``tas`` (one time series per grid cell); the
predictors are the scalar annual indices ``tas_global_mean`` (Tglob),
``tas_interhemispheric_diff`` (dT_NS), and ``amoc_strength`` (AMOC). Each grid
cell gets a single regression, but the years from all simulations are **pooled**
into one fit (with a single common intercept and no per-run fixed effects), so
that between-run differences in how the indices co-vary break the strong
within-run collinearity.

All six predictor sets share one common sample: the years for which every
predictor in the union is present (intersection of non-NaN), pooled across runs.
The fit is closed-form OLS with a design matrix shared across cells, validated
against ``statsmodels`` in the analysis script.
"""

import os

import numpy as np
import xarray as xr
from scipy import stats

import data_loader as dl

RUNS = ["historical-ssp585", "abrupt-4xCO2", "piControl", "u03-hos"]

# Scalar predictors: variable name -> (short label, coefficient units for tas response).
PREDICTORS = {
    "tas_global_mean": {"label": "Tglob", "coef_units": "K/K"},
    "tas_interhemispheric_diff": {"label": "dT_NS", "coef_units": "K/K"},
    "amoc_strength": {"label": "AMOC", "coef_units": "K/Sv"},
}

# Predictor sets requested for analysis (column subsets of the predictor union).
PREDICTOR_SETS = [
    {"number": 1, "predictors": ["tas_global_mean"]},
    {"number": 2, "predictors": ["tas_interhemispheric_diff"]},
    {"number": 3, "predictors": ["amoc_strength"]},
    {"number": 4, "predictors": ["tas_global_mean", "tas_interhemispheric_diff"]},
    {"number": 5, "predictors": ["tas_global_mean", "amoc_strength"]},
    {
        "number": 6,
        "predictors": [
            "tas_global_mean",
            "tas_interhemispheric_diff",
            "amoc_strength",
        ],
    },
]

# Union of all predictors used in any set; defines the common sample of years.
PREDICTOR_UNION = ["tas_global_mean", "tas_interhemispheric_diff", "amoc_strength"]


def build_pooled(runs=RUNS, predictor_union=PREDICTOR_UNION):
    """Pool gridded ``tas`` and scalar predictors across runs onto one sample axis.

    For each run, keep only the years for which every predictor in
    ``predictor_union`` is present (intersection of non-NaN), then concatenate
    runs along a new ``sample`` dimension. Returns ``(predictors, tas)`` where
    ``predictors`` is an xarray Dataset of the scalar variables on ``sample`` and
    ``tas`` is the response with dims ``(sample, lat, lon)``. A ``run`` coordinate
    labels each sample's source simulation.
    """
    pred_parts, tas_parts = [], []
    for run in runs:
        scal = xr.open_dataset(
            os.path.join(dl.PROCESSED_DIR, f"scalars_annual_CESM2_{run}.nc")
        )[predictor_union]
        tas = xr.open_dataset(
            os.path.join(dl.PROCESSED_DIR, f"tas_annual_CESM2_{run}.nc")
        )["tas"]

        valid = scal.to_dataframe().dropna().index  # years with all predictors
        scal = scal.sel(year=valid)
        tas = tas.sel(year=valid)

        run_label = xr.DataArray(np.full(valid.size, run), dims="year")
        pred_parts.append(scal.assign_coords(run=("year", run_label.data)))
        tas_parts.append(tas.assign_coords(run=("year", run_label.data)))

    predictors = xr.concat(pred_parts, dim="year").rename(year="sample")
    tas = xr.concat(tas_parts, dim="year").rename(year="sample")
    return predictors, tas


def fit_grid_ols(predictors, tas):
    """Closed-form OLS of ``tas`` (sample, lat, lon) on ``predictors`` + intercept.

    The design matrix (intercept followed by the predictor columns) is shared by
    every grid cell, so all cells are solved at once via the normal equations.
    Returns an xarray Dataset with per-parameter ``coef``, ``se``, ``tstat`` and
    ``pvalue`` (dims ``param, lat, lon``) plus ``r2`` (lat, lon) and scalar
    ``nobs``. ``param`` is ``["intercept", *predictor names]``.
    """
    names = list(predictors.data_vars)
    columns = np.column_stack([predictors[v].values for v in names])
    X = np.column_stack([np.ones(columns.shape[0]), columns])  # (n, k)
    n, k = X.shape
    df = n - k

    lat, lon = tas["lat"], tas["lon"]
    Z = tas.values.reshape(n, -1)  # (n, ncells)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ Z)  # (k, ncells)
    resid = Z - X @ beta
    sigma2 = (resid**2).sum(axis=0) / df  # (ncells,)
    se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))  # (k, ncells)
    tstat = beta / se
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), df)

    ss_tot = ((Z - Z.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (resid**2).sum(axis=0) / ss_tot

    param = ["intercept"] + names
    shape = (k, lat.size, lon.size)
    dims = ("param", "lat", "lon")
    coords = {"param": param, "lat": lat, "lon": lon}
    return xr.Dataset(
        {
            "coef": (dims, beta.reshape(shape)),
            "se": (dims, se.reshape(shape)),
            "tstat": (dims, tstat.reshape(shape)),
            "pvalue": (dims, pvalue.reshape(shape)),
            "r2": (("lat", "lon"), r2.reshape(lat.size, lon.size)),
        },
        coords=coords,
        attrs={"nobs": n, "df": df},
    )


def variance_inflation_factors(predictors):
    """VIF for each predictor in a pooled-sample Dataset (diagnostic of collinearity)."""
    X = np.column_stack([predictors[v].values for v in predictors.data_vars])
    X = (X - X.mean(0)) / X.std(0)
    return dict(zip(predictors.data_vars, np.diag(np.linalg.inv(np.corrcoef(X.T)))))
