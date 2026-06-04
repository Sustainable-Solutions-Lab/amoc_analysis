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

# Scalar predictors: variable name -> (short label, the predictor's own units).
# A regression coefficient then has units [predictand units] / [predictor units].
PREDICTORS = {
    "tas_global_mean": {"label": "Tglob", "tag": "Tglob", "units": "K"},
    "tas_interhemispheric_diff": {"label": "dT_NS", "tag": "dT_NS", "units": "K"},
    "amoc_strength": {"label": "AMOC", "tag": "AMOC", "units": "Sv"},
    # Gram-Schmidt residual columns (added by add_orthogonalized_columns); a
    # residual keeps its parent index's units. Used by the orthogonalized sets.
    "dT_NS|Tglob": {"label": "dT_NS⊥Tglob", "tag": "dT_NSperpTglob", "units": "K"},
    "AMOC|Tglob,dT_NS": {
        "label": "AMOC⊥(Tglob,dT_NS)", "tag": "AMOCperpTglob.dT_NS", "units": "Sv",
    },
    "AMOC|Tglob": {"label": "AMOC⊥Tglob", "tag": "AMOCperpTglob", "units": "Sv"},
    "dT_NS|Tglob,AMOC": {
        "label": "dT_NS⊥(Tglob,AMOC)", "tag": "dT_NSperpTglob.AMOC", "units": "K",
    },
    # Quadratic response-surface terms (added by add_quadratic_columns); base
    # predictors are centered on their pooled means first, so a term keeps the
    # units of its centered factors (e.g. Tglob^2 -> K^2, Tglob*AMOC -> K Sv).
    "q_Tglob": {"label": "Tglob", "tag": "Tglob", "units": "K"},
    "q_Tglob2": {"label": "Tglob²", "tag": "Tglob2", "units": "K^2"},
    "q_AMOC": {"label": "AMOC", "tag": "AMOC", "units": "Sv"},
    "q_AMOC2": {"label": "AMOC²", "tag": "AMOC2", "units": "Sv^2"},
    "q_dT_NS": {"label": "dT_NS", "tag": "dT_NS", "units": "K"},
    "q_dT_NS2": {"label": "dT_NS²", "tag": "dT_NS2", "units": "K^2"},
    "q_Tglob.AMOC": {"label": "Tglob·AMOC", "tag": "TglobxAMOC", "units": "K Sv"},
    "q_Tglob.dT_NS": {"label": "Tglob·dT_NS", "tag": "TglobxdT_NS", "units": "K^2"},
    "q_AMOC.dT_NS": {"label": "AMOC·dT_NS", "tag": "AMOCxdT_NS", "units": "K Sv"},
}

# Gram-Schmidt residual columns: key -> (target index, [indices to regress out]).
ORTHOGONAL_COLUMNS = {
    "dT_NS|Tglob": ("tas_interhemispheric_diff", ["tas_global_mean"]),
    "AMOC|Tglob,dT_NS": (
        "amoc_strength",
        ["tas_global_mean", "tas_interhemispheric_diff"],
    ),
    "AMOC|Tglob": ("amoc_strength", ["tas_global_mean"]),
    "dT_NS|Tglob,AMOC": (
        "tas_interhemispheric_diff",
        ["tas_global_mean", "amoc_strength"],
    ),
}

# Gridded predictands (response fields). Each maps every run to the processed
# file and variable that supplies the field for that run. Precipitation mixes
# total `pr` (historical-ssp585, abrupt-4xCO2) with convective `prc` (piControl,
# u03-hos) — physically different quantities, pooled here only to prototype.
PREDICTANDS = {
    "tas": {
        "label": "tas",
        "units": "K",
        "cmap": "RdBu_r",  # warm (positive) = red
        "by_run": {r: {"file": f"tas_annual_CESM2_{r}.nc", "var": "tas"} for r in RUNS},
    },
    "precip": {
        "label": "precip",
        "units": "kg m-2 s-1",
        "cmap": "RdBu",  # wetter (positive) = blue, drier = red (precip convention)
        "by_run": {
            "historical-ssp585": {"file": "pr_annual_CESM2_historical-ssp585.nc", "var": "pr"},
            "abrupt-4xCO2": {"file": "pr_annual_CESM2_abrupt-4xCO2.nc", "var": "pr"},
            "piControl": {"file": "prc_annual_CESM2_piControl.nc", "var": "prc"},
            "u03-hos": {"file": "prc_annual_CESM2_u03-hos.nc", "var": "prc"},
        },
    },
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
    # Orthogonalized (Gram-Schmidt) sets: mutually orthogonal columns (VIF=1).
    {
        "number": 7,  # order tas -> NS -> AMOC
        "predictors": ["tas_global_mean", "dT_NS|Tglob", "AMOC|Tglob,dT_NS"],
    },
    {
        "number": 8,  # order tas -> AMOC -> NS
        "predictors": ["tas_global_mean", "AMOC|Tglob", "dT_NS|Tglob,AMOC"],
    },
    # Full quadratic response surface (centered predictors): 9 terms + intercept.
    {
        "number": 9,
        "predictors": [
            "q_Tglob", "q_Tglob2", "q_AMOC", "q_AMOC2", "q_dT_NS", "q_dT_NS2",
            "q_Tglob.AMOC", "q_Tglob.dT_NS", "q_AMOC.dT_NS",
        ],
    },
]

# Union of all predictors used in any set; defines the common sample of years.
PREDICTOR_UNION = ["tas_global_mean", "tas_interhemispheric_diff", "amoc_strength"]


def build_pooled(runs=RUNS, predictor_union=PREDICTOR_UNION, predictand=None):
    """Pool a gridded predictand and scalar predictors across runs onto one axis.

    For each run, keep only the years for which every predictor in
    ``predictor_union`` is present (intersection of non-NaN), then concatenate
    runs along a new ``sample`` dimension. ``predictand`` is an entry of
    ``PREDICTANDS`` (defaults to ``tas``) giving the per-run response file/var.
    Returns ``(predictors, response)`` where ``predictors`` is a Dataset of the
    scalar variables on ``sample`` and ``response`` has dims ``(sample, lat,
    lon)``. A ``run`` coordinate labels each sample's source simulation.
    """
    if predictand is None:
        predictand = PREDICTANDS["tas"]

    pred_parts, resp_parts = [], []
    for run in runs:
        scal = xr.open_dataset(
            os.path.join(dl.PROCESSED_DIR, f"scalars_annual_CESM2_{run}.nc")
        )[predictor_union]
        spec = predictand["by_run"][run]
        resp = xr.open_dataset(os.path.join(dl.PROCESSED_DIR, spec["file"]))[
            spec["var"]
        ].rename(predictand["label"])

        valid = scal.to_dataframe().dropna().index  # years with all predictors
        scal = scal.sel(year=valid)
        resp = resp.sel(year=valid)

        run_label = xr.DataArray(np.full(valid.size, run), dims="year")
        pred_parts.append(scal.assign_coords(run=("year", run_label.data)))
        resp_parts.append(resp.assign_coords(run=("year", run_label.data)))

    predictors = xr.concat(pred_parts, dim="year").rename(year="sample")
    response = xr.concat(resp_parts, dim="year").rename(year="sample")
    return predictors, response


def _residual(y, given):
    """Residual of 1-D ``y`` after OLS on ``given`` (list of 1-D arrays) + intercept."""
    X = np.column_stack([np.ones(y.size)] + list(given))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def add_orthogonalized_columns(predictors):
    """Augment a pooled predictor Dataset with Gram-Schmidt residual columns.

    Each entry of ``ORTHOGONAL_COLUMNS`` is added as a new variable: the target
    index with the listed indices regressed out (on the pooled ``sample`` axis).
    The result is orthogonal to those indices, so an orthogonalized set has
    mutually uncorrelated columns (VIF = 1).
    """
    out = predictors.copy()
    for key, (target, given) in ORTHOGONAL_COLUMNS.items():
        resid = _residual(
            predictors[target].values, [predictors[g].values for g in given]
        )
        out[key] = ("sample", resid)
    return out


def add_quadratic_columns(predictors):
    """Augment with the centered quadratic response-surface terms (for set 9).

    Each base predictor is centered on its pooled mean, then squares and pairwise
    products are formed from the centered values. Centering is essential: with raw
    values (Tglob ~ 287 K) the linear and squared terms are ~collinear and the
    design is singular; centering drops cond(XᵀX) from ~1e20 to ~1e5.
    """
    c = {
        "Tglob": predictors["tas_global_mean"].values,
        "AMOC": predictors["amoc_strength"].values,
        "dT_NS": predictors["tas_interhemispheric_diff"].values,
    }
    c = {k: v - v.mean() for k, v in c.items()}

    out = predictors.copy()
    out["q_Tglob"] = ("sample", c["Tglob"])
    out["q_AMOC"] = ("sample", c["AMOC"])
    out["q_dT_NS"] = ("sample", c["dT_NS"])
    out["q_Tglob2"] = ("sample", c["Tglob"] ** 2)
    out["q_AMOC2"] = ("sample", c["AMOC"] ** 2)
    out["q_dT_NS2"] = ("sample", c["dT_NS"] ** 2)
    out["q_Tglob.AMOC"] = ("sample", c["Tglob"] * c["AMOC"])
    out["q_Tglob.dT_NS"] = ("sample", c["Tglob"] * c["dT_NS"])
    out["q_AMOC.dT_NS"] = ("sample", c["AMOC"] * c["dT_NS"])
    return out


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
