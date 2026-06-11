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
# file and variable that supplies the field for that run. Precipitation is
# convective `prc` uniformly for all runs. For historical-ssp585 the r1->r4
# member splice (historical r1i1p1f1, ssp585 r4i1p1f1) is identical for `prc`,
# `tas`, and AMOC, so predictand and predictors stay mutually consistent.
PREDICTANDS = {
    "tas": {
        "label": "tas",
        "units": "K",
        "cmap": "RdBu_r",  # warm (positive) = red
        "by_run": {r: {"file": f"tas_annual_CESM2_{r}.nc", "var": "tas"} for r in RUNS},
    },
    "prc": {
        "label": "prc",
        "units": "kg m-2 s-1",
        "cmap": "RdBu",  # wetter (positive) = blue, drier = red (precip convention)
        "by_run": {
            r: {"file": f"prc_annual_CESM2_{r}.nc", "var": "prc"} for r in RUNS
        },
    },
    # Total precipitation, available for only two runs (no piControl/u03-hos). A
    # separate 2-run analysis; build_pooled derives its run set from these by_run keys.
    "pr": {
        "label": "pr",
        "units": "kg m-2 s-1",
        "cmap": "RdBu",
        "by_run": {
            "historical-ssp585": {"file": "pr_annual_CESM2_historical-ssp585.nc", "var": "pr"},
            "abrupt-4xCO2": {"file": "pr_annual_CESM2_abrupt-4xCO2.nc", "var": "pr"},
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
    # Global-temperature x AMOC interaction model (centered main effects so the
    # interaction coefficient is well conditioned and each main effect is read at
    # the other's mean): Tglob, AMOC, Tglob*AMOC.
    {
        "number": 10,
        "predictors": ["q_Tglob", "q_AMOC", "q_Tglob.AMOC"],
    },
]

# Sets run by default (set 5 = Tglob + AMOC; set 10 = Tglob*AMOC interaction); the
# rest are produced only with the scripts' ``--all-sets`` flag.
DEFAULT_SET_NUMBERS = [5, 10]


def select_predictor_sets(all_sets=False):
    """Predictor sets to run: all ten when ``all_sets`` else just the
    ``DEFAULT_SET_NUMBERS`` subset, preserving ``PREDICTOR_SETS`` order."""
    if all_sets:
        return PREDICTOR_SETS
    return [s for s in PREDICTOR_SETS if s["number"] in DEFAULT_SET_NUMBERS]


# Smoothing variants the orchestration scripts can run: 'annual' (interannual) and
# 'decadal10' (non-overlapping 10-year block means, written to a decadal10/ subdir).
# decadal10 is the default; the annual variant is opt-in via each script's
# ``--do-annuals`` flag.
SMOOTHINGS = [
    {"tag": "annual", "block": None, "subdir": ""},
    {"tag": "decadal10", "block": 10, "subdir": "decadal10"},
]


def select_smoothings(do_annuals=False):
    """Smoothing variants to run: decadal10 only by default; both (annual first when
    requested) when ``do_annuals``. The decadal10 subdir layout is unchanged either way."""
    if do_annuals:
        return SMOOTHINGS
    return [s for s in SMOOTHINGS if s["tag"] == "decadal10"]


# Union of all predictors used in any set; defines the common sample of years.
PREDICTOR_UNION = ["tas_global_mean", "tas_interhemispheric_diff", "amoc_strength"]

# Base predictors that the centered (``q_``) columns of sets 9-10 are demeaned by,
# mapping the short tag used in column names to (raw predictor, units). The pooled
# mean of each is the offset a downstream user must subtract before applying a
# centered term -- it matters for the cross-product (interaction) coefficient,
# whose marginal effect is read relative to the other variable's centering mean.
CENTERED_BASES = {
    "Tglob": ("tas_global_mean", "K"),
    "AMOC": ("amoc_strength", "Sv"),
    "dT_NS": ("tas_interhemispheric_diff", "K"),
}


def build_pooled(runs=None, predictor_union=PREDICTOR_UNION, predictand=None, block=None):
    """Pool a gridded predictand and scalar predictors across runs onto one axis.

    For each run, keep only the years for which every predictor in
    ``predictor_union`` is present (intersection of non-NaN), then concatenate
    runs along a new ``sample`` dimension. ``predictand`` is an entry of
    ``PREDICTANDS`` (defaults to ``tas``) giving the per-run response file/var.
    ``runs`` defaults to the predictand's own ``by_run`` keys, so a predictand
    defined on a subset of runs (e.g. total ``pr``, only 2 runs) pools just those.
    Returns ``(predictors, response)`` where ``predictors`` is a Dataset of the
    scalar variables on ``sample`` and ``response`` has dims ``(sample, lat,
    lon)``. A ``run`` coordinate labels each sample's source simulation.

    If ``block`` is given (e.g. 10), each run's predictors and response are
    low-pass filtered to slower timescales by ``block``-year non-overlapping
    block means (``data_loader.block_average_on_years``) before pooling -- the
    same operator applied to both, per run/segment, so their regression
    relationship is evaluated on the smoothed (decimated) series. ``block=None``
    keeps the full annual (interannual) sample.
    """
    if predictand is None:
        predictand = PREDICTANDS["tas"]
    if runs is None:
        runs = list(predictand["by_run"])

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

        if block is not None:
            scal = dl.block_average_on_years(scal, block)
            resp = dl.block_average_on_years(resp, block)

        n_run = scal.sizes["year"]
        run_label = xr.DataArray(np.full(n_run, run), dims="year")
        pred_parts.append(scal.assign_coords(run=("year", run_label.data)))
        resp_parts.append(resp.assign_coords(run=("year", run_label.data)))

    predictors = xr.concat(pred_parts, dim="year").rename(year="sample")
    response = xr.concat(resp_parts, dim="year").rename(year="sample")
    return predictors, response


def build_pooled_monthly(
    predictand, month, runs=None, predictor_union=PREDICTOR_UNION, block=None
):
    """Pool one calendar month's gridded predictand and the annual scalar predictors.

    The monthly analog of :func:`build_pooled`: for each run the response is the
    ``month``-slice of ``{var}_monthly_CESM2_{run}.nc`` (dims ``year, lat, lon``),
    regressed against the same *annual* scalar predictors (Tglob, AMOC, dT_NS) from
    ``scalars_annual_CESM2_{run}.nc`` -- so calendar month ``month`` is related to the
    annual indices of the same year. Years with any predictor missing are dropped;
    with ``block`` set, each run's predictors and that month's response are
    block-averaged over ``block`` years before pooling (= the 10-year average of that
    calendar month). ``runs`` defaults to the predictand's ``by_run`` keys.
    """
    if runs is None:
        runs = list(predictand["by_run"])

    pred_parts, resp_parts = [], []
    for run in runs:
        scal = xr.open_dataset(
            os.path.join(dl.PROCESSED_DIR, f"scalars_annual_CESM2_{run}.nc")
        )[predictor_union]
        var = predictand["by_run"][run]["var"]
        resp = (
            xr.open_dataset(
                os.path.join(dl.PROCESSED_DIR, f"{var}_monthly_CESM2_{run}.nc")
            )[var]
            .sel(month=month)
            .rename(predictand["label"])
        )

        valid = scal.to_dataframe().dropna().index  # years with all predictors
        scal = scal.sel(year=valid)
        resp = resp.sel(year=valid)

        if block is not None:
            scal = dl.block_average_on_years(scal, block)
            resp = dl.block_average_on_years(resp, block)

        n_run = scal.sizes["year"]
        run_label = xr.DataArray(np.full(n_run, run), dims="year")
        pred_parts.append(scal.assign_coords(run=("year", run_label.data)))
        resp_parts.append(resp.assign_coords(run=("year", run_label.data)))

    predictors = xr.concat(pred_parts, dim="year").rename(year="sample")
    response = xr.concat(resp_parts, dim="year").rename(year="sample")
    return predictors, response


def build_pooled_scalar(
    response_var, runs=RUNS, predictor_union=PREDICTOR_UNION, block=None
):
    """Pool a scalar response and the scalar predictors across runs onto one axis.

    Like :func:`build_pooled`, but the response is a scalar series read from the
    same ``scalars_annual_CESM2_{run}.nc`` file as the predictors (e.g.
    ``precip_max_lat``, the ITCZ proxy). For each run, only years where every
    predictor in ``predictor_union`` **and** ``response_var`` are present are kept
    (complete-case deletion); the kept years are then concatenated across runs.
    With ``block`` set, each run's predictors and response are block-averaged
    before pooling (same operator on both). Returns ``(predictors, response)``
    where both are on a ``sample`` dimension with a ``run`` coordinate.
    """
    pred_parts, resp_parts = [], []
    for run in runs:
        ds = xr.open_dataset(
            os.path.join(dl.PROCESSED_DIR, f"scalars_annual_CESM2_{run}.nc")
        )
        scal = ds[predictor_union]
        resp = ds[response_var]

        valid = ds[predictor_union + [response_var]].to_dataframe().dropna().index
        scal = scal.sel(year=valid)
        resp = resp.sel(year=valid)

        if block is not None:
            scal = dl.block_average_on_years(scal, block)
            resp = dl.block_average_on_years(resp, block)

        n_run = scal.sizes["year"]
        run_label = xr.DataArray(np.full(n_run, run), dims="year")
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
    raw = {
        "Tglob": predictors["tas_global_mean"].values,
        "AMOC": predictors["amoc_strength"].values,
        "dT_NS": predictors["tas_interhemispheric_diff"].values,
    }
    means = {k: float(v.mean()) for k, v in raw.items()}
    c = {k: raw[k] - means[k] for k in raw}

    out = predictors.copy()
    # Record the centering means so the netcdf and figures can report the offset a
    # downstream user must subtract before applying the centered (cross-product) terms.
    for tag, m in means.items():
        out.attrs[f"centering_mean_{tag}"] = m
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


def centering_means_for_set(predictors, names):
    """Centering means referenced by a set's centered (``q_``) predictors.

    ``predictors`` is the augmented Dataset from :func:`add_quadratic_columns`
    (which stored ``centering_mean_<tag>`` attrs); ``names`` is a set's predictor
    list. Returns an ordered dict ``tag -> (mean, units)`` for each base predictor
    whose tag appears in any centered column of the set, and an empty dict for sets
    built from raw predictors only.
    """
    return {
        tag: (predictors.attrs[f"centering_mean_{tag}"], units)
        for tag, (_, units) in CENTERED_BASES.items()
        if any(n.startswith("q_") and tag in n for n in names)
    }


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


def fit_scalar_ols(predictors, y, alpha=0.05):
    """Closed-form OLS of a scalar response ``y`` on ``predictors`` + intercept.

    The scalar analog of :func:`fit_grid_ols` (same normal-equations math, one
    response series instead of one per grid cell), used for the ITCZ-latitude
    regressions. Returns an xarray Dataset with per-parameter ``coef``, ``se``,
    ``tstat``, ``pvalue`` (dims ``param``) and the ``(1-alpha)`` ``conf_int``
    (dims ``param, bound``), plus scalar ``r2``, ``nobs`` and ``df`` in attrs.
    ``param`` is ``["intercept", *predictor names]``. Confidence intervals are
    included here (unlike the gridded fit) for the scatter-with-fit plots.
    """
    names = list(predictors.data_vars)
    columns = np.column_stack([predictors[v].values for v in names])
    X = np.column_stack([np.ones(columns.shape[0]), columns])  # (n, k)
    z = np.asarray(y.values, dtype=float)
    n, k = X.shape
    df = n - k

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ z)
    resid = z - X @ beta
    sigma2 = (resid**2).sum() / df
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    tstat = beta / se
    pvalue = 2.0 * stats.t.sf(np.abs(tstat), df)
    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
    conf_int = np.column_stack([beta - tcrit * se, beta + tcrit * se])

    r2 = 1.0 - (resid**2).sum() / ((z - z.mean()) ** 2).sum()

    param = ["intercept"] + names
    return xr.Dataset(
        {
            "coef": ("param", beta),
            "se": ("param", se),
            "tstat": ("param", tstat),
            "pvalue": ("param", pvalue),
            "conf_int": (("param", "bound"), conf_int),
        },
        coords={"param": param, "bound": ["lo", "hi"]},
        attrs={"nobs": n, "df": df, "r2": float(r2)},
    )


def predict_scalar_ols(fit, predictors):
    """Fitted values ``X @ beta`` for a :func:`fit_scalar_ols` result.

    ``fit`` carries ``coef`` indexed by ``param = ["intercept", *names]``;
    ``predictors`` must contain those named columns on the ``sample`` axis. Returns
    a 1-D numpy array of predicted responses (one per sample).
    """
    names = [p for p in fit["param"].values if p != "intercept"]
    pred = np.full(predictors.sizes["sample"], float(fit["coef"].sel(param="intercept")))
    for nm in names:
        pred = pred + float(fit["coef"].sel(param=nm)) * predictors[nm].values
    return pred


def variance_inflation_factors(predictors):
    """VIF for each predictor in a pooled-sample Dataset (diagnostic of collinearity)."""
    X = np.column_stack([predictors[v].values for v in predictors.data_vars])
    X = (X - X.mean(0)) / X.std(0)
    return dict(zip(predictors.data_vars, np.diag(np.linalg.inv(np.corrcoef(X.T)))))
