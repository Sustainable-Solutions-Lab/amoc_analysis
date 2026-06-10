"""Load CESM2 monthly NetCDF output and produce normalized annual means.

The precipitation analysis uses **convective precipitation (``prc``) uniformly for
every run** (total ``pr`` was never available for all runs). Two file conventions
coexist in ``data/input/``:

- **CMORized** files (``historical``, ``ssp585``, ``abrupt-4xCO2``) store
  temperature as ``tas`` (K); convective precipitation is the CMIP ``prc`` variable
  (kg m-2 s-1). The ssp585 future is split across two files. Its ensemble member is
  ``r4i1p1f1`` while the historical segment is ``r1i1p1f1`` -- but this r1->r4 splice
  is identical for ``prc``, ``tas``, and AMOC (the ``tas`` ssp585 file is labeled
  r1 but its ``variant_label`` is r4), so the predictand and the predictors stay
  mutually consistent within each period.
- **Raw CAM-history** files (``piControl``, ``u03-hos``) store temperature as
  ``TREFHT`` (K) and *convective* precipitation (liq + ice) under the variable name
  ``pr``, alongside a full set of CAM metadata variables. That precipitation is
  labeled ``units = "m/s"``, but its values are actually a water mass flux in
  kg m-2 s-1 (they match the CMIP precip magnitude; true m s-1 precipitation would
  be ~1000x smaller), so the label is treated as a mislabel and the data is used
  as-is without scaling.

This module reconciles both into a common convention following CMIP variable
names: ``tas`` (K) and ``prc`` (convective precipitation, kg m-2 s-1). It computes
month-length-weighted annual means (correct for the CESM2 ``noleap`` calendar)
and attaches provenance attributes documenting the original source and units.
"""

import datetime
import os

import numpy as np
import xarray as xr

# Per-variable output metadata. CMIP convention: ``prc`` is convective
# precipitation, a water mass flux (kg m-2 s-1). Only ``tas`` and ``prc`` are
# produced (total ``pr`` was never available for all runs).
VAR_METADATA = {
    "tas": {"units": "K", "long_name": "Near-Surface Air Temperature"},
    "prc": {
        "units": "kg m-2 s-1",
        "long_name": "Convective Precipitation",
        "precip_kind": "convective",
    },
}

# Directory layout relative to the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(_REPO_ROOT, "data", "input")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

ANNUAL_MEAN_METHOD = "month-length weighted (noleap)"

# Source-of-truth manifest: maps each input file to its target variable and the
# handling needed to normalize it. ``source_variable`` is the variable to read
# (renamed to the target ``var`` on output). ``segments`` (combined experiments)
# lists the per-file pieces concatenated along ``year``.
#
# Each entry produces exactly one output file: {var}_annual_CESM2_{experiment}.nc
INPUT_MANIFEST = [
    {
        "var": "tas",
        "experiment": "historical-ssp585",
        "segments": [
            {
                "file": "tas_Amon_CESM2_historical_r1i1p1f1_gn_18500115-20141215.nc",
                "source_variable": "tas",
            },
            {
                "file": "tas_Amon_CESM2_ssp585_r1i1p1f1_gn_20150115-21001215.nc",
                "source_variable": "tas",
            },
        ],
    },
    {
        "var": "prc",
        "experiment": "historical-ssp585",
        # Convective precip: ssp585 is split into two files, and its realization
        # (r4i1p1f1) differs from the historical/tas realization (r1i1p1f1) -- the
        # only prc ssp585 available; the member mismatch is documented as a caveat.
        "segments": [
            {
                "file": "prc_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc",
                "source_variable": "prc",
            },
            {
                "file": "prc_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-206412.nc",
                "source_variable": "prc",
            },
            {
                "file": "prc_Amon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc",
                "source_variable": "prc",
            },
        ],
    },
    {
        "var": "tas",
        "experiment": "abrupt-4xCO2",
        "segments": [
            {
                "file": "tas_Amon_CESM2_abrupt-4xCO2-002.nc",
                "source_variable": "tas",
            }
        ],
    },
    {
        "var": "prc",
        "experiment": "abrupt-4xCO2",
        "segments": [
            {
                "file": "prc_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101--099912.nc",
                "source_variable": "prc",
            }
        ],
    },
    {
        "var": "tas",
        "experiment": "piControl",
        "segments": [
            {
                "file": "tas_Amon_CESM2_piControl_070001-079912.nc",
                "source_variable": "TREFHT",
            }
        ],
    },
    {
        "var": "prc",
        "experiment": "piControl",
        "segments": [
            {
                "file": "pr_Amon_CESM2_piControl_070001-079912.nc",
                "source_variable": "pr",
            }
        ],
    },
    {
        "var": "tas",
        "experiment": "u03-hos",
        "segments": [
            {
                "file": "tas_Amon_CESM2_u03-hos_1850001-202112.nc",
                "source_variable": "TREFHT",
            }
        ],
    },
    {
        "var": "prc",
        "experiment": "u03-hos",
        "segments": [
            {
                "file": "pr_Amon_CESM2_u03-hos_1850001-202112.nc",
                "source_variable": "pr",
            }
        ],
    },
]


def _center_monthly_time(ds):
    """Center each monthly mean's ``time`` stamp within the month it represents.

    CMORized files stamp monthly means mid-month (day ~15), so the stamp already
    sits inside the correct month. The raw CAM-history files stamp each mean at
    the *end* of its averaging interval — the first of the following month (e.g.
    the January mean is labeled Feb 1, 00Z) — which would misassign December into
    the next year and miscount ``days_in_month``. Shifting those back 15 days
    lands every stamp inside the month it represents (``time_bnds`` is not used
    because it is absent from some files). Detected via the first stamp's
    day-of-month so the shift is applied only to end-of-interval files.
    """
    if int(ds["time"].dt.day.values[0]) == 1:
        ds = ds.assign_coords(time=ds["time"] - datetime.timedelta(days=15))
    return ds


def annual_mean(da):
    """Month-length-weighted annual mean of a monthly DataArray.

    Weights each month by its number of days (``time.dt.days_in_month``), which
    on the ``noleap`` calendar gives the exact time-mean over each year. Returns
    a DataArray with the monthly ``time`` dimension replaced by integer ``year``.
    """
    weights = da["time"].dt.days_in_month
    numerator = (da * weights).groupby("time.year").sum("time")
    denominator = weights.groupby("time.year").sum("time")
    return numerator / denominator


def block_average_on_years(obj, block):
    """Non-overlapping ``block``-year means over the ``year`` dimension.

    Works on a Dataset or DataArray carrying a ``year`` dimension. Years are first
    split into contiguous segments (a segment break is any year-to-year jump > 1,
    e.g. a discontinuity left by dropping NaN years), then within each segment
    grouped into consecutive blocks of ``block`` years and averaged. A block never
    spans a segment break, and trailing partial blocks (fewer than ``block`` years) are
    dropped. The returned object's ``year`` coordinate is each block's mean year
    (the block midpoint).

    This is the slow-timescale low-pass: applied identically to predictors and
    predictand, it decimates the annual series to ~independent decadal samples, so
    the regression characterizes slow variability with honest degrees of freedom.
    """
    years = np.asarray(obj["year"].values)
    segment = np.cumsum(np.r_[0, np.diff(years) > 1])        # segment id per year
    seg_start = np.r_[0, np.nonzero(np.diff(segment))[0] + 1]  # first index of each segment
    within = np.arange(years.size) - seg_start[segment]       # position within segment
    block_in_seg = within // block
    label = segment * (block_in_seg.max() + 1) + block_in_seg  # unique (segment, block) key

    grouper = xr.DataArray(label, dims="year", coords={"year": obj["year"]}, name="block")
    ones = xr.DataArray(np.ones(years.size), dims="year", coords={"year": obj["year"]})
    counts = ones.groupby(grouper).sum().values
    midyear = (
        xr.DataArray(years.astype(float), dims="year", coords={"year": obj["year"]})
        .groupby(grouper)
        .mean()
        .values
    )
    averaged = obj.groupby(grouper).mean("year")

    full = counts == block  # drop any trailing partial block
    return (
        averaged.isel(block=full)
        .rename(block="year")
        .assign_coords(year=midyear[full])
    )


def _load_segment(segment, target_var):
    """Open one input file, select/rename the source variable, convert units,
    and return its month-length-weighted annual mean as a DataArray named
    ``target_var`` with provenance attributes attached."""
    path = os.path.join(INPUT_DIR, segment["file"])
    # chunks={} streams via dask using the file's native on-disk chunking,
    # keeping peak memory low for the ~2.5 GB monthly files.
    ds = xr.open_dataset(path, chunks={})
    ds = _center_monthly_time(ds)
    da = ds[segment["source_variable"]]
    original_units = da.attrs.get("units", "")

    meta = VAR_METADATA[target_var]
    annual = annual_mean(da).rename(target_var)
    annual.attrs = {
        **meta,
        "source_file": segment["file"],
        "source_variable": segment["source_variable"],
        "original_units": original_units,
        "annual_mean_method": ANNUAL_MEAN_METHOD,
    }
    # The raw CAM-history precip is labeled "m/s" but its values are a water mass
    # flux in kg m-2 s-1 (matching the CMIP precip magnitude); record the mislabel
    # and leave the data unscaled rather than applying a bogus density factor.
    if target_var == "prc" and original_units == "m/s":
        annual.attrs["units_note"] = (
            "source units attribute was 'm/s' but values are kg m-2 s-1 "
            "(water mass flux); used as-is without conversion"
        )
    return annual


def load_and_normalize(entry):
    """Build the annual-mean DataArray for one manifest entry.

    Concatenates multi-segment experiments (e.g. historical+ssp585) along
    ``year``. Segments are assumed contiguous and non-overlapping in time.
    """
    pieces = [_load_segment(seg, entry["var"]) for seg in entry["segments"]]
    if len(pieces) == 1:
        return pieces[0]
    combined = xr.concat(pieces, dim="year")
    # Preserve attributes from the first segment; note all contributing sources.
    combined.attrs = dict(pieces[0].attrs)
    combined.attrs["source_file"] = "; ".join(
        seg["file"] for seg in entry["segments"]
    )
    return combined


# --- Scalar (one-value-per-year) diagnostics -------------------------------

AMOC_FILE = "CESM2_AMOC_experiments.nc"


def latitude_band_weights(lat):
    """Exact zonal-band area weights for a regular (ascending) latitude grid.

    Each cell's weight is proportional to its meridional band area,
    ``sin(edge_north) - sin(edge_south)``, with cell edges taken as the
    midpoints between adjacent centers and the outermost edges clamped to ±90°.
    This is exact for a regular lon×lat grid and correctly accounts for the
    CESM2 FV grid's half-width polar cells (unlike ``cos(lat)``, which zeros the
    ±90° cells). Longitude spacing is uniform and cancels in any mean.
    """
    lat_rad = np.deg2rad(np.asarray(lat))
    edges = np.empty(lat_rad.size + 1)
    edges[1:-1] = 0.5 * (lat_rad[:-1] + lat_rad[1:])
    edges[0], edges[-1] = -np.pi / 2, np.pi / 2
    weights = np.sin(edges[1:]) - np.sin(edges[:-1])
    return xr.DataArray(weights, coords={"lat": lat}, dims="lat")


def global_mean(da):
    """Area-weighted global mean over (``lat``, ``lon``)."""
    return da.weighted(latitude_band_weights(da["lat"])).mean(("lat", "lon"))


def interhemispheric_difference(da):
    """Area-weighted Northern- minus Southern-Hemisphere mean (NH lat>0, SH lat<0)."""
    weights = latitude_band_weights(da["lat"])
    nh = (da["lat"] > 0).values
    sh = (da["lat"] < 0).values
    nh_mean = da.isel(lat=nh).weighted(weights.isel(lat=nh)).mean(("lat", "lon"))
    sh_mean = da.isel(lat=sh).weighted(weights.isel(lat=sh)).mean(("lat", "lon"))
    return nh_mean - sh_mean


def tropical_precip_centroid_lat(da, band):
    """Precipitation-mass centroid latitude (deg N) -- an ITCZ-position index.

    The area- and precipitation-weighted mean latitude of the zonal-mean
    precipitation within ``|lat| <= band``::

        phi = sum_i lat_i * P_i * a_i / sum_i P_i * a_i,

    with ``P_i`` the zonal mean (mean over ``lon``) and ``a_i`` the exact band area
    weight (``latitude_band_weights``, so each latitude contributes in proportion to
    its band area). Unlike the bare argmax of ``P``, this integrates over *both*
    branches of a double ITCZ, so the index moves continuously as the branches'
    relative strength shifts rather than jumping when the taller branch flips
    hemispheres. ``da`` has dims ``(year, lat, lon)``; returns ``DataArray(year)``
    named ``precip_centroid_lat`` (degrees_north).
    """
    zm = da.mean("lon")
    w = latitude_band_weights(da["lat"])
    weighted = (zm * w).where(np.abs(da["lat"]) <= band)  # mass per latitude in band
    centroid = (weighted * da["lat"]).sum("lat") / weighted.sum("lat")
    return centroid.rename("precip_centroid_lat")


def amoc_strength_on_years(segments, years):
    """Place AMOC strength (Sv) onto ``years``.

    ``segments`` is a list of ``{"variable", "year_start"[, "file"]}``: each source
    variable's values are written starting at ``year_start`` on the target axis,
    read from ``file`` (default ``CESM2_AMOC_experiments.nc``, whose series are the
    first N years of their run). The historical-ssp585 run instead draws a single
    gap-free 1850-2100 series from ``AMOC_4models_hist_ssp585.nc``. Years not
    covered by any segment are left missing (NaN).
    """
    years = np.asarray(years)
    values = np.full(years.size, np.nan)
    for seg in segments:
        ds = xr.open_dataset(os.path.join(INPUT_DIR, seg.get("file", AMOC_FILE)))
        v = ds[seg["variable"]].values
        start = int(np.nonzero(years == seg["year_start"])[0][0])
        values[start : start + v.size] = v
    return xr.DataArray(
        values, coords={"year": years}, dims="year", name="amoc_strength"
    )


# One scalar file per simulation, sharing each run's gridded year axis. AMOC
# values 1..N map to the run's first N years (NaN-padded), except historical-ssp585,
# which carries a gap-free 1850-2100 AMOC series from AMOC_4models_hist_ssp585.nc.
# The greenland-hosing run has no gridded tas, so it carries AMOC only on a bare
# 1..100 index. ``precip_file``/``precip_var`` give the gridded annual precipitation
# source for the ITCZ centroid diagnostic (``precip_centroid_lat_*``) -- convective
# ``prc`` for every run (None where no gridded precip exists).
SCALAR_SIMULATIONS = [
    {
        "experiment": "historical-ssp585",
        "tas_file": "tas_annual_CESM2_historical-ssp585.nc",
        "precip_file": "prc_annual_CESM2_historical-ssp585.nc",
        "precip_var": "prc",
        "amoc_segments": [
            {
                "file": "AMOC_4models_hist_ssp585.nc",
                "variable": "CESM2",
                "year_start": 1850,
            },
        ],
        "amoc_note": (
            "1850-2100 continuous from AMOC_4models_hist_ssp585.nc (CESM2); "
            "the former 1950-2000 gap is filled."
        ),
    },
    {
        "experiment": "abrupt-4xCO2",
        "tas_file": "tas_annual_CESM2_abrupt-4xCO2.nc",
        "precip_file": "prc_annual_CESM2_abrupt-4xCO2.nc",
        "precip_var": "prc",
        "amoc_segments": [{"variable": "abrupt_4xCO2", "year_start": 1}],
        "amoc_note": "AMOC first 100 years mapped to run years 1-100; NaN after.",
    },
    {
        "experiment": "piControl",
        "tas_file": "tas_annual_CESM2_piControl.nc",
        "precip_file": "prc_annual_CESM2_piControl.nc",
        "precip_var": "prc",
        "amoc_segments": [{"variable": "piControl", "year_start": 700}],
        "amoc_note": "AMOC first 100 years mapped to run years 700-799.",
    },
    {
        "experiment": "u03-hos",
        "tas_file": "tas_annual_CESM2_u03-hos.nc",
        "precip_file": "prc_annual_CESM2_u03-hos.nc",
        "precip_var": "prc",
        "amoc_segments": [{"variable": "hosing_0.3Sv_uniform", "year_start": 1850}],
        "amoc_note": "AMOC first 100 years mapped to run years 1850-1949; NaN after.",
    },
    {
        "experiment": "hosing-0.1Sv-greenland",
        "tas_file": None,
        "precip_file": None,
        "precip_var": None,
        "amoc_segments": [{"variable": "hosing_0.1Sv_greenland", "year_start": 1}],
        "amoc_note": "No gridded tas; AMOC on year index 1-100 (not calendar years).",
    },
]
