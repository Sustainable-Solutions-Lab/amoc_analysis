"""Load CESM1 annual-mean CAM output and normalize it to CMIP variable names.

Input format (``data/input/*.nc``): one file per simulation, holding **annual
means** (already time-averaged upstream with CDO) of 39 raw CAM history fields on
the ``f19g16`` finite-volume atmosphere grid (``lat`` = 96 incl. the poles x
``lon`` = 144, 1.89 deg x 2.5 deg). Dimensions are ``(time, lat, lon)``, one step
per year. The files come from the B1850CN (CESM1) NAHosMIP experiment set: a
preindustrial control plus 1x/2x/4xCO2 runs with 0, +0.3 Sv, or -0.3 Sv North
Atlantic freshwater hosing (see ``EXPERIMENTS``).

The ``time`` axis is ``"years since YYYY-7-2"`` on a ``365_day`` calendar, written
by ``cdo settaxis`` -- CF-``years`` units that xarray/cftime cannot decode -- so
files are opened with ``decode_times=False`` and ``time`` is converted to an
integer calendar ``year`` (reference year + offset). Every file has a regular
1-year step.

This module maps the CAM names used by the analysis to CMIP names (see
``VARIABLES``): ``TREFHT`` -> ``tas`` (K), ``PRECC`` -> ``prc`` (convective
precipitation), ``PRECT`` -> ``pr`` (total precipitation). CAM precipitation is a
liquid-water-equivalent rate in m s-1 (the units label is genuine: the global mean
PRECT matches the global mean QFLX evaporation, ~2.86 mm/day), so it is multiplied
by the density of water, 1000 kg m-3, to give kg m-2 s-1. Provenance attributes
record the source file, variable, and conversion.

AMOC strength is not in the gridded files; it is read from the separate
``AMOC_FILE`` (see ``amoc_strength_on_years``).
"""

import os

import numpy as np
import xarray as xr

SOURCE_ID = "CESM1"

# Directory layout relative to the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(_REPO_ROOT, "data", "input")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

# Density of liquid water (kg m-3): converts CAM precipitation rates (m s-1) to
# the CMIP water mass flux (kg m-2 s-1).
WATER_DENSITY = 1000.0

# Analysis variable (CMIP name) -> CAM source variable, multiplicative conversion
# to the output units, and output metadata.
VARIABLES = {
    "tas": {
        "source_variable": "TREFHT",
        "scale": 1.0,
        "units": "K",
        "long_name": "Near-Surface Air Temperature",
    },
    "prc": {
        "source_variable": "PRECC",
        "scale": WATER_DENSITY,
        "units": "kg m-2 s-1",
        "long_name": "Convective Precipitation",
        "precip_kind": "convective",
    },
    "pr": {
        "source_variable": "PRECT",
        "scale": WATER_DENSITY,
        "units": "kg m-2 s-1",
        "long_name": "Precipitation",
        "precip_kind": "total",
    },
}

# One input file per simulation, in case-grid order (see ``CASE_GRID``).
# ``co2_multiple`` is the CO2 concentration relative to preindustrial and
# ``hosing_sv`` the North Atlantic freshwater forcing (Sv; negative = freshwater
# removal), both read from the CESM case names in the file names. Year ranges are
# the labels written by ``cdo settaxis``: the 1xCO2 control is 1850-2150 and every
# other run starts in 2051 (control year 201, cf. the ``yr200`` in the no-hosing
# file names).
INPUT_FILES = [
    {"file": "B1850CN_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "co2_multiple": 1, "hosing_sv": -0.3},
    {"file": "B1850CN_f19g16_GCC_piCtrl300yr_annual_mean.nc", "co2_multiple": 1, "hosing_sv": 0.0},
    {"file": "B1850CN_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "co2_multiple": 1, "hosing_sv": 0.3},
    {"file": "B1850CN_2xCO2_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "co2_multiple": 2, "hosing_sv": -0.3},
    {"file": "B1850CN_2xCO2_noh_f19g16_yr200_annual_mean.nc", "co2_multiple": 2, "hosing_sv": 0.0},
    {"file": "B1850CN_2xCO2_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "co2_multiple": 2, "hosing_sv": 0.3},
    {"file": "B1850CN_4xCO2_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "co2_multiple": 4, "hosing_sv": -0.3},
    {"file": "B1850CN_4xCO2_noh_f19g16_yr200_annual_mean.nc", "co2_multiple": 4, "hosing_sv": 0.0},
    {"file": "B1850CN_4xCO2_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "co2_multiple": 4, "hosing_sv": 0.3},
]

# Case-name suffix for each hosing level: m = -0.3 Sv, none = 0 Sv, p = +0.3 Sv.
HOSING_SUFFIX = {-0.3: "_m03Sv", 0.0: "", 0.3: "_p03Sv"}


def case_name(co2_multiple, hosing_sv):
    """Case name ``[124]xCO2[_m03Sv|_p03Sv]`` (e.g. 2, -0.3 -> ``2xCO2_m03Sv``)."""
    return f"{co2_multiple}xCO2{HOSING_SUFFIX[hosing_sv]}"


# Case name -> input file spec; the case name labels the run everywhere downstream.
EXPERIMENTS = {
    case_name(spec["co2_multiple"], spec["hosing_sv"]): spec for spec in INPUT_FILES
}

# Layout for any figure mapping several cases: rows are CO2 levels (1, 2, 4x),
# columns are hosing levels (-0.3, 0, +0.3 Sv). CASE_GRID[row][col] is a case name.
CO2_LEVELS = [1, 2, 4]
HOSING_LEVELS = [-0.3, 0.0, 0.3]
CASE_GRID = [[case_name(co2, hos) for hos in HOSING_LEVELS] for co2 in CO2_LEVELS]

# Each entry produces exactly one processed file: annual_file(var, experiment).
INPUT_MANIFEST = [
    {"var": var, "experiment": experiment}
    for experiment in EXPERIMENTS
    for var in VARIABLES
]


def annual_file(var, experiment):
    """Processed gridded annual-mean file name for ``var`` in ``experiment``."""
    return f"{var}_annual_{SOURCE_ID}_{experiment}.nc"


def scalar_file(experiment):
    """Processed per-simulation scalar time-series file name."""
    return f"scalars_annual_{SOURCE_ID}_{experiment}.nc"


def _years_from_time(time):
    """Integer calendar years from a ``"years since YYYY-M-D"`` time axis.

    Each value is a whole-year offset from the reference date (cdo stamps each
    annual mean at the same day-of-year as the reference), so the calendar year
    is the reference year plus the offset.
    """
    reference_year = int(time.attrs["units"].split("since")[1].split("-")[0])
    return reference_year + time.values.astype(int)


def open_experiment(experiment):
    """Open one simulation's annual-mean file with an integer ``year`` dimension.

    Returns the raw CAM Dataset (native variable names and units) with ``time``
    replaced by ``year``, so any of its 39 fields can be selected directly.
    """
    path = os.path.join(INPUT_DIR, EXPERIMENTS[experiment]["file"])
    ds = xr.open_dataset(path, decode_times=False)
    years = _years_from_time(ds["time"])
    return ds.rename(time="year").assign_coords(year=years)


def load_annual_field(experiment, var):
    """Annual-mean ``(year, lat, lon)`` field ``var`` (a ``VARIABLES`` key) for
    ``experiment``, converted to CMIP units, with provenance attributes."""
    spec = VARIABLES[var]
    source = open_experiment(experiment)[spec["source_variable"]]
    field = (source * spec["scale"]).rename(var)
    field.attrs = {
        **{k: v for k, v in spec.items() if k not in ("source_variable", "scale")},
        "source_file": EXPERIMENTS[experiment]["file"],
        "source_variable": spec["source_variable"],
        "original_units": source.attrs["units"],
        "conversion_factor": spec["scale"],
    }
    return field


def load_and_normalize(entry):
    """Build the annual-mean DataArray for one ``INPUT_MANIFEST`` entry."""
    return load_annual_field(entry["experiment"], entry["var"])


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


# --- Scalar (one-value-per-year) diagnostics -------------------------------

# Precomputed AMOC strength for these runs, supplied separately from the gridded
# files. Expected layout: one variable per case name (``EXPERIMENTS`` key), each on an integer
# calendar ``year`` coordinate matching the gridded files' year labels, in Sv.
AMOC_FILE = "AMOC_CESM1_B1850CN_f19g16.nc"


def latitude_band_weights(lat):
    """Exact zonal-band area weights for a regular (ascending) latitude grid.

    Each cell's weight is proportional to its meridional band area,
    ``sin(edge_north) - sin(edge_south)``, with cell edges taken as the
    midpoints between adjacent centers and the outermost edges clamped to ±90°.
    This is exact for a regular lon×lat grid and correctly accounts for the
    FV grid's half-width polar cells (unlike ``cos(lat)``, which zeros the
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


def amoc_strength_on_years(experiment, years):
    """AMOC strength (Sv) for ``experiment`` placed onto ``years``.

    Reads the ``experiment`` variable of ``AMOC_FILE`` and aligns it by calendar
    year; years the AMOC series does not cover are left missing (NaN), so the
    regressions drop them by complete-case deletion.
    """
    amoc = xr.open_dataset(os.path.join(INPUT_DIR, AMOC_FILE))[experiment]
    return amoc.reindex(year=np.asarray(years)).rename("amoc_strength")


# One scalar file per simulation, on the run's gridded year axis. Every run has
# gridded tas and precipitation; ``precip_var`` selects the precipitation source
# for the ITCZ centroid diagnostic (``precip_centroid_lat_*``).
SCALAR_SIMULATIONS = [
    {
        "experiment": experiment,
        "tas_file": annual_file("tas", experiment),
        "precip_file": annual_file("prc", experiment),
        "precip_var": "prc",
    }
    for experiment in EXPERIMENTS
]
