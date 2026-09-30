"""Load CESM1.2 annual-mean CAM output and normalize it to CMIP variable names.

Input format (``data/input/*.nc``): one file per simulation, holding **annual
means** (already time-averaged upstream with CDO) of 39 raw CAM history fields on
the ``f19g16`` finite-volume atmosphere grid (``lat`` = 96 incl. the poles x
``lon`` = 144, 1.89 deg x 2.5 deg). Dimensions are ``(time, lat, lon)``, one step
per year. The files come from the B1850CN (CESM1.2) NAHosMIP experiment set: a
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
PRECT matches the global mean QFLX evaporation, ~2.86 mm/day). All water fluxes
(precipitation, snowfall, evaporation, P - E) are expressed in mm/day of liquid
water: m s-1 times 1000 mm/m x 86400 s/day, and QFLX (kg m-2 s-1) times 86400,
since 1 kg m-2 of water is a 1 mm layer. Provenance attributes
record the source file, variable, and conversion.

AMOC strength is not in the gridded files; it is read from the separate CSV
``AMOC_FILE`` (see ``amoc_strength_on_years``).
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

SOURCE_ID = "CESM1"

# Directory layout relative to the repository root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(_REPO_ROOT, "data", "input")
PROCESSED_DIR = os.path.join(_REPO_ROOT, "data", "processed")

# Water fluxes are reported in mm/day of liquid water. CAM precipitation rates are
# m s-1 (x 1000 mm/m x 86400 s/day); QFLX is a mass flux in kg m-2 s-1, and 1 kg m-2
# of water is a 1 mm layer, so it needs only the x 86400 s/day.
SECONDS_PER_DAY = 86400.0
M_PER_S_TO_MM_PER_DAY = 1000.0 * SECONDS_PER_DAY
KG_M2_S_TO_MM_PER_DAY = SECONDS_PER_DAY
WATER_FLUX_UNITS = "mm/day"

# Analysis variables. Each has a ``definition`` (a formula in CAM field names,
# recorded as provenance), a ``compute`` function mapping the raw CAM Dataset to
# the field, and output ``units``/``long_name``. Three CAM fields carry CMIP names
# (tas, prc, pr); the other raw fields keep their CAM names, with water
# fluxes converted to mm/day; derived fields are combinations.
# Source-attribute mislabels corrected here: RHREFHT is labeled "fraction" but is
# in percent (values ~20-110); SRFRAD ("Net radiative flux at surface") is
# FSNS + FLDS (absorbed shortwave plus downwelling longwave), not a net flux.
_CMIP_RENAMED_FIELDS = [
    # (analysis name, CAM field, scale, units, long_name)
    ("tas", "TREFHT", 1.0, "K", "Near-Surface Air Temperature"),
    ("prc", "PRECC", M_PER_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Convective Precipitation"),
    ("pr", "PRECT", M_PER_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Precipitation (total)"),
]
_CAM_FIELDS = [
    # (CAM field, scale, units, long_name)
    ("TREFMNAV", 1.0, "K", "Average of TREFHT daily minimum"),
    ("TREFMXAV", 1.0, "K", "Average of TREFHT daily maximum"),
    ("QREFHT", 1.0, "kg/kg", "Reference height specific humidity"),
    ("RHREFHT", 1.0, "%", "Reference height relative humidity"),
    ("PRECL", M_PER_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Large-scale (stable) precipitation (liq + ice)"),
    ("PRECSH", M_PER_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Shallow convection precipitation"),
    ("PRECSC", M_PER_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Convective snowfall (water equivalent)"),
    ("PRECSL", M_PER_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Large-scale (stable) snowfall (water equivalent)"),
    ("QFLX", KG_M2_S_TO_MM_PER_DAY, WATER_FLUX_UNITS, "Surface water flux (evaporation)"),
    ("LHFLX", 1.0, "W/m2", "Surface latent heat flux (upward)"),
    ("SHFLX", 1.0, "W/m2", "Surface sensible heat flux (upward)"),
    ("FSDS", 1.0, "W/m2", "Downwelling solar flux at surface"),
    ("FSDSC", 1.0, "W/m2", "Clearsky downwelling solar flux at surface"),
    ("FSNS", 1.0, "W/m2", "Net solar flux at surface (downward)"),
    ("FSNSC", 1.0, "W/m2", "Clearsky net solar flux at surface (downward)"),
    ("FLDS", 1.0, "W/m2", "Downwelling longwave flux at surface"),
    ("FLDSC", 1.0, "W/m2", "Clearsky downwelling longwave flux at surface"),
    ("FLNS", 1.0, "W/m2", "Net longwave flux at surface (upward)"),
    ("FLNSC", 1.0, "W/m2", "Clearsky net longwave flux at surface (upward)"),
    ("SRFRAD", 1.0, "W/m2", "Absorbed solar plus downwelling longwave at surface (FSNS + FLDS)"),
    ("SOLIN", 1.0, "W/m2", "Solar insolation"),
    ("FSNT", 1.0, "W/m2", "Net solar flux at top of model (downward)"),
    ("FSNTC", 1.0, "W/m2", "Clearsky net solar flux at top of model (downward)"),
    ("FLNT", 1.0, "W/m2", "Net longwave flux at top of model (upward)"),
    ("FLNTC", 1.0, "W/m2", "Clearsky net longwave flux at top of model (upward)"),
    ("FLUT", 1.0, "W/m2", "Upwelling longwave flux at top of model"),
    ("FLUTC", 1.0, "W/m2", "Clearsky upwelling longwave flux at top of model"),
    ("SWCF", 1.0, "W/m2", "Shortwave cloud forcing"),
    ("LWCF", 1.0, "W/m2", "Longwave cloud forcing"),
    ("CLDTOT", 1.0, "fraction", "Vertically-integrated total cloud"),
    ("CLDLOW", 1.0, "fraction", "Vertically-integrated low cloud"),
    ("CLDMED", 1.0, "fraction", "Vertically-integrated mid-level cloud"),
    ("CLDHGH", 1.0, "fraction", "Vertically-integrated high cloud"),
    ("TMQ", 1.0, "kg/m2", "Total (vertically integrated) precipitable water"),
    ("SNOWHICE", 1.0, "m", "Water equivalent snow depth over sea ice"),
    ("SNOWHLND", 1.0, "m", "Water equivalent snow depth over land"),
]
_DERIVED_FIELDS = [
    # (analysis name, definition, compute, units, long_name)
    ("pr_minus_evap", "PRECT * 8.64e7 - QFLX * 86400",
     lambda ds: ds["PRECT"] * M_PER_S_TO_MM_PER_DAY - ds["QFLX"] * KG_M2_S_TO_MM_PER_DAY,
     WATER_FLUX_UNITS, "Precipitation minus evaporation"),
    ("prsn", "(PRECSC + PRECSL) * 8.64e7",
     lambda ds: (ds["PRECSC"] + ds["PRECSL"]) * M_PER_S_TO_MM_PER_DAY,
     WATER_FLUX_UNITS, "Snowfall (water equivalent)"),
    ("toa_net_down", "FSNT - FLNT",
     lambda ds: ds["FSNT"] - ds["FLNT"],
     "W/m2", "Net downward radiation at top of model"),
    ("sfc_net_energy_down", "FSNS - FLNS - LHFLX - SHFLX",
     lambda ds: ds["FSNS"] - ds["FLNS"] - ds["LHFLX"] - ds["SHFLX"],
     "W/m2", "Net downward surface energy flux (radiation + turbulent)"),
    ("cloud_radiative_effect", "SWCF + LWCF",
     lambda ds: ds["SWCF"] + ds["LWCF"],
     "W/m2", "Net cloud radiative effect at top of model"),
    ("diurnal_temperature_range", "TREFMXAV - TREFMNAV",
     lambda ds: ds["TREFMXAV"] - ds["TREFMNAV"],
     "K", "Mean diurnal temperature range"),
    ("planetary_albedo", "1 - FSNT / SOLIN",
     lambda ds: 1 - ds["FSNT"] / ds["SOLIN"],
     "1", "Planetary albedo (annual mean fluxes)"),
]


def _scaled_field(cam_name, scale):
    """``compute`` function returning CAM field ``cam_name`` times ``scale``."""
    return lambda ds: ds[cam_name] * scale


def _definition(cam_name, scale):
    return cam_name if scale == 1.0 else f"{cam_name} * {scale:g}"


VARIABLES = {
    **{name: {"definition": _definition(cam, scale), "compute": _scaled_field(cam, scale),
              "units": units, "long_name": long_name}
       for name, cam, scale, units, long_name in _CMIP_RENAMED_FIELDS},
    **{cam: {"definition": _definition(cam, scale), "compute": _scaled_field(cam, scale),
             "units": units, "long_name": long_name}
       for cam, scale, units, long_name in _CAM_FIELDS},
    **{name: {"definition": definition, "compute": compute,
              "units": units, "long_name": long_name}
       for name, definition, compute, units, long_name in _DERIVED_FIELDS},
}

# Named variable sets for scripts that loop over variables, so a quick test can
# run a few fields instead of all 46 (``--variables minimal``). ``minimal`` is
# surface temperature and total precipitation; ``key`` adds convective precipitation
# and the main water-cycle, humidity, cloud and energy-budget fields. Every variable is also its own set, so
# a CLI can mix set names and variable names (see ``resolve_variables``).
VARIABLE_SETS = {
    **{var: [var] for var in VARIABLES},
    "minimal": ["tas", "pr"],
    "key": [
        "tas", "diurnal_temperature_range", "pr", "prc", "pr_minus_evap", "prsn",
        "RHREFHT", "TMQ", "CLDTOT", "cloud_radiative_effect", "toa_net_down",
        "sfc_net_energy_down", "planetary_albedo",
    ],
    "all": list(VARIABLES),
}


def resolve_variables(names):
    """Expand set and variable names (``VARIABLE_SETS`` keys) into an ordered,
    duplicate-free list of ``VARIABLES`` keys."""
    return list(dict.fromkeys(var for name in names for var in VARIABLE_SETS[name]))


def add_variables_argument(parser):
    """Add the shared ``--variables`` option (``VARIABLE_SETS`` names, default
    ``all``) to an ``argparse`` parser; expand it with ``resolve_variables``."""
    parser.add_argument(
        "--variables", nargs="+", default=["all"], choices=list(VARIABLE_SETS),
        metavar="NAME",
        help="set names (minimal, key, all) and/or variable names (default: all)",
    )



# One input file per simulation, in case-grid order (see ``CASE_GRID``), with
# the run's column label in ``AMOC_FILE``.
# ``co2_multiple`` is the CO2 concentration relative to preindustrial and
# ``hosing_sv`` the North Atlantic freshwater forcing (Sv; negative = freshwater
# removal), both read from the CESM case names in the file names. Year ranges are
# the labels written by ``cdo settaxis``: the 1xCO2 control is 1850-2150 and every
# other run starts in 2051 (control year 201, cf. the ``yr200`` in the no-hosing
# file names).
INPUT_FILES = [
    {"file": "B1850CN_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "amoc_column": "1x CO2, -0.3 Sv", "co2_multiple": 1, "hosing_sv": -0.3},
    {"file": "B1850CN_f19g16_GCC_piCtrl300yr_annual_mean.nc", "amoc_column": "piControl", "co2_multiple": 1, "hosing_sv": 0.0},
    {"file": "B1850CN_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "amoc_column": "1x CO2, +0.3 Sv", "co2_multiple": 1, "hosing_sv": 0.3},
    {"file": "B1850CN_2xCO2_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "amoc_column": "2x CO2, -0.3 Sv", "co2_multiple": 2, "hosing_sv": -0.3},
    {"file": "B1850CN_2xCO2_noh_f19g16_yr200_annual_mean.nc", "amoc_column": "2x CO2, no hosing", "co2_multiple": 2, "hosing_sv": 0.0},
    {"file": "B1850CN_2xCO2_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "amoc_column": "2x CO2, +0.3 Sv", "co2_multiple": 2, "hosing_sv": 0.3},
    {"file": "B1850CN_4xCO2_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "amoc_column": "4x CO2, -0.3 Sv", "co2_multiple": 4, "hosing_sv": -0.3},
    {"file": "B1850CN_4xCO2_noh_f19g16_yr200_annual_mean.nc", "amoc_column": "4x CO2, no hosing", "co2_multiple": 4, "hosing_sv": 0.0},
    {"file": "B1850CN_4xCO2_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc", "amoc_column": "4x CO2, +0.3 Sv", "co2_multiple": 4, "hosing_sv": 0.3},
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


def case_grid_time_mean(var, first_year, last_year):
    """Time mean of ``var`` over calendar years ``first_year``..``last_year``
    (inclusive) for every case, as a ``(co2, hosing, lat, lon)`` DataArray laid
    out like ``CASE_GRID`` (``co2`` = CO2 multiple, ``hosing`` = hosing in Sv).

    Every case must cover the full year range (``sel`` raises otherwise).
    """
    years = np.arange(first_year, last_year + 1)
    rows = [
        xr.concat(
            [load_annual_field(case, var).sel(year=years).mean("year") for case in row],
            dim=pd.Index(HOSING_LEVELS, name="hosing"),
        )
        for row in CASE_GRID
    ]
    grid = xr.concat(rows, dim=pd.Index(CO2_LEVELS, name="co2"))
    spec = VARIABLES[var]
    grid.attrs = {"units": spec["units"], "long_name": spec["long_name"],
                  "definition": spec["definition"],
                  "time_mean": f"{first_year}-{last_year}"}
    return grid.rename(var)

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
    ``experiment``, in the analysis units, with provenance attributes."""
    spec = VARIABLES[var]
    field = spec["compute"](open_experiment(experiment)).rename(var)
    field.attrs = {
        "units": spec["units"],
        "long_name": spec["long_name"],
        "definition": spec["definition"],
        "source_file": EXPERIMENTS[experiment]["file"],
    }
    return field


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

# Annual-mean AMOC strength (Sv) at 26.5 N: a ``year`` column (2051-2150) plus one
# column per run, labeled as in ``INPUT_FILES[...]["amoc_column"]``. Its years use
# the same labels as the gridded files (the control's first AMOC value equals the
# hosing runs' first-year values, i.e. 2051 is the branch year in both).
AMOC_FILE = "amoc_timeseries_26p5N_9experiments_v5_annual.csv"


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

    Reads the run's column of ``AMOC_FILE`` and aligns it by calendar year; years
    the AMOC series does not cover are left missing (NaN), so the regressions drop
    them by complete-case deletion.
    """
    table = pd.read_csv(os.path.join(INPUT_DIR, AMOC_FILE), index_col="year")
    series = table[EXPERIMENTS[experiment]["amoc_column"]].reindex(np.asarray(years))
    return xr.DataArray(
        series.values, coords={"year": np.asarray(years)}, dims="year",
        name="amoc_strength",
    )
