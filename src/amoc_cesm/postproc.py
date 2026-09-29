"""The NAHosMIP_v2 protocol: five years, four components, full ocean transports.

`data/input/postproc` holds a second, richer delivery of the six hosed runs --
39 atmospheric, 14 land, 11 sea-ice and 11 ocean variables, each as an annual
and a seasonal mean -- but only for model years 2051-2055. It complements
`SALT_extracted` (100 years, surface salinity only) rather than replacing it.

Two properties of these files decide how they are read here.

**The annual means are Dec-Nov years, not calendar years.** CESM stamps a
monthly mean with the *end* of its interval, so `cdo selyear,2051/2055` picks up
December 2050 through November 2051 as "2051". Every annual record therefore
spans day 334 to day 334. This is not new -- the `Annual_Mean_2D_Fileds_ATMs`
files use the same convention -- so the two are directly comparable, but neither
is a calendar year. The one exception is the *first ocean record*, which spans
334 days (January-November 2051) because the ocean archive has no December 2050:
that record is an 11-month mean and is flagged rather than silently averaged in.

**The `case` global attribute is wrong.** Every file inherits
`case = "B1850CN_f19g16_GCC_piCtrl300yr"` from the branch parent. `source_case`
carries the real identity, which is what `POSTPROC_CASES` is keyed on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .regrid import ATM_NLAT, ATM_NLON, pop_grid, regrid_pop_file

POSTPROC_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "postproc"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "output" / "postproc_v2"

COMPONENTS = ("atm", "ice", "lnd", "ocn")
FREQUENCIES = ("annual", "seasonal")

#: Case label -> directory name. Labels follow the same scheme as the SALT
#: extracts, with a `_v2` suffix because these are a third set of integrations:
#: their year-2051 fields match neither the `NAHosMIP_FIX` nor the `yr200`
#: delivery of the same nominal experiment.
POSTPROC_CASES: dict[str, str] = {
    "1xCO2_neghos_v2": "B1850CN_neghos0p3Sv_f19g16_NAHosMIP_v2",
    "1xCO2_poshos_v2": "B1850CN_hos0p3Sv_f19g16_NAHosMIP_v2",
    "2xCO2_neghos_v2": "B1850CN_2xCO2_neghos0p3Sv_f19g16_NAHosMIP_v2",
    "2xCO2_poshos_v2": "B1850CN_2xCO2_hos0p3Sv_f19g16_NAHosMIP_v2",
    "4xCO2_neghos_v2": "B1850CN_4xCO2_neghos0p3Sv_f19g16_NAHosMIP_v2",
    "4xCO2_poshos_v2": "B1850CN_4xCO2_hos0p3Sv_f19g16_NAHosMIP_v2",
}

CASE_FORCING: dict[str, tuple[int, float]] = {
    "1xCO2_neghos_v2": (1, -0.3), "1xCO2_poshos_v2": (1, +0.3),
    "2xCO2_neghos_v2": (2, -0.3), "2xCO2_poshos_v2": (2, +0.3),
    "4xCO2_neghos_v2": (4, -0.3), "4xCO2_poshos_v2": (4, +0.3),
}

#: An annual record shorter than this is a partial year (see the module note).
FULL_YEAR_DAYS = 360

# AMOC index definition: the strongest Atlantic overturning cell, taken below
# the wind-driven surface layer and inside the latitude band where the deep
# cell lives. 500 m and 20-60N are the conventional choices; 26.5N is reported
# separately because that is the latitude the RAPID array observes.
AMOC_MIN_DEPTH_M = 500.0
AMOC_LAT_BAND = (20.0, 60.0)
RAPID_LAT = 26.5


def case_dir(case: str) -> Path:
    return POSTPROC_DIR / POSTPROC_CASES[case]


def var_path(case: str, component: str, var: str, freq: str = "annual") -> Path:
    return case_dir(case) / component / f"{var}_{freq}.nc"


def variables_in(case: str, component: str) -> list[str]:
    """Variable names archived for one component of one case."""
    return sorted({p.name.rsplit("_", 1)[0]
                   for p in (case_dir(case) / component).glob("*_annual.nc")})


def open_var(case: str, component: str, var: str, freq: str = "annual") -> xr.Dataset:
    return xr.open_dataset(var_path(case, component, var, freq))


def _interval_days(ds_raw: xr.Dataset) -> np.ndarray:
    bounds = ds_raw[ds_raw["time"].attrs["bounds"]].values
    return np.asarray(bounds[:, 1] - bounds[:, 0], dtype=float)


def _tag(ds: xr.Dataset, case: str, freq: str, source: Path) -> xr.Dataset:
    co2, hosing = CASE_FORCING[case]
    ds.attrs.update(
        case=case, source_case=POSTPROC_CASES[case], co2=co2, hosing_Sv=hosing,
        frequency=freq, source_file=str(source.relative_to(POSTPROC_DIR.parents[2])),
        year_convention="annual means span December-November, not calendar years",
    )
    return ds


def _flag_partial(ds: xr.Dataset, ds_raw: xr.Dataset, freq: str) -> xr.Dataset:
    """Mark records whose averaging interval is short of a full year.

    Only the first ocean record is affected, but it is flagged as data rather
    than described in a comment so that anything averaging over time can drop it
    deliberately instead of absorbing an 11-month mean as if it were 12.
    """
    days = _interval_days(ds_raw)
    ds = ds.assign_coords(interval_days=("time", days))
    if freq == "annual":
        ds = ds.assign_coords(partial_year=("time", days < FULL_YEAR_DAYS))
    return ds


def regrid_sss(case: str, freq: str = "annual", grid: xr.Dataset | None = None) -> xr.Dataset:
    """Surface salinity of one case, binned onto the CAM 144x96 grid.

    The `postproc` ocean files carry `TLAT`/`TLONG` but neither `TAREA` nor
    `KMT`, so the grid comes from a file that has all four -- gx1v6 is the same
    grid in every run.
    """
    source = var_path(case, "ocn", "SALT", freq)
    grid = grid if grid is not None else pop_grid()
    out = regrid_pop_file(source, var="SALT", grid=grid, level=0)
    out = out.rename({"SALT": "SSS"})
    out["SSS"].attrs["long_name"] = "Sea surface salinity (5 m)"
    out = _flag_partial(out, xr.open_dataset(source, decode_times=False), freq)
    return _tag(out, case, freq, source)


def _named(values: np.ndarray) -> list[str]:
    """POP stores region and component names as padded byte strings."""
    return [v.decode().strip() if isinstance(v, bytes) else str(v).strip() for v in values]


def transports(case: str, freq: str = "annual") -> xr.Dataset:
    """MOC, N_HEAT and N_SALT with readable coordinates, plus AMOC indices.

    The raw files index region and component by integer, with the meanings in
    separate byte-string variables; here they become string coordinates, so a
    selection reads `sel(transport_reg="Atlantic ...")` rather than `isel(1)`.
    Depth is converted from centimetres to metres.

    Three scalar indices come out of the Atlantic streamfunction. `amoc_max`
    and `amoc_rapid_26n` sum all three overturning components -- Eulerian mean
    plus the bolus and submesoscale parameterisations -- which is the total mass
    transport the model achieves. `amoc_max_eulerian` takes the Eulerian mean
    alone; that is the definition behind the `verification/*_amoc_index.csv`
    files shipped with this delivery, which it reproduces exactly. The two
    differ by about 0.15 Sv, the eddy parameterisations opposing the mean flow.
    """
    moc_src = var_path(case, "ocn", "MOC", freq)
    moc_ds = xr.open_dataset(moc_src)
    heat_ds = xr.open_dataset(var_path(case, "ocn", "N_HEAT", freq))
    salt_ds = xr.open_dataset(var_path(case, "ocn", "N_SALT", freq))

    # The MOC file names its overturning components but not its regions; the
    # transport files name both, and POP uses one region list for all three.
    regions = _named(heat_ds["transport_regions"].values)
    assert moc_ds.sizes["transport_reg"] == len(regions) == 2, (
        f"{case}: expected 2 transport regions, got {moc_ds.sizes['transport_reg']}"
    )
    atlantic = regions[1]
    assert atlantic.startswith("Atlantic"), f"{case}: region 1 is {atlantic!r}, not the Atlantic"

    moc = moc_ds["MOC"].assign_coords(
        transport_reg=("transport_reg", regions),
        moc_comp=("moc_comp", _named(moc_ds["moc_components"].values)),
        moc_z=("moc_z", moc_ds["moc_z"].values * 1e-2),
    )
    moc["moc_z"].attrs = {"units": "m", "long_name": "depth", "positive": "down"}

    def name_transport(da, ds):
        return da.assign_coords(
            transport_reg=("transport_reg", regions),
            transport_comp=("transport_comp", _named(ds["transport_components"].values)),
        )

    out = xr.Dataset({
        "MOC": moc,
        "N_HEAT": name_transport(heat_ds["N_HEAT"], heat_ds),
        "N_SALT": name_transport(salt_ds["N_SALT"], salt_ds),
    })

    deep = moc.sel(transport_reg=atlantic).sum("moc_comp").sel(
        moc_z=slice(AMOC_MIN_DEPTH_M, None)
    )
    out["amoc_max"] = deep.sel(lat_aux_grid=slice(*AMOC_LAT_BAND)).max(
        ("moc_z", "lat_aux_grid")
    )
    out["amoc_max"].attrs = {
        "units": "Sv",
        "long_name": (f"Maximum Atlantic overturning below {AMOC_MIN_DEPTH_M:.0f} m, "
                      f"{AMOC_LAT_BAND[0]:.0f}-{AMOC_LAT_BAND[1]:.0f}N"),
    }
    out["amoc_rapid_26n"] = deep.sel(lat_aux_grid=RAPID_LAT, method="nearest").max("moc_z")
    out["amoc_rapid_26n"].attrs = {
        "units": "Sv",
        "long_name": f"Maximum Atlantic overturning below {AMOC_MIN_DEPTH_M:.0f} m at 26.5N",
    }

    out["amoc_max_eulerian"] = (
        moc.sel(transport_reg=atlantic, moc_comp="Eulerian Mean")
        .sel(moc_z=slice(AMOC_MIN_DEPTH_M, None))
        .sel(lat_aux_grid=slice(*AMOC_LAT_BAND))
        .max(("moc_z", "lat_aux_grid"))
    )
    out["amoc_max_eulerian"].attrs = {
        "units": "Sv",
        "long_name": "Maximum Atlantic Eulerian-mean overturning "
                     f"below {AMOC_MIN_DEPTH_M:.0f} m, "
                     f"{AMOC_LAT_BAND[0]:.0f}-{AMOC_LAT_BAND[1]:.0f}N",
        "comment": "matches verification/<case>_amoc_index.csv",
    }

    out = _flag_partial(out, xr.open_dataset(moc_src, decode_times=False), freq)
    out.attrs["atlantic_region"] = atlantic
    return _tag(out, case, freq, moc_src)


def sss_path(case: str, freq: str = "annual") -> Path:
    return OUT_DIR / "sss" / f"{case}_SSS_{freq}_{ATM_NLON}x{ATM_NLAT}.nc"


def transports_path(case: str, freq: str = "annual") -> Path:
    return OUT_DIR / "transports" / f"{case}_transports_{freq}.nc"


def load_sss(case: str, freq: str = "annual") -> xr.Dataset:
    return xr.open_dataset(sss_path(case, freq))


def load_transports(case: str, freq: str = "annual") -> xr.Dataset:
    return xr.open_dataset(transports_path(case, freq))
