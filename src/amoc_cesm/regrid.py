"""Regrid POP ocean fields (gx1v6, 384x320 curvilinear) onto the CAM f19 grid.

The ocean files in ``data/input/SALT_extracted`` are on the displaced-pole POP
grid; every other field in this project is on the 144x96 CAM finite-volume grid.
This module bins the ocean cells onto that grid so the two can be differenced
and plotted together.

Method: first-order area-weighted binning. Each POP T-cell is assigned whole to
the CAM cell containing its centre (``TLAT``, ``TLONG``) and averaged with
``TAREA`` weights. This conserves the area-weighted mean exactly but not the
cell-by-cell overlap: a POP cell straddling a CAM cell edge is counted entirely
on one side. Exact conservative remapping would need the POP cell *corners*
(``ULAT``/``ULONG``), which these extracted files do not carry. The error is
small here because POP gx1v6 (~1 deg, finer near the equator and in the Arctic)
is roughly four times finer than CAM f19 (2.5 x 1.9 deg), so a typical target
cell averages 8-30 source cells -- see ``source_cell_counts``.

Land is handled by the ocean-area weighting: the output at each CAM cell is the
mean over the ocean part of that cell only, and cells with no ocean are NaN.
The ocean area that went into each cell is returned alongside as ``ocean_area``,
which is the correct weight for area-averaging the regridded field (better than
cos(lat), which would ignore the land fraction).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse

#: CAM finite-volume 1.9x2.5 grid, as used by every atmospheric field here.
ATM_NLAT = 96
ATM_NLON = 144


def atm_grid() -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes and longitudes of the CAM f19 grid."""
    lat = np.linspace(-90.0, 90.0, ATM_NLAT)
    lon = np.arange(ATM_NLON) * (360.0 / ATM_NLON)
    return lat, lon


def _lat_edges(lat: np.ndarray) -> np.ndarray:
    """Cell edges for the CAM latitude axis: pole cells are half-width."""
    return np.concatenate([[-90.0], 0.5 * (lat[:-1] + lat[1:]), [90.0]])


def target_index(tlat: np.ndarray, tlon: np.ndarray) -> np.ndarray:
    """Flat CAM-cell index (ilat * ATM_NLON + ilon) for each POP cell centre."""
    lat, lon = atm_grid()
    dlon = lon[1] - lon[0]

    ilat = np.searchsorted(_lat_edges(lat), tlat, side="right") - 1
    ilat = np.clip(ilat, 0, ATM_NLAT - 1)
    # Longitude is periodic and uniform; the first cell is centred on 0, so its
    # western edge is at -dlon/2 and cells past 360-dlon/2 wrap back into it.
    ilon = np.floor((tlon + 0.5 * dlon) / dlon).astype(int) % ATM_NLON
    return ilat * ATM_NLON + ilon


def _weight_matrix(ds: xr.Dataset) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Sparse (n_target, n_ocean) area-weight matrix, ocean area, ocean mask.

    ``TAREA`` is in cm^2 in the POP output; areas are returned in m^2.
    """
    ocean = ds["KMT"].values > 0
    idx = target_index(ds["TLAT"].values[ocean], ds["TLONG"].values[ocean])
    area = ds["TAREA"].values[ocean] * 1e-4  # cm^2 -> m^2

    n_src = idx.size
    n_tgt = ATM_NLAT * ATM_NLON
    weights = sparse.csr_matrix(
        (area, (idx, np.arange(n_src))), shape=(n_tgt, n_src)
    )
    ocean_area = np.asarray(weights.sum(axis=1)).ravel()
    return weights, ocean_area, ocean


def source_cell_counts(ds: xr.Dataset) -> xr.DataArray:
    """How many POP ocean cells fall in each CAM cell -- a regridding diagnostic."""
    ocean = ds["KMT"].values > 0
    idx = target_index(ds["TLAT"].values[ocean], ds["TLONG"].values[ocean])
    counts = np.bincount(idx, minlength=ATM_NLAT * ATM_NLON).reshape(ATM_NLAT, ATM_NLON)
    lat, lon = atm_grid()
    return xr.DataArray(counts, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"),
                        name="source_cells")


def _midpoint_time(ds_raw: xr.Dataset) -> xr.DataArray:
    """Interval-midpoint time from ``time_bound``.

    POP stamps each monthly mean with the *end* of its averaging interval, so
    the January 2051 mean carries the time 2051-02-01. Grouping by ``.dt.year``
    on that axis would put December of each year into the following year. The
    midpoint of ``time_bound`` puts every stamp inside its own month.
    """
    bounds = ds_raw["time_bound"]
    mid = bounds.mean(dim=bounds.dims[1]).values
    attrs = ds_raw["time"].attrs
    return xr.coding.times.decode_cf_datetime(
        mid, attrs["units"], attrs["calendar"], use_cftime=True
    )


def regrid_pop_file(
    path: Path | str,
    var: str = "SALT",
    time_chunk: int = 240,
) -> xr.Dataset:
    """Regrid one POP file onto the CAM grid.

    Returns a Dataset with ``<var>(time, lat, lon)`` and ``ocean_area(lat, lon)``.
    The vertical dimension must be a single level (these are 5 m SSS extracts).
    """
    path = Path(path)
    ds = xr.open_dataset(path)
    ds_raw = xr.open_dataset(path, decode_times=False)

    weights, ocean_area, ocean = _weight_matrix(ds)
    empty = ocean_area == 0.0

    da = ds[var]
    if "z_t" in da.dims:
        assert da.sizes["z_t"] == 1, f"{path.name}: expected a single level, got {da.sizes['z_t']}"
        da = da.isel(z_t=0)

    n_time = da.sizes["time"]
    out = np.empty((n_time, ATM_NLAT * ATM_NLON), dtype=np.float32)

    for start in range(0, n_time, time_chunk):
        stop = min(start + time_chunk, n_time)
        chunk = da.isel(time=slice(start, stop)).values.reshape(stop - start, -1)
        chunk = chunk[:, ocean.ravel()]
        # The land mask is time-invariant and identical to KMT == 0 (checked on
        # these files); a NaN inside the ocean mask would silently bias a cell.
        assert np.isfinite(chunk).all(), f"{path.name}: NaN inside the ocean mask at t={start}"
        out[start:stop] = (weights @ chunk.T).T / np.where(empty, np.nan, ocean_area)

    lat, lon = atm_grid()
    time = _midpoint_time(ds_raw)
    result = xr.Dataset(
        {
            var: (("time", "lat", "lon"), out.reshape(n_time, ATM_NLAT, ATM_NLON)),
            "ocean_area": (("lat", "lon"),
                           np.where(empty, np.nan, ocean_area).reshape(ATM_NLAT, ATM_NLON)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    result[var].attrs = {k: v for k, v in ds[var].attrs.items()
                         if k not in ("scale_factor", "_FillValue", "missing_value",
                                      "coordinates", "grid_loc")}
    result[var].attrs["depth_m"] = float(ds["z_t"].values[0]) * 1e-2
    result["ocean_area"].attrs = {"long_name": "ocean area of the CAM cell", "units": "m2"}
    result["lat"].attrs = {"units": "degrees_north", "long_name": "latitude"}
    result["lon"].attrs = {"units": "degrees_east", "long_name": "longitude"}
    result.attrs = {
        "source_file": path.name,
        "regrid_method": "first-order area-weighted binning of POP T-cells "
                         "(TAREA weights) onto the CAM f19 144x96 grid",
        "time_convention": "interval midpoint from time_bound (POP stamps the "
                           "interval end)",
    }
    return result


def annual_mean(ds: xr.Dataset) -> xr.Dataset:
    """Length-of-month weighted annual mean, indexed by integer ``year``.

    Only whole years are kept: a run truncated mid-year would otherwise
    contribute a partial annual mean that looks like a full one.
    """
    static = ds[["ocean_area"]]
    monthly = ds.drop_vars("ocean_area")

    days = monthly["time"].dt.days_in_month
    n_months = days.groupby("time.year").count()
    weighted = (monthly * days).groupby("time.year").sum(skipna=False)
    out = weighted / days.groupby("time.year").sum()

    out = out.sel(year=n_months["year"].values[n_months.values == 12])
    for name, var in monthly.data_vars.items():
        out[name].attrs = var.attrs
    return out.merge(static)


# --- The SALT extracts -------------------------------------------------------
#
# Case label -> source file. The +0.3 Sv (hosing) experiments exist twice, as a
# `yr200` run and as a `NAHosMIP_FIX` run. These are *not* the same run
# truncated differently: they diverge from the first month. The FIX runs are
# canonical and take the plain `*_poshos` names; the `yr200` runs are kept under
# an explicit suffix so the two can still be compared. The other six names line
# up with the case directories under `Annual_Mean_2D_Fileds_ATMs`.

SALT_DIR = Path(__file__).resolve().parents[2] / "data" / "input" / "SALT_extracted"
REGRID_DIR = Path(__file__).resolve().parents[2] / "data" / "output" / "regrid" / "SSS"

SALT_SOURCES: dict[str, str] = {
    "picontrol":           "B1850CN_f19g16_GCC_piCtrl300yr_SSS.nc",
    "1xCO2_neghos":        "B1850CN_neghos0p3Sv_f19g16_yr200_SSS.nc",
    "2xCO2_neghos":        "B1850CN_2xCO2_neghos0p3Sv_f19g16_yr200_SSS.nc",
    "4xCO2_neghos":        "B1850CN_4xCO2_neghos0p3Sv_f19g16_yr200_SSS.nc",
    "2xCO2_noh":           "B1850CN_2xCO2_noh_f19g16_yr200_SSS.nc",
    "4xCO2_noh":           "B1850CN_4xCO2_noh_f19g16_yr200_SSS.nc",
    "1xCO2_poshos":        "B1850CN_hos0p3Sv_f19g16_NAHosMIP_FIX_SSS.nc",
    "2xCO2_poshos":        "B1850CN_2xCO2_hos0p3Sv_f19g16_NAHosMIP_FIX_SSS.nc",
    "4xCO2_poshos":        "B1850CN_4xCO2_hos0p3Sv_f19g16_NAHosMIP_FIX_SSS.nc",
    "1xCO2_poshos_yr200":  "B1850CN_hos0p3Sv_f19g16_yr200_SSS.nc",
    "2xCO2_poshos_yr200":  "B1850CN_2xCO2_hos0p3Sv_f19g16_yr200_SSS.nc",
    "4xCO2_poshos_yr200":  "B1850CN_4xCO2_hos0p3Sv_f19g16_yr200_SSS.nc",
}


def regrid_path(case: str, freq: str = "ann") -> Path:
    return REGRID_DIR / f"{case}_SSS_{freq}_144x96.nc"


def load_sss(case: str, freq: str = "ann") -> xr.Dataset:
    """Load a regridded SSS case: ``freq='ann'`` (yearly) or ``'mon'`` (monthly)."""
    return xr.open_dataset(regrid_path(case, freq))


#: The nine cases of the 3x3 design, using the canonical (FIX) hosing runs.
CANONICAL_CASES: tuple[str, ...] = (
    "1xCO2_neghos", "picontrol", "1xCO2_poshos",
    "2xCO2_neghos", "2xCO2_noh", "2xCO2_poshos",
    "4xCO2_neghos", "4xCO2_noh", "4xCO2_poshos",
)

#: The shorter alternative hosing runs, kept for comparison against the FIX runs.
ALTERNATE_CASES: tuple[str, ...] = (
    "1xCO2_poshos_yr200", "2xCO2_poshos_yr200", "4xCO2_poshos_yr200",
)
