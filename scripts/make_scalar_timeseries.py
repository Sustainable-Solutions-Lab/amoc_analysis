"""Build per-simulation scalar (one-value-per-year) annual time-series files.

For each simulation, writes ``data/processed/scalars_annual_CESM1_{exp}.nc``
holding, on the simulation's gridded year axis:

- ``amoc_strength`` (Sv) — from ``data_loader.AMOC_FILE`` (NaN where not covered)
- ``tas_global_mean`` (K) — area-weighted global annual-mean temperature
- ``tas_interhemispheric_diff`` (K) — area-weighted NH-mean minus SH-mean
- ``precip_centroid_lat_20``, ``precip_centroid_lat_30`` (deg N) —
  precipitation-mass centroid latitude (ITCZ proxy) over 20°S–20°N and 30°S–30°N

Temperature and precipitation scalars are computed from the gridded annual-mean
input files via ``data_loader.load_annual_field``.

    python scripts/make_scalar_timeseries.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import xarray as xr

import data_loader as dl

TAS_METADATA = {
    "tas_global_mean": {
        "units": "K",
        "long_name": "Global area-weighted annual-mean near-surface air temperature",
    },
    "tas_interhemispheric_diff": {
        "units": "K",
        "long_name": (
            "Interhemispheric near-surface air temperature difference "
            "(NH minus SH, area-weighted)"
        ),
    },
}

# Precipitation-mass centroid of the zonal-mean precip, over two tropical bands.
# The centroid integrates over both branches of a double ITCZ, so it varies
# continuously (unlike the argmax, which jumps between the two branches).
ITCZ_BANDS = [20.0, 30.0]
PRECIP_VAR = "pr"  # total precipitation (CAM PRECT)
PRECIP_NOTE = (
    "ITCZ proxy = area- and precip-weighted mean latitude (precipitation-mass "
    "centroid) of the zonal-mean precip within the band. Precip source is "
    "total pr (CAM PRECT) for all runs."
)


def main():
    os.makedirs(dl.PROCESSED_DIR, exist_ok=True)

    for experiment in dl.EXPERIMENTS:
        out_name = dl.scalar_file(experiment)
        out_path = os.path.join(dl.PROCESSED_DIR, out_name)
        print(f"building {out_name} ...", flush=True)

        data_vars = {}

        # Temperature scalars from the gridded annual tas.
        tas = dl.load_annual_field(experiment, "tas")
        years = tas["year"].values
        gmean = dl.global_mean(tas).rename("tas_global_mean")
        idiff = dl.interhemispheric_difference(tas).rename("tas_interhemispheric_diff")
        for da in (gmean, idiff):
            da.attrs = {**TAS_METADATA[da.name], "source_file": tas.attrs["source_file"]}
            data_vars[da.name] = da

        # ITCZ centroid(s) from the gridded annual total precipitation
        # (shares the run's year axis, so it slots into the same Dataset).
        precip = dl.load_annual_field(experiment, PRECIP_VAR)
        for band in ITCZ_BANDS:
            name = f"precip_centroid_lat_{int(band)}"
            cen = dl.tropical_precip_centroid_lat(precip, band).rename(name)
            cen.attrs = {
                "units": "degrees_north",
                "long_name": (
                    f"Precipitation-mass centroid latitude (ITCZ proxy), "
                    f"{int(band)}S-{int(band)}N"
                ),
                "band_deg": band,
                "note": PRECIP_NOTE,
                "source_file": precip.attrs["source_file"],
                "source_variable": PRECIP_VAR,
            }
            data_vars[name] = cen

        amoc = dl.amoc_strength_on_years(experiment, years)
        amoc.attrs = {
            "units": "Sv",
            "long_name": "AMOC strength",
            "source_file": dl.AMOC_FILE,
            "source_variable": dl.EXPERIMENTS[experiment]["amoc_column"],
            "note": "26.5N annual-mean AMOC, 2051-2150; NaN outside that range.",
        }
        data_vars["amoc_strength"] = amoc

        ds = xr.Dataset(data_vars)
        ds.attrs.update(
            {"source_id": dl.SOURCE_ID, "experiment": experiment, "frequency": "annual"}
        )
        ds.to_netcdf(out_path)

        yr = ds["year"].values
        n_amoc = int(ds["amoc_strength"].notnull().sum())
        print(
            f"  -> {out_path}  (years {int(yr[0])}-{int(yr[-1])}, n={yr.size}; "
            f"amoc valid={n_amoc}; vars={list(ds.data_vars)})",
            flush=True,
        )

    print(f"\nDone. {len(dl.EXPERIMENTS)} files written to {dl.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
