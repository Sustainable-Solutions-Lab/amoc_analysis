"""Build per-simulation scalar (one-value-per-year) annual time-series files.

For each simulation, writes ``data/processed/scalars_annual_CESM2_{exp}.nc``
holding, on the simulation's gridded year axis:

- ``amoc_strength`` (Sv) — from ``CESM2_AMOC_experiments.nc``
- ``tas_global_mean`` (K) — area-weighted global annual-mean temperature
- ``tas_interhemispheric_diff`` (K) — area-weighted NH-mean minus SH-mean

Temperature scalars are derived from the precomputed annual ``tas`` files (area
weighting commutes with the month-length-weighted annual mean), so this does not
re-read the monthly data. Run ``scripts/make_annual_means.py`` first.

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


def main():
    os.makedirs(dl.PROCESSED_DIR, exist_ok=True)

    for sim in dl.SCALAR_SIMULATIONS:
        experiment = sim["experiment"]
        out_name = f"scalars_annual_CESM2_{experiment}.nc"
        out_path = os.path.join(dl.PROCESSED_DIR, out_name)
        print(f"building {out_name} ...", flush=True)

        data_vars = {}

        # Temperature scalars from the annual tas file (absent for greenland-hosing).
        if sim["tas_file"] is not None:
            tas = xr.open_dataset(os.path.join(dl.PROCESSED_DIR, sim["tas_file"]))["tas"]
            years = tas["year"].values
            gmean = dl.global_mean(tas).rename("tas_global_mean")
            idiff = dl.interhemispheric_difference(tas).rename(
                "tas_interhemispheric_diff"
            )
            for da in (gmean, idiff):
                da.attrs = {**TAS_METADATA[da.name], "source_file": sim["tas_file"]}
                data_vars[da.name] = da
        else:
            # No gridded tas: use the AMOC series' own length as the year index.
            n = xr.open_dataset(os.path.join(dl.INPUT_DIR, dl.AMOC_FILE))[
                sim["amoc_segments"][0]["variable"]
            ].sizes["time"]
            years = range(1, n + 1)

        amoc = dl.amoc_strength_on_years(sim["amoc_segments"], list(years))
        amoc.attrs = {
            "units": "Sv",
            "long_name": "AMOC strength",
            "source_file": dl.AMOC_FILE,
            "note": sim["amoc_note"],
        }
        data_vars["amoc_strength"] = amoc

        ds = xr.Dataset(data_vars)
        ds.attrs.update(
            {"source_id": "CESM2", "experiment": experiment, "frequency": "annual"}
        )
        ds.to_netcdf(out_path)

        yr = ds["year"].values
        n_amoc = int(ds["amoc_strength"].notnull().sum())
        print(
            f"  -> {out_path}  (years {int(yr[0])}-{int(yr[-1])}, n={yr.size}; "
            f"amoc valid={n_amoc}; vars={list(ds.data_vars)})",
            flush=True,
        )

    print(f"\nDone. {len(dl.SCALAR_SIMULATIONS)} files written to {dl.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
