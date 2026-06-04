"""Precompute month-length-weighted annual means of the CESM2 gridded monthly
fields and write them to ``data/processed/``.

Thin wrapper around :mod:`src.data_loader`. Run once after placing the monthly
NetCDF files in ``data/input/``; downstream analyses then read the small annual
files instead of reprocessing ~7 GB of monthly data each time.

    python scripts/make_annual_means.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl

# Dataset-level provenance for the combined historical+ssp585 simulation, which
# splices two ensemble members (see README): the historical filename advertises
# r1i1p1f1, while the ssp585 file's variant_label attribute is r4i1p1f1.
COMBINED_ATTRS = {
    "historical-ssp585": {
        "note": (
            "Single continuous simulation formed by splicing historical "
            "(1850-2014) and ssp585 (2015-2100). Ensemble members differ: "
            "historical variant_label r1i1p1f1, ssp585 variant_label r4i1p1f1."
        )
    }
}


def main():
    os.makedirs(dl.PROCESSED_DIR, exist_ok=True)

    for entry in dl.INPUT_MANIFEST:
        var, experiment = entry["var"], entry["experiment"]
        out_name = f"{var}_annual_CESM2_{experiment}.nc"
        out_path = os.path.join(dl.PROCESSED_DIR, out_name)

        print(f"building {out_name} ...", flush=True)
        annual = dl.load_and_normalize(entry)

        ds = annual.to_dataset(name=var)
        ds.attrs.update(
            {"source_id": "CESM2", "experiment": experiment, "frequency": "annual"}
        )
        ds.attrs.update(COMBINED_ATTRS.get(experiment, {}))

        ds.to_netcdf(out_path)
        years = ds["year"].values
        print(
            f"  -> {out_path}  (years {int(years[0])}-{int(years[-1])}, "
            f"n={years.size})",
            flush=True,
        )

    # AMOC strength is handled by scripts/make_scalar_timeseries.py, which writes
    # the per-simulation scalar files (amoc_strength + temperature scalars).
    print(f"\nDone. {len(dl.INPUT_MANIFEST)} files written to {dl.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
