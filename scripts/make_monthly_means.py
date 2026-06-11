"""Precompute the CESM2 gridded monthly fields reshaped to ``(year, month, lat,
lon)`` and write them to ``data/processed/``.

The monthly analog of ``scripts/make_annual_means.py``: instead of collapsing each
year to a month-length-weighted mean, it keeps every calendar month so the
per-calendar-month regressions (``scripts/run_monthly_regressions.py``) can read
``{var}_monthly_CESM2_{experiment}.nc`` directly. NetCDF output is zlib-compressed
because these files are ~12x the size of the annual ones.

    python scripts/make_monthly_means.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl

# Same combined-simulation provenance note as the annual pipeline (see README): the
# historical+ssp585 series splices two ensemble members.
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
        out_name = f"{var}_monthly_CESM2_{experiment}.nc"
        out_path = os.path.join(dl.PROCESSED_DIR, out_name)

        print(f"building {out_name} ...", flush=True)
        monthly = dl.load_and_normalize_monthly(entry)

        ds = monthly.to_dataset(name=var)
        ds.attrs.update(
            {"source_id": "CESM2", "experiment": experiment, "frequency": "monthly"}
        )
        ds.attrs.update(COMBINED_ATTRS.get(experiment, {}))

        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"}}
        ds.to_netcdf(out_path, encoding=encoding)
        years = ds["year"].values
        print(
            f"  -> {out_path}  (years {int(years[0])}-{int(years[-1])}, "
            f"n_year={years.size}, n_month={ds.sizes['month']})",
            flush=True,
        )

    print(f"\nDone. {len(dl.INPUT_MANIFEST)} files written to {dl.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
