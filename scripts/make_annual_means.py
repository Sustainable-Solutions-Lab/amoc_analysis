"""Normalize the CESM1 annual-mean gridded fields and write them to
``data/processed/``.

Thin wrapper around :mod:`src.data_loader`. The input files in ``data/input/``
are already annual means; this step selects the analysis variables (``tas``,
``prc``, ``pr``), converts them to CMIP names and units, and puts each run on an
integer ``year`` axis, one file per variable and simulation.

    python scripts/make_annual_means.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader as dl


def main():
    os.makedirs(dl.PROCESSED_DIR, exist_ok=True)

    for entry in dl.INPUT_MANIFEST:
        var, experiment = entry["var"], entry["experiment"]
        out_name = dl.annual_file(var, experiment)
        out_path = os.path.join(dl.PROCESSED_DIR, out_name)

        print(f"building {out_name} ...", flush=True)
        annual = dl.load_and_normalize(entry)

        ds = annual.to_dataset(name=var)
        forcing = dl.EXPERIMENTS[experiment]
        ds.attrs.update(
            {
                "source_id": dl.SOURCE_ID,
                "experiment": experiment,
                "frequency": "annual",
                "co2_multiple": forcing["co2_multiple"],
                "hosing_sv": forcing["hosing_sv"],
            }
        )

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
