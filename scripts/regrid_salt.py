"""Regrid the POP sea-surface-salinity extracts onto the CAM 144x96 grid.

Writes, for each case, a monthly file and a length-of-month-weighted annual-mean
file under data/output/regrid/SSS/.

    python scripts/regrid_salt.py [case ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amoc_cesm.regrid import (  # noqa: E402
    REGRID_DIR, SALT_DIR, SALT_SOURCES, annual_mean, regrid_path, regrid_pop_file,
)

ENCODING = {"zlib": True, "complevel": 4, "dtype": "float32"}


def main(cases: list[str]) -> None:
    REGRID_DIR.mkdir(parents=True, exist_ok=True)
    for case in cases:
        src = SALT_DIR / SALT_SOURCES[case]
        t0 = time.time()
        monthly = regrid_pop_file(src)
        yearly = annual_mean(monthly)

        for freq, ds in (("mon", monthly), ("ann", yearly)):
            out = regrid_path(case, freq)
            ds.to_netcdf(out, encoding={v: ENCODING for v in ds.data_vars})
            print(f"{case:22s} {freq}  {ds.sizes.get('time', ds.sizes.get('year')):5d}  "
                  f"{out.stat().st_size / 1e6:6.1f} MB")
        print(f"{case:22s} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1:] or list(SALT_SOURCES))
