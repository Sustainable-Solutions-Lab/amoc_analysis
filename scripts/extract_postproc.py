"""Extract the NAHosMIP_v2 ocean fields: SSS on the CAM grid, MOC and transports.

    python scripts/extract_postproc.py [case ...]

Writes per case and frequency:
    data/output/postproc_v2/sss/<case>_SSS_<freq>_144x96.nc
    data/output/postproc_v2/transports/<case>_transports_<freq>.nc
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amoc_cesm.postproc import (  # noqa: E402
    FREQUENCIES, POSTPROC_CASES, regrid_sss, sss_path, transports, transports_path,
)
from amoc_cesm.regrid import pop_grid  # noqa: E402

ENCODING = {"zlib": True, "complevel": 4}


def _write(ds, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: ENCODING for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    print(f"  {path.name:52s} {path.stat().st_size / 1e6:6.2f} MB")


def main(cases: list[str]) -> None:
    grid = pop_grid()  # gx1v6 is the same in every run; read it once
    for case in cases:
        print(case)
        for freq in FREQUENCIES:
            _write(regrid_sss(case, freq, grid=grid), sss_path(case, freq))
            _write(transports(case, freq), transports_path(case, freq))


if __name__ == "__main__":
    main(sys.argv[1:] or list(POSTPROC_CASES))
