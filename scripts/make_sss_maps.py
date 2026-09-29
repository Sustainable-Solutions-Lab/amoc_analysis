"""End-of-run sea-surface-salinity maps: 3x3 absolute, 3x3 anomaly, FIX vs yr200.

Requires the regridded files from `scripts/regrid_salt.py`.

    python scripts/make_sss_maps.py [n_years]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from amoc_cesm.books import write_book  # noqa: E402
from amoc_cesm.config import BOOK_DIR, FIGURE_DIR  # noqa: E402
from amoc_cesm.workflows.sss_maps import N_YEARS, pages  # noqa: E402


def main(n_years: int = N_YEARS) -> None:
    path = BOOK_DIR / f"sss_end_of_run_last{n_years}yr.pdf"
    n = write_book(pages(n_years), path, png_dir=FIGURE_DIR / "sss_end_of_run")
    print(f"{n} pages -> {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_YEARS)
