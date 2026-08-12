#!/usr/bin/env python
"""Report which cases, variables, and years are currently on disk.

Run this after new simulation output arrives:

    python scripts/inventory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import xarray as xr  # noqa: E402

from amoc_cesm.config import CASES, CO2_LEVELS, HOSING_LEVELS, get_case  # noqa: E402
from amoc_cesm.io import FULL_YEAR_DAYS, var_path, variables_in  # noqa: E402


def time_span(case, var: str) -> str:
    try:
        with xr.open_dataset(var_path(case, var), decode_times=True) as ds:
            years = ds["time"].dt.year.values
        with xr.open_dataset(var_path(case, var), decode_times=False) as ds:
            spans = ds["time_bnds"].values[:, 1] - ds["time_bnds"].values[:, 0]
        partial = years[spans < FULL_YEAR_DAYS]
        note = f"  partial: {', '.join(str(y) for y in partial)}" if len(partial) else ""
        return f"{len(years)} yr ({years[0]}-{years[-1]}){note}"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unreadable ({exc})"


def main() -> None:
    print("=== 3x3 design: cases present on disk ===")
    header = "        " + "".join(f"{h:>+16.1f} Sv" for h in HOSING_LEVELS)
    print(header)
    for co2 in CO2_LEVELS:
        row = f"{co2}xCO2 "
        for hos in HOSING_LEVELS:
            case = get_case(co2, hos)
            mark = case.name if case.exists else f"[{case.name}]"
            row += f"{mark:>19}"
        print(row)
    print("  [name] = not yet present\n")

    all_vars: set[str] = set()
    present = [c for c in CASES if c.exists]
    per_case = {c.name: set(variables_in(c)) for c in present}
    for vs in per_case.values():
        all_vars |= vs

    print("=== per-case summary ===")
    for case in present:
        vs = per_case[case.name]
        ref = "FLUT" if "FLUT" in vs else sorted(vs)[0]
        print(f"{case.name:<16} {case.label:<18} {len(vs):3d} vars   {time_span(case, ref)}")

    missing = {
        c.name: sorted(all_vars - per_case[c.name]) for c in present
    }
    missing = {k: v for k, v in missing.items() if v}
    if missing:
        print("\n=== variables missing from some cases ===")
        for name, vs in missing.items():
            print(f"{name:<16} missing: {', '.join(vs)}")

    print(f"\n{len(all_vars)} distinct variables:")
    print("  " + " ".join(sorted(all_vars)))


if __name__ == "__main__":
    main()
