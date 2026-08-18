"""Paths and the 3x3 experiment registry for the CESM1 AMOC/CO2 factorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "input" / "Annual_Mean_2D_Fileds_ATMs"

# Results live under data/ alongside the inputs, and so are covered by the same
# .gitignore rule — generated output never reaches the repository.
OUTPUT_DIR = REPO_ROOT / "data" / "output"
BOOK_DIR = OUTPUT_DIR / "books"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Two naming conventions are in play. The cases delivered first are
# `<VAR>_ann_mean.nc`; the +0.3 Sv runs that arrived later come straight from
# CDO as `<VAR>_annual.nc`. Same grid, same annual means, same Dec-Nov year
# convention — only the file name differs, so a case reports its own.
FILE_SUFFIXES = ("_ann_mean.nc", "_annual.nc")

CO2_LEVELS = (1, 2, 4)          # multiples of pre-industrial CO2
HOSING_LEVELS = (-0.3, 0.0, 0.3)  # Sv of North Atlantic freshwater hosing

# Analysis window shared by every case, in model years. 2051 is dropped because
# 4xCO2_noh archives it as an 11-month mean; the end is set by the shortest
# canonical run, 4xCO2_hosing_FIX, which stops at 2135. picontrol is restricted
# to this same window rather than its full 1850-2150 record: the control is not
# stationary, and slow ocean drift would otherwise alias into the differences.
ANALYSIS_YEARS = slice(2052, 2135)

# Quasi-steady-state window: 50 years, placed at the latest point every
# canonical case reaches. It is not 2101-2150 any more — the +0.3 Sv runs end at
# 2150, 2140 and 2135, so that window would have quietly averaged 50 years for
# some panels and 35 for others. `analysis.steady_state` asserts the full window
# is present rather than trusting this constant to stay in step with the data.
STEADY_STATE_YEARS = slice(2086, 2135)


@dataclass(frozen=True)
class Case:
    """One member of the 3x3 (CO2 x hosing) design."""

    name: str        # directory name under DATA_ROOT
    co2: int         # 1, 2 or 4 (xCO2)
    hosing: float    # Sv

    @property
    def path(self) -> Path:
        return DATA_ROOT / self.name

    @property
    def exists(self) -> bool:
        return self.path.is_dir()

    @property
    def suffix(self) -> str:
        """Which of FILE_SUFFIXES this case's directory uses."""
        found = [s for s in FILE_SUFFIXES if any(self.path.glob(f"*{s}"))]
        assert len(found) == 1, (
            f"{self.name}: expected exactly one file-naming convention, found {found}"
        )
        return found[0]

    @property
    def label(self) -> str:
        return f"{self.co2}xCO2 / {self.hosing:+.1f} Sv"


# The 3x3 design, by directory name. The +0.3 Sv column uses the `_hosing_FIX`
# runs; see ALTERNATE_CASES for the `_hosing_oldcrash` twins they replace.
CASES: tuple[Case, ...] = (
    Case("1xCO2_neghos",     1, -0.3),
    Case("picontrol",        1,  0.0),
    Case("1xCO2_hosing_FIX", 1, +0.3),
    Case("2xCO2_neghos",     2, -0.3),
    Case("2xCO2_noh",        2,  0.0),
    Case("2xCO2_hosing_FIX", 2, +0.3),
    Case("4xCO2_neghos",     4, -0.3),
    Case("4xCO2_noh",        4,  0.0),
    Case("4xCO2_hosing_FIX", 4, +0.3),
)

# The superseded +0.3 Sv runs. These crashed partway (hence the name) and were
# rerun as the FIX cases, so they are shorter — 50, 45 and 25 years against
# 100, 90 and 85. They are a separate registry rather than part of CASES: they
# are the same three cells of the design, so they cannot share a grid with the
# FIX runs, but they are worth a book of their own.
ALTERNATE_CASES: tuple[Case, ...] = (
    Case("1xCO2_hosing_oldcrash", 1, +0.3),
    Case("2xCO2_hosing_oldcrash", 2, +0.3),
    Case("4xCO2_hosing_oldcrash", 4, +0.3),
)

CONTROL = CASES[1]  # picontrol: 1xCO2, no hosing


def get_case(co2: int, hosing: float) -> Case:
    for case in CASES:
        if case.co2 == co2 and abs(case.hosing - hosing) < 1e-9:
            return case
    raise KeyError(f"no case with co2={co2}, hosing={hosing}")


def get_case_by_name(name: str) -> Case:
    for case in CASES + ALTERNATE_CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown case name: {name!r}")


def available_cases() -> list[Case]:
    """Cases whose directory is present on disk."""
    return [c for c in CASES if c.exists]
