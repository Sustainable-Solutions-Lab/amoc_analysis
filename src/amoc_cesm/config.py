"""Paths and the 3x3 experiment registry for the CESM1 AMOC/CO2 factorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "input" / "Annual_Mean_2D_Fileds_ATMs"
FIGURE_DIR = REPO_ROOT / "figures"
OUTPUT_DIR = REPO_ROOT / "output"

FILE_SUFFIX = "_ann_mean.nc"

CO2_LEVELS = (1, 2, 4)          # multiples of pre-industrial CO2
HOSING_LEVELS = (-0.3, 0.0, 0.3)  # Sv of North Atlantic freshwater hosing


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
    def label(self) -> str:
        return f"{self.co2}xCO2 / {self.hosing:+.1f} Sv"


# Directory names for cases that have run. Cases not yet on disk are still
# listed so that the experimental design stays explicit; use `available_cases()`
# to get only the ones present. Names for the not-yet-delivered +0.3 Sv cases
# are a guess and should be corrected when those runs land.
CASES: tuple[Case, ...] = (
    Case("1xCO2_neghos", 1, -0.3),
    Case("picontrol",    1,  0.0),
    Case("1xCO2_poshos", 1, +0.3),
    Case("2xCO2_neghos", 2, -0.3),
    Case("2xCO2_noh",    2,  0.0),
    Case("2xCO2_poshos", 2, +0.3),
    Case("4xCO2_neghos", 4, -0.3),
    Case("4xCO2_noh",    4,  0.0),
    Case("4xCO2_poshos", 4, +0.3),
)

CONTROL = CASES[1]  # picontrol: 1xCO2, no hosing


def get_case(co2: int, hosing: float) -> Case:
    for case in CASES:
        if case.co2 == co2 and abs(case.hosing - hosing) < 1e-9:
            return case
    raise KeyError(f"no case with co2={co2}, hosing={hosing}")


def get_case_by_name(name: str) -> Case:
    for case in CASES:
        if case.name == name:
            return case
    raise KeyError(f"unknown case name: {name!r}")


def available_cases() -> list[Case]:
    """Cases whose directory is present on disk."""
    return [c for c in CASES if c.exists]
