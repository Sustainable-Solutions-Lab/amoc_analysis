"""Per-variable display metadata: units, scaling, colormaps, sanity ranges.

File metadata is not trustworthy, so this table is the authority on units. Every
entry was checked against the actual values in the control run's 2101-2150 mean:

- `RHREFHT` is labeled ``units = "fraction"`` but holds percent (global mean
  79.3, range 22.8-111.3). Values above 100 are real: supersaturation at
  reference height over polar ice.
- All `PREC*` rates are m/s, giving anomalies around 1e-9 unless converted.
- `QFLX` is kg/m2/s, numerically mm/s for water.
- `CLD*` really are fractions 0-1, so their labels are correct.

Colormaps follow one rule: sequential maps use a single hue, light to dark;
anomaly maps use a two-hue diverging map with a neutral midpoint so that zero
reads as "nothing". No rainbows.
"""

from __future__ import annotations

from dataclasses import dataclass

import xarray as xr

from .io import global_mean

M_PER_S_TO_MM_PER_DAY = 86400.0 * 1000.0
KG_PER_M2_S_TO_MM_PER_DAY = 86400.0


@dataclass(frozen=True)
class VariableInfo:
    """How to display one variable."""

    units: str                       # display units, after `scale`
    scale: float = 1.0               # multiply file values by this
    sequential_cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    valid_range: tuple[float, float] | None = None  # extremes must lie inside
    valid_mean: tuple[float, float] | None = None   # global mean must lie inside
    long_name: str | None = None     # overrides the file's long_name


_PRECIP = VariableInfo(
    units="mm day$^{-1}$",
    scale=M_PER_S_TO_MM_PER_DAY,
    sequential_cmap="Blues",
    diverging_cmap="BrBG",
    valid_range=(0.0, 100.0),
    valid_mean=(1e-3, 10.0),  # spans total precip (~2.9) down to snow components
)

_RADIATION_KW = dict(sequential_cmap="inferno", diverging_cmap="RdBu_r")
_CLOUD = VariableInfo(
    units="fraction",
    sequential_cmap="Blues",
    diverging_cmap="RdBu_r",
    valid_range=(0.0, 1.0),
    valid_mean=(0.05, 0.95),
)

VARIABLES: dict[str, VariableInfo] = {
    # Moisture --------------------------------------------------------------
    "RHREFHT": VariableInfo(
        units="%",
        sequential_cmap="Blues",
        diverging_cmap="BrBG",
        valid_range=(0.0, 130.0),
        valid_mean=(40.0, 100.0),
        long_name="Reference height relative humidity",
    ),
    "QREFHT": VariableInfo(
        units="g kg$^{-1}$",
        scale=1000.0,
        sequential_cmap="Blues",
        diverging_cmap="BrBG",
        valid_range=(0.0, 30.0),
        valid_mean=(3.0, 20.0),
    ),
    "TMQ": VariableInfo(
        units="kg m$^{-2}$",
        sequential_cmap="Blues",
        diverging_cmap="BrBG",
        valid_range=(0.0, 80.0),
        valid_mean=(10.0, 50.0),
    ),
    "QFLX": VariableInfo(
        units="mm day$^{-1}$",
        scale=KG_PER_M2_S_TO_MM_PER_DAY,
        sequential_cmap="Blues",
        diverging_cmap="BrBG",
        valid_range=(-10.0, 40.0),
        valid_mean=(0.5, 10.0),
    ),
    "PRECT": _PRECIP, "PRECC": _PRECIP, "PRECL": _PRECIP,
    "PRECSC": _PRECIP, "PRECSL": _PRECIP, "PRECSH": _PRECIP,
    # Temperature -----------------------------------------------------------
    "TREFHT": VariableInfo(units="K", sequential_cmap="magma", valid_range=(180.0, 340.0), valid_mean=(250.0, 320.0)),
    "TREFMNAV": VariableInfo(units="K", sequential_cmap="magma", valid_range=(180.0, 340.0), valid_mean=(250.0, 320.0)),
    "TREFMXAV": VariableInfo(units="K", sequential_cmap="magma", valid_range=(180.0, 340.0), valid_mean=(250.0, 320.0)),
    # Clouds ----------------------------------------------------------------
    "CLDTOT": _CLOUD, "CLDLOW": _CLOUD, "CLDMED": _CLOUD, "CLDHGH": _CLOUD,
    # Snow ------------------------------------------------------------------
    "SNOWHICE": VariableInfo(units="m", sequential_cmap="Purples", valid_range=(0.0, 20.0)),
    "SNOWHLND": VariableInfo(units="m", sequential_cmap="Purples", valid_range=(0.0, 20.0)),
}

# Radiative fluxes all share display conventions; ranges are wide because
# clear-sky, net, and cloud-forcing fields differ in sign and magnitude.
for _name in (
    "FLDS", "FLDSC", "FLNS", "FLNSC", "FLNT", "FLNTC", "FLUT", "FLUTC",
    "FSDS", "FSDSC", "FSNS", "FSNSC", "FSNT", "FSNTC",
    "LHFLX", "SHFLX", "LWCF", "SWCF", "SOLIN", "SRFRAD",
):
    VARIABLES[_name] = VariableInfo(units="W m$^{-2}$", valid_range=(-600.0, 800.0), **_RADIATION_KW)


def info(var: str, da: xr.DataArray) -> VariableInfo:
    """Display metadata for a variable, falling back to the file's own labels.

    The fallback is cosmetic only — it never rescales, so an unlisted variable
    plots in whatever units the file uses rather than in silently wrong ones.
    """
    if var in VARIABLES:
        return VARIABLES[var]
    return VariableInfo(units=da.attrs["units"], long_name=da.attrs.get("long_name"))


def to_display_units(da: xr.DataArray, var: str, check_range: bool = True) -> xr.DataArray:
    """Scale a field into display units.

    ``check_range`` asserts the result lies inside the variable's expected
    physical range. This is the one place defensive code earns its keep: a unit
    error produces a plausible-looking number rather than a traceback, so it is
    caught loudly here. Pass ``check_range=False`` for anomalies, whose values
    are differences and need not lie in the absolute range.
    """
    meta = info(var, da)
    out = da * meta.scale
    out.attrs = dict(da.attrs)
    out.attrs["units"] = meta.units
    if meta.long_name is not None:
        out.attrs["long_name"] = meta.long_name

    if check_range:
        _assert_physical(out, var, meta, da.attrs["units"])
    return out


def _assert_physical(da: xr.DataArray, var: str, meta: VariableInfo, file_units: str) -> None:
    """Catch unit errors, which produce plausible numbers rather than tracebacks.

    Both checks are needed. The range catches values landing somewhere absurd;
    the global mean catches a field rescaled into a range that is still nominally
    valid — dividing RH by 100 leaves every value inside 0-130 %, and only the
    mean reveals that the field is now a fraction.
    """
    context = (
        f"Check the unit conversion in variables.py (scale={meta.scale:g}) "
        f"against the file's units attribute {file_units!r}."
    )
    if meta.valid_range is not None:
        lo, hi = meta.valid_range
        actual = (float(da.min()), float(da.max()))
        assert lo <= actual[0] and actual[1] <= hi, (
            f"{var} outside expected range {meta.valid_range}: got {actual}. {context}"
        )
    if meta.valid_mean is not None:
        lo, hi = meta.valid_mean
        actual_mean = float(global_mean(da))
        assert lo <= actual_mean <= hi, (
            f"{var} global mean {actual_mean:.4g} outside expected "
            f"{meta.valid_mean}. {context}"
        )


def display_name(var: str, da: xr.DataArray) -> str:
    """Long name for titles, preferring our override over the file's."""
    meta = info(var, da)
    return meta.long_name or da.attrs["long_name"]
