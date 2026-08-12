# amoc-cesm

Separating the climate effects of **global warming** from the effects of **AMOC
slowdown/shutdown**, using annual-mean 2D atmospheric fields from CESM1.

## Experimental design

A 3×3 factorial: CO₂ concentration × North Atlantic freshwater hosing.

|            | −0.3 Sv        | 0 Sv         | +0.3 Sv          |
|------------|----------------|--------------|------------------|
| **1×CO₂**  | `1xCO2_neghos` | `picontrol`  | *(pending)*      |
| **2×CO₂**  | `2xCO2_neghos` | `2xCO2_noh`  | *(pending)*      |
| **4×CO₂**  | `4xCO2_neghos` | `4xCO2_noh`  | *(pending)*      |

Negative hosing (freshwater extraction from the North Atlantic) strengthens the
AMOC; positive hosing weakens or shuts it down. Comparing along the CO₂ axis
isolates the warming response; comparing along the hosing axis isolates the AMOC
response; the interaction terms say how much the two effects are separable.

**Status:** the three +0.3 Sv cases have not yet been delivered, and directory
names for them in `src/amoc_cesm/config.py` are placeholders to be corrected
when those runs land. Run `python scripts/inventory.py` at any time for the
current state of the data on disk.

## Data

Not in this repository (~1.6 GB and growing). Expected location:

```
data/input/Annual_Mean_2D_Fileds_ATMs/<case>/<VAR>_ann_mean.nc
```

Each file holds one variable on a regular 144×96 lon/lat grid (CAM finite
volume, 2.5°×1.9°), as `(time, lat, lon)` annual means, `365_day` calendar.

- Perturbation cases: 100 years (model years 2051–2150).
- `picontrol`: 301 years (1850–2150), usable as a control and for internal
  variability / significance testing.

### Time axes are already aligned

All cases share the same 365-day calendar and the same annual time stamps, and
the perturbation years 2051–2150 are exactly the last 100 years of `picontrol`
(raw values identical to the day). Loading therefore replaces `time` with an
integer `year` coordinate by default, so cases difference directly:

```python
load_var("4xCO2_noh", "FLUT") - load_var("picontrol", "FLUT")   # aligns on year
```

Two records are **11-month means with January missing** (`time_bnds` span 334
days rather than 365): `picontrol` 1850 and `4xCO2_noh` 2051. `load_var` drops
them by default (`drop_partial=True`), which leaves 2052–2150 as the span common
to every case. The `4xCO2_noh` gap is in all 38 of its variables, so it is a
property of the run's archive, not of one file.

Year alignment is bookkeeping, not pairing: the perturbation runs are branches of
`picontrol`, so their weather is uncorrelated with the control in the same
calendar year. Difference time means rather than individual years — pairing buys
no noise cancellation.

39 variables are available (radiation, clouds, precipitation, surface fluxes,
near-surface temperature and humidity). `TREFHT` is currently present only in
`picontrol` — the perturbation cases carry `TREFMNAV`/`TREFMXAV` but not the
mean; worth requesting from the run archive.

## Setup

```bash
python -m venv .venv          # already created
.venv/bin/pip install -r requirements.txt
```

## Layout

```
src/amoc_cesm/     importable package
  config.py        paths, Case registry for the 3x3 design
  io.py            loading, area-weighted means, climatologies
scripts/           runnable analysis scripts
  inventory.py     what cases/variables/years are on disk
figures/           generated figures (not committed)
output/            generated tables and derived data (not committed)
data/              input NetCDF (not committed)
```

## Usage

```python
import sys; sys.path.insert(0, "src")
from amoc_cesm import get_case, load_var, global_mean, climatology
from amoc_cesm.io import load_ensemble

# one case, last 50 years
case = get_case(co2=4, hosing=-0.3)
print(float(global_mean(climatology(load_var(case, "LHFLX"), years=slice(2101, 2150)))))

# every available case stacked on a `case` dimension, with co2/hosing coords
ens = load_ensemble("FLUT", years=slice(2052, 2150))
print(global_mean(ens).mean("year"))
```
