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

case = get_case(co2=4, hosing=-0.3)
lhflx = load_var(case, "LHFLX")
print(float(global_mean(climatology(lhflx, last_n_years=50))))
```
