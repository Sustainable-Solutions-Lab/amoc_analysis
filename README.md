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

### Analysis windows

`ANALYSIS_YEARS = 2052-2150` is the window common to every case, and is the
default for all loading — including `picontrol`, which is trimmed to these same
years rather than averaged over its full record. Same-year weather is
uncorrelated across branches, so this cancels no noise; the reason to match years
is slow transient drift in the ocean, which is shared with the control over the
same span and therefore differences out. Pass `years=None` to recover the full
1851–2150 control, e.g. for internal-variability statistics.

`STEADY_STATE_YEARS = 2101-2150` — the last 50 years of every simulation — is the
quasi-steady-state window for the factorial comparisons.

Two differencing modes, in `amoc_cesm.analysis`:

```python
transient_anomaly("4xCO2_noh", "TREFMXAV")       # year for year vs. picontrol
steady_state_anomaly("4xCO2_noh", "TREFMXAV")    # last-50-year means, differenced
steady_state("4xCO2_noh", "TREFMXAV")            # last-50-year mean, no reference
```

Both take `reference=` to compare against any case, not just the control.

**`4xCO2_noh` is a placeholder** to be replaced when the final run is available.

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
  config.py        paths, Case registry for the 3x3 design, analysis windows
  io.py            loading, area-weighted means, climatologies
  analysis.py      transient and steady-state anomalies
  significance.py  Welch t-test vs. the control, optional FDR control
  variables.py     display units, scaling, colormaps, unit-error assertions
  plotting.py      the 3x3 grid page: Robinson maps + zonal-mean sidebars
  books.py         multi-page PDF assembly
  workflows/
    steady_state.py  quasi-steady-state workflow (pages now, tables later)
    transient.py     time-dependent workflow (to be built)
scripts/           runnable entry points
  inventory.py               what cases/variables/years are on disk
  make_steady_state_book.py  build the quasi-steady-state PDF book
data/input/        input NetCDF (not committed)
data/output/       generated books, figures, tables (not committed)
```

## Two analysis workflows

Everything below the reduction step is shared: `plotting.grid_3x3` takes a dict
of `{(co2, hosing): DataArray}` and knows nothing about which analysis produced
it, so both workflows draw the same 3×3 page — rows are CO₂ levels, columns are
hosing levels, and cases that have not run render as labeled placeholders.

- **Quasi-steady-state** (`workflows/steady_state.py`) — the 2101–2150 mean of
  each run treated as an equilibrium climate. Built.
- **Time-dependent** (`workflows/transient.py`) — year-for-year against the
  control. To be built; it adds only its own reductions.

### Making a book

```bash
python scripts/make_steady_state_book.py RHREFHT PRECT CLDTOT
python scripts/make_steady_state_book.py TREFMXAV --png    # also write page PNGs
python scripts/make_steady_state_book.py --all             # every common variable
python scripts/make_steady_state_book.py --all --name draft   # fixed name instead
```

Books land in `data/output/books/` as
`steady_state_book_<yyyy-mm-dd-hh-mm-ss>.pdf`. The timestamp means successive
runs accumulate rather than overwrite: a book records what the data looked like
when it was built, and cases are still arriving. `--name` overrides the stem when
you want a stable filename.

**Four pages per variable**, each on its own color scale:

| page | what it shows | reference for each panel |
|---|---|---|
| absolute | climatology, shared sequential scale | — |
| anomaly | total response to both perturbations | `picontrol` |
| **warming effect** | CO₂ response with the AMOC state held | 1×CO₂ in the *same column* |
| **AMOC effect** | hosing response with forcing held | no-hosing in the *same row* |

The last two are what separate the effects. On the warming page, reading down a
column shows the CO₂ response growing; reading *across* a row shows whether that
response depends on the AMOC state — which is the interaction. The AMOC page is
the same idea transposed. Reference panels (the 1×CO₂ row, the 0 Sv column) are
zero by construction and stay in place labeled as such, so every page in the
book keeps the same 3×3 skeleton and panels never move.

The two effects differ by roughly a factor of ten for some fields, so each page
computes its own symmetric scale rather than sharing one that would flatten the
smaller effect.

**Page order is always alphabetical by variable**, whatever order they were
requested in, with each variable's absolute and anomaly pages kept adjacent. The
pairing is never split, so a book stays navigable as fields are added — the
sorting lives in `workflows.steady_state.book_pages`, so future workflows that
reuse it inherit the same rule.

**Contours** are drawn on every page at the colorbar's own labeled values
(`plotting.tick_levels` feeds both, so a line always sits on a labeled value).
On absolute pages they cover the whole panel and no statistics are involved — a
contour there is just an isoline, and the zero level is kept because a field like
`SHFLX` genuinely crosses zero. On the three difference pages the zero level is
dropped (it would trace a sign change, not a magnitude) and the lines are clipped
to significant regions.

**Significance** is a Welch t-test on the 50 annual values at each grid point,
Benjamini-Hochberg controlled by default (testing ~14k points at α = 0.05 would
otherwise yield ~700 false positives). It is shown as **black contours over a
color field that covers the whole map** — nothing is masked or stippled away:

- `--significance-style field` (default) contours the anomaly at **the
  colorbar's own labeled values**, drawn only where the difference is
  significant. Contour levels and colorbar ticks come from one shared array
  (`plotting.tick_levels`), so a line always sits exactly on a labeled value.
  Zero is excluded — a zero contour traces a sign change rather than a
  magnitude. Negative contours are dashed, so sign reads without the color.
- `--significance-style outline` instead traces the boundary of significant
  regions, saying where the signal is trustworthy but nothing about its size.

Anomaly color limits are the 98th percentile of |anomaly|, not the maximum, so a
few extreme polar cells don't wash out the pattern; the colorbar carries extend
arrows showing values run past both ends.

### Units are asserted, not assumed

`variables.py` is the authority on units because file metadata is not reliable —
`RHREFHT` is labeled `fraction` but holds percent, and all `PREC*` rates are m/s.
Each variable carries an expected range *and* an expected global mean, both
asserted after scaling. Two checks are needed: dividing RH by 100 leaves every
value inside a valid 0–130 % range, and only the mean reveals the error. All 39
variables pass their own checks on the real data.

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
