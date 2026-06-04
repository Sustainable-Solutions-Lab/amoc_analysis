# AMOC Analysis

Statistical analysis (primarily linear regression) of climate model output from
the **CESM2** model, with a focus on the Atlantic Meridional Overturning
Circulation (AMOC) and its relationships to other climate variables.

## Goals

- Extract AMOC-relevant diagnostics from CESM2 output (e.g. AMOC streamfunction
  strength, North Atlantic surface temperature/salinity, surface fluxes).
- Use linear regression to characterize relationships between AMOC strength and
  other climate variables and forcings.
- Quantify trends, sensitivities, and their statistical uncertainty.
- Produce publication-quality figures summarizing the results.

## Setup

Developed with **Python 3.10**. Dependency versions in `requirements.txt` are
unpinned; the analysis is deterministic (no random components).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies:

- **numpy**, **pandas** — numerical arrays and tabular data
- **xarray**, **netCDF4**, **dask** — reading CESM2 NetCDF output; `dask`
  streams the ~2.5 GB monthly files so annual means are computed out-of-core
- **scipy**, **statsmodels** — linear regression and statistics
  (closed-form vectorized OLS for the per-grid-cell maps; `statsmodels`
  validates it and is available for single fits)
- **matplotlib**, **cartopy** — figures and coastline maps (Cartopy downloads
  Natural Earth coastline data on first use, which needs network access)

## Repository layout

```
amoc_analysis/
├── data/
│   ├── input/      # CESM2 model output (read-only — do not modify)
│   └── output/     # generated results, tables, figures (git-ignored)
├── src/            # analysis modules (data loading, regression, plotting)
├── scripts/        # thin command-line wrappers around src modules
├── requirements.txt
├── README.md
└── CLAUDE.md       # coding conventions for this project
```

## Data

CESM2 output is placed in `./data/input/` (NetCDF). This directory is treated as
read-only reference data — see `CLAUDE.md`. The gridded fields are too large to
commit to git and are not tracked in this repository, so **a colleague must
obtain these input files separately** and place them in `data/input/` with these
exact names (the filenames are hard-coded in `src/data_loader.py`):

```
CESM2_AMOC_experiments.nc
tas_Amon_CESM2_historical_r1i1p1f1_gn_18500115-20141215.nc
tas_Amon_CESM2_ssp585_r1i1p1f1_gn_20150115-21001215.nc
tas_Amon_CESM2_abrupt-4xCO2-002.nc
tas_Amon_CESM2_piControl_070001-079912.nc
tas_Amon_CESM2_u03-hos_1850001-202112.nc
pr_Amon_CESM2_historical_r1i1p1f1_gn_18500115-20141215.nc
pr_Amon_CESM2_ssp585_r1i1p1f1_gn_20150115-21001215.nc
pr_Amon_CESM2_abrupt-4xCO2-001.nc
pr_Amon_CESM2_piControl_070001-079912.nc
pr_Amon_CESM2_u03-hos_1850001-202112.nc
```

> **TODO (data source):** record where these files come from (archive/DOI/URL or
> internal path) so the inputs themselves are reproducible — this is the one
> prerequisite not contained in the repository.

### AMOC strength time series

`CESM2_AMOC_experiments.nc` — precomputed AMOC strength (units **Sv**) as annual
time series of length 100 (`time` = year index 1–100). Each experiment is a
separate variable:

| Variable | Description |
| --- | --- |
| `piControl` | preindustrial control |
| `hosing_0.3Sv_uniform` | 0.3 Sv freshwater hosing, uniform North Atlantic |
| `hosing_0.1Sv_greenland` | 0.1 Sv freshwater hosing, Greenland |
| `abrupt_4xCO2` | abrupt quadrupling of CO₂ |
| `historical_early_1850-1949` | historical, early window |
| `historical+ssp585_late_2001-2100` | historical+SSP5-8.5, late window |

### Gridded monthly fields

Near-surface air temperature and precipitation on the CESM2 native grid
(`lat` = 192 × `lon` = 288, nominal 1°), monthly (`Amon`). Two variables across
five experiments:

| Experiment (file stem) | Time range | n (months) | `tas`/`pr` provenance |
| --- | --- | --- | --- |
| `historical_r1i1p1f1_gn` | 1850-01 → 2014-12 | 1980 | CMORized |
| `ssp585_r1i1p1f1_gn`¹ | 2015-01 → 2100-12 | 1032 | CMORized |
| `abrupt-4xCO2`² | 0001-01 → 0999-12 | 11988 | CMORized |
| `piControl_070001-079912` | 0700 → 0800 | 1200 | raw CESM2 (CAM history) |
| `u03-hos_1850001-202112`³ | 1850-02 → 2022-01 | 2064 | raw CESM2 (CAM history) |

**Important provenance caveats:**

- **Two distinct data conventions.** The `historical`, `ssp585`, and
  `abrupt-4xCO2` files are CMORized: temperature is `tas` (Near-Surface Air
  Temperature, **K**) and precipitation is `pr` (total Precipitation,
  **kg m⁻² s⁻¹**). The `piControl` and `u03-hos` files are raw CESM2 CAM history
  output: temperature is `TREFHT` (Reference height temperature, **K**) and
  precipitation is `PRECC`-style **Convective** precipitation rate (liq + ice),
  in **m s⁻¹** — *not* total precipitation. These raw files also carry the full
  CAM metadata set (hybrid-sigma coefficients `hyam`/`hybm`, `co2vmr`,
  `sol_tsi`, etc.). Loading code must map variable names and reconcile
  units/quantities before comparing across experiments.
- ¹ The `ssp585` filenames say `r1i1p1f1`, but the file attribute
  `variant_label` is **`r4i1p1f1`** — the actual ensemble member is r4, not r1.
- ² `abrupt-4xCO2` is supplied as a single 999-year monthly run (~2.5 GB per
  variable); the `pr` and `tas` files carry different file suffixes
  (`-001` vs `-002`).
- ³ `u03-hos` is the freshwater-hosing experiment; its time axis spans
  1850–2022 on a `noleap` calendar.

### Processed annual means (`data/processed/`, git-ignored)

`scripts/make_annual_means.py` precomputes month-length-weighted annual means
(correct for the `noleap` calendar) of the gridded fields, following CMIP
variable names: `tas` (K), `pr` (**total** precipitation, kg m⁻² s⁻¹), and
`prc` (**convective** precipitation, kg m⁻² s⁻¹). The raw CAM files contribute
`TREFHT` (renamed `tas`) and convective precipitation as `prc` — these distinct
names keep convective output from being mistaken for total `pr`. Each variable
carries provenance attributes (`source_file`, `source_variable`,
`original_units`, `annual_mean_method`, `precip_kind`). Output (8 files):

| File | Coverage |
| --- | --- |
| `{tas,pr}_annual_CESM2_historical-ssp585.nc` | 1850–2100 (251 yr, spliced) |
| `{tas,pr}_annual_CESM2_abrupt-4xCO2.nc` | years 1–999 |
| `tas_annual_CESM2_piControl.nc`, `prc_annual_CESM2_piControl.nc` | years 700–799 |
| `tas_annual_CESM2_u03-hos.nc`, `prc_annual_CESM2_u03-hos.nc` | 1850–2021 |

**Two precipitation caveats:**

- **Total vs convective.** Only `historical`, `ssp585`, and `abrupt-4xCO2`
  provide total precipitation (`pr`). `piControl` and `u03-hos` provide
  convective precipitation only (`prc`) — a different quantity; do not regress
  `prc` against `pr` as if they were the same field.
- **Mislabeled source units.** The raw `prc` source carries `units = "m/s"`,
  but its values are a water mass flux already in kg m⁻² s⁻¹ (they match the
  total-`pr` magnitude; true m s⁻¹ precipitation would be ~1000× smaller). The
  data is used as-is without scaling; outputs record this in a `units_note`
  attribute.

The combined `historical-ssp585` files treat historical (1850–2014) and
ssp585 (2015–2100) as one continuous simulation; note the ensemble members
differ (historical r1i1p1f1, ssp585 r4i1p1f1).

### Scalar time series (`data/processed/`, git-ignored)

`scripts/make_scalar_timeseries.py` writes one `scalars_annual_CESM2_{exp}.nc`
per simulation, each holding these one-value-per-year series on the simulation's
gridded year axis:

- `amoc_strength` (Sv) — from `CESM2_AMOC_experiments.nc`
- `tas_global_mean` (K) — area-weighted global annual-mean temperature
- `tas_interhemispheric_diff` (K) — area-weighted NH-mean minus SH-mean

| File | year axis | notes |
| --- | --- | --- |
| `…historical-ssp585.nc` | 1850–2100 | AMOC spliced (1850–1949, 2001–2100); **1950–2000 missing** |
| `…abrupt-4xCO2.nc` | 1–999 | AMOC years 1–100; NaN after |
| `…piControl.nc` | 700–799 | AMOC years 700–799 |
| `…u03-hos.nc` | 1850–2021 | AMOC years 1850–1949; NaN after |
| `…hosing-0.1Sv-greenland.nc` | 1–100 | AMOC only — no gridded `tas` for this run |

The AMOC series in `CESM2_AMOC_experiments.nc` are the **first 100 years** of
each run, placed at the start of that run's year axis (the historical case
splices its two windows). Temperature scalars are derived from the annual `tas`
files — area weighting commutes with the month-length-weighted annual mean, so
this is exact (verified to 0 K against recomputation from monthly data). Area
weighting uses exact zonal-band weights, `sin(edge_N) − sin(edge_S)`, which
handle the FV grid's half-width polar cells. Regressions drop years with any
missing dependent/predictor value (see `CLAUDE.md`).

## Analysis

### Pooled per-grid-point regressions

`scripts/run_regressions.py` regresses a gridded annual-mean **predictand** (one
time series per grid cell) on the scalar indices `tas_global_mean` (Tglob),
`tas_interhemispheric_diff` (dT_NS), and `amoc_strength` (AMOC). It runs for two
predictands: **`tas`** (temperature, K) and **`precip`** (the pooled-prototype
field — total `pr` for historical-ssp585/abrupt-4xCO2, convective `prc` for
piControl/u03-hos; see precip caveats above).

The years of all four simulations (historical-ssp585, abrupt-4xCO2, piControl,
u03-hos; greenland-hosing is excluded — no gridded field) are **pooled into one
fit per grid cell** with a single common intercept and no per-run fixed effects.
Pooling exploits the runs' disagreement about how the indices co-vary (piControl
~uncorrelated; u03-hos flips the sign of dT_NS), sharply reducing the within-run
collinearity. All sets use one common sample: the years where every predictor is
present (= AMOC-present years), **500 rows** (historical-ssp585 200, others 100).

Nine predictor sets are produced (one multi-panel coefficient map per set, per
predictand):

| Set | Predictors |
| --- | --- |
| 1–3 | each index alone: Tglob; dT_NS; AMOC |
| 4–6 | combinations: Tglob+dT_NS; Tglob+AMOC; Tglob+dT_NS+AMOC |
| 7 | orthogonalized, order tas→NS→AMOC: `Tglob`, `dT_NS⊥Tglob`, `AMOC⊥(Tglob,dT_NS)` |
| 8 | orthogonalized, order tas→AMOC→NS: `Tglob`, `AMOC⊥Tglob`, `dT_NS⊥(Tglob,AMOC)` |
| 9 | full quadratic (centered): Tglob, Tglob², AMOC, AMOC², dT_NS, dT_NS², Tglob·AMOC, Tglob·dT_NS, AMOC·dT_NS |

- **Sets 4–6** use full multiple OLS → each map is that predictor's **partial**
  coefficient (effect holding the others fixed).
- **Sets 7–8** are Gram–Schmidt orthogonalizations (`add_orthogonalized_columns`):
  each residual column is the index with the earlier ones regressed out, so the
  columns are mutually orthogonal (VIF = 1) and give a hierarchical decomposition
  whose attribution depends on the chosen order (compare 7 vs 8).
- **Set 9** is the full quadratic response surface (`add_quadratic_columns`); the
  three base indices are **centered on their pooled means** before squares and
  products are formed (essential for conditioning: cond(XᵀX) drops from ~1e20 to
  ~1e5). Its 9 term coefficients are mapped on a 3×3 grid.

Coefficient maps (`src/output.py`) use a diverging colormap with symmetric bounds
(white = 0; `RdBu_r` for tas with warm = red, `RdBu` for precip with wet = blue)
and **stipple cells where p > 0.05**. Each set writes a PDF and a NetCDF of the
coefficient/SE/t/p/R² fields to `data/output/regression/<predictand>/`, plus a
caveats `README.txt`. `scripts/plot_predictor_scatter.py` writes
`data/output/regression/predictor_scatter.pdf`, a 4-panel scatter of the pooled
predictors colored by simulation.

Caveats: p-values are **nominal OLS** (within-run autocorrelation makes them
optimistic — an effective-degrees-of-freedom correction is planned but not yet
applied); the 3-index set retains high collinearity (VIF ≈ 22) and the quadratic
set even after centering (max VIF ≈ 1500), so those coefficients are weakly
constrained. Fits are validated against `statsmodels` (agreement < 1e-6), and the
global mean of the `tas`-on-Tglob coefficient is exactly 1.0 (a consistency check).

## Reproducing the results

After placing the input files in `data/input/` (see [Data](#data)) and installing
dependencies, run, in order:

```bash
python scripts/make_annual_means.py        # data/processed/{tas,pr,prc}_annual_*.nc
python scripts/make_scalar_timeseries.py   # data/processed/scalars_annual_*.nc
python scripts/run_regressions.py          # data/output/regression/{tas,precip}/coef_set*.{pdf,nc}
python scripts/plot_predictor_scatter.py   # data/output/regression/predictor_scatter.pdf
```

Each script is a thin wrapper over `src/` and prints what it writes. All outputs
land under `data/` (git-ignored) and are fully regenerable from the inputs.

## Status

Preprocessing (`scripts/make_annual_means.py`,
`scripts/make_scalar_timeseries.py`), pooled per-grid-point regression analysis
(`scripts/run_regressions.py`, sets 1–9 for tas and precip), and predictor
scatter (`scripts/plot_predictor_scatter.py`), built on `src/data_loader.py`,
`src/regression.py`, and `src/output.py`.
