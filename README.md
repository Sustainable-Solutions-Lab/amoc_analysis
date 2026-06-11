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
AMOC_4models_hist_ssp585.nc
tas_Amon_CESM2_historical_r1i1p1f1_gn_18500115-20141215.nc
tas_Amon_CESM2_ssp585_r1i1p1f1_gn_20150115-21001215.nc
tas_Amon_CESM2_abrupt-4xCO2-002.nc
tas_Amon_CESM2_piControl_070001-079912.nc
tas_Amon_CESM2_u03-hos_1850001-202112.nc
prc_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc
prc_Amon_CESM2_ssp585_r4i1p1f1_gn_201501-206412.nc
prc_Amon_CESM2_ssp585_r4i1p1f1_gn_206501-210012.nc
prc_Amon_CESM2_abrupt-4xCO2_r1i1p1f1_gn_000101--099912.nc
pr_Amon_CESM2_piControl_070001-079912.nc
pr_Amon_CESM2_u03-hos_1850001-202112.nc
```

The **main** precipitation analysis uses **convective precipitation (`prc`)** across
all four runs (total `pr` is not available for all of them). The
`historical`/`ssp585`/`abrupt-4xCO2` runs supply `prc` directly (ssp585 split across
two files); `piControl`/`u03-hos` supply convective precip under the variable name
`pr` in their raw files.

A **second, total-precipitation (`pr`) analysis** is produced *in addition*, from the
total-`pr` files preserved in `data/input/pr_data/` (`historical`, `ssp585`,
`abrupt-4xCO2` only — no `piControl`/`u03-hos`). It is therefore a **2-run** pooled
analysis (historical-ssp585 + abrupt-4xCO2). Required files in `data/input/pr_data/`:

```
pr_Amon_CESM2_historical_r1i1p1f1_gn_18500115-20141215.nc
pr_Amon_CESM2_ssp585_r1i1p1f1_gn_20150115-21001215.nc
pr_Amon_CESM2_abrupt-4xCO2-001.nc
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

`AMOC_4models_hist_ssp585.nc` — AMOC strength (**Sv**) on a `year` axis 1850–2100
(251 values, no gaps) for four CMIP6 models, one variable each (`CESM2`,
`HadGEM3-GC31-MM`, `CanESM5`, `IPSL-CM6A-LR`). The `CESM2` series is the
**gap-free historical+ssp585 AMOC** used for the `historical-ssp585` run; it
matches the two `CESM2_AMOC_experiments.nc` historical windows to ~5e-5 in their
overlap and fills the former 1950–2000 gap, so regressions can use the entire
historical period. The other runs still draw AMOC from `CESM2_AMOC_experiments.nc`.

### Gridded monthly fields

Near-surface air temperature and precipitation on the CESM2 native grid
(`lat` = 192 × `lon` = 288, nominal 1°), monthly (`Amon`). Two variables across
five experiments:

| Experiment (file stem) | Time range | n (months) | `tas`/`prc` provenance |
| --- | --- | --- | --- |
| `historical_r1i1p1f1_gn` | 1850-01 → 2014-12 | 1980 | CMORized |
| `ssp585_r4i1p1f1_gn`¹ | 2015-01 → 2100-12 | 1032 | CMORized (split into two files) |
| `abrupt-4xCO2`² | 0001-01 → 0999-12 | 11988 | CMORized |
| `piControl_070001-079912` | 0700 → 0800 | 1200 | raw CESM2 (CAM history) |
| `u03-hos_1850001-202112`³ | 1850-02 → 2022-01 | 2064 | raw CESM2 (CAM history) |

**Important provenance caveats:**

- **Two distinct data conventions.** The `historical`, `ssp585`, and
  `abrupt-4xCO2` files are CMORized: temperature is `tas` (Near-Surface Air
  Temperature, **K**) and convective precipitation is `prc` (**kg m⁻² s⁻¹**). The
  `piControl` and `u03-hos` files are raw CESM2 CAM history output: temperature is
  `TREFHT` (Reference height temperature, **K**) and `PRECC`-style **convective**
  precipitation rate (liq + ice) under the variable name `pr`, labeled **m s⁻¹**
  but actually a kg m⁻² s⁻¹ water mass flux (see below). These raw files also carry
  the full CAM metadata set (hybrid-sigma coefficients `hyam`/`hybm`, `co2vmr`,
  `sol_tsi`, etc.). Loading code maps variable names and reconciles units before
  comparing across experiments. The analysis uses convective `prc` for every run.
- ¹ The `ssp585` future is split into two files (2015–2064, 2065–2100) and is
  ensemble member **`r4i1p1f1`** (the `prc` filenames say so; the `tas` ssp585
  filename says r1 but its `variant_label` is also r4). So `tas` and `prc` share
  the same r1-historical → r4-ssp585 splice.
- ² `abrupt-4xCO2` is supplied as a single 999-year monthly run (~2.5 GB per
  variable); the `prc` (`r1i1p1f1`) and `tas` (`-002`) files carry different file
  suffixes but both span years 1–999.
- ³ `u03-hos` is the freshwater-hosing experiment; its time axis spans
  1850–2022 on a `noleap` calendar.

### Processed annual means (`data/processed/`, git-ignored)

`scripts/make_annual_means.py` precomputes month-length-weighted annual means
(correct for the `noleap` calendar) of the gridded fields, following CMIP
variable names: `tas` (K), `prc` (**convective** precipitation, kg m⁻² s⁻¹), and
`pr` (**total** precipitation, kg m⁻² s⁻¹, two runs). The raw CAM files contribute
`TREFHT` (renamed `tas`) and their convective precip (source variable `pr`, renamed
`prc`); total `pr` comes from `data/input/pr_data/`. Each variable carries provenance
attributes (`source_file`, `source_variable`, `original_units`, `annual_mean_method`,
`precip_kind`). Output (10 files):

| File | Coverage |
| --- | --- |
| `{tas,prc}_annual_CESM2_historical-ssp585.nc` | 1850–2100 (251 yr, spliced) |
| `{tas,prc}_annual_CESM2_abrupt-4xCO2.nc` | years 1–999 |
| `tas_annual_CESM2_piControl.nc`, `prc_annual_CESM2_piControl.nc` | years 700–799 |
| `tas_annual_CESM2_u03-hos.nc`, `prc_annual_CESM2_u03-hos.nc` | 1850–2021 |
| `pr_annual_CESM2_historical-ssp585.nc` (total precip) | 1850–2100 (251 yr, spliced) |
| `pr_annual_CESM2_abrupt-4xCO2.nc` (total precip) | years 1–999 |

**Precipitation caveats:**

- **Convective only.** The analysis uses convective precipitation (`prc`) for
  every run — total `pr` was never available for all runs. Do not interpret `prc`
  as total precipitation.
- **Mislabeled source units.** The raw `piControl`/`u03-hos` `prc` source carries
  `units = "m/s"`, but its values are a water mass flux already in kg m⁻² s⁻¹
  (they match the CMIP precip magnitude; true m s⁻¹ precipitation would be ~1000×
  smaller). The data is used as-is without scaling; outputs record this in a
  `units_note` attribute.

The combined `historical-ssp585` files treat historical (1850–2014) and
ssp585 (2015–2100) as one continuous simulation; the ensemble members differ
(historical r1i1p1f1, ssp585 r4i1p1f1), but identically for `tas` and `prc`, so
the predictand and the temperature-derived predictors share the same splice.

### Scalar time series (`data/processed/`, git-ignored)

`scripts/make_scalar_timeseries.py` writes one `scalars_annual_CESM2_{exp}.nc`
per simulation, each holding these one-value-per-year series on the simulation's
gridded year axis:

- `amoc_strength` (Sv) — from `CESM2_AMOC_experiments.nc`, except
  `historical-ssp585`, which uses the gap-free 1850–2100 `CESM2` series in
  `AMOC_4models_hist_ssp585.nc`
- `tas_global_mean` (K) — area-weighted global annual-mean temperature
- `tas_interhemispheric_diff` (K) — area-weighted NH-mean minus SH-mean
- `precip_centroid_lat_20`, `precip_centroid_lat_30` (°N) — the **precipitation-mass
  centroid latitude**, an **ITCZ-position index**, over 20°S–20°N and 30°S–30°N. It
  is the area- and precip-weighted mean latitude of the zonal-mean precipitation,
  `Σ φ·P·a / Σ P·a` (reusing the exact band area weights). Unlike a bare argmax,
  the centroid integrates over *both* branches of a double ITCZ, so it varies
  continuously instead of jumping between branches — important because CESM2 has a
  pronounced southern (double-ITCZ) precip maximum. Present only for runs with a
  gridded precip file (all but greenland-hosing); the source is convective `prc`
  for every run.

| File | year axis | notes |
| --- | --- | --- |
| `…historical-ssp585.nc` | 1850–2100 | AMOC complete 1850–2100 (from `AMOC_4models_hist_ssp585.nc`) |
| `…abrupt-4xCO2.nc` | 1–999 | AMOC years 1–100; NaN after |
| `…piControl.nc` | 700–799 | AMOC years 700–799 |
| `…u03-hos.nc` | 1850–2021 | AMOC years 1850–1949; NaN after |
| `…hosing-0.1Sv-greenland.nc` | 1–100 | AMOC only — no gridded `tas` for this run |

The AMOC series in `CESM2_AMOC_experiments.nc` are the **first 100 years** of
each run, placed at the start of that run's year axis. The `historical-ssp585`
run instead uses the continuous 1850–2100 `CESM2` series from
`AMOC_4models_hist_ssp585.nc` (no 1950–2000 gap), so regressions can use the full
historical period. Temperature scalars are derived from the annual `tas`
files — area weighting commutes with the month-length-weighted annual mean, so
this is exact (verified to 0 K against recomputation from monthly data). Area
weighting uses exact zonal-band weights, `sin(edge_N) − sin(edge_S)`, which
handle the FV grid's half-width polar cells. Regressions drop years with any
missing dependent/predictor value (see `CLAUDE.md`).

## Analysis

### Pooled per-grid-point regressions

`scripts/run_regressions.py` regresses a gridded annual-mean **predictand** (one
time series per grid cell) on the scalar indices `tas_global_mean` (Tglob),
`tas_interhemispheric_diff` (dT_NS), and `amoc_strength` (AMOC). It runs for three
predictands: **`tas`** (temperature, K) and **`prc`** (convective precipitation,
kg m⁻² s⁻¹, all four runs), plus **`pr`** (total precipitation, kg m⁻² s⁻¹, pooled
over only historical-ssp585 + abrupt-4xCO2 — the runs with total-`pr` data; see
precip caveats above). `pr` outputs land in `data/output/regression/pr/` (and
`eof/pr/`), alongside the `tas`/`prc` trees.

The years of all four simulations (historical-ssp585, abrupt-4xCO2, piControl,
u03-hos; greenland-hosing is excluded — no gridded field) are **pooled into one
fit per grid cell** with a single common intercept and no per-run fixed effects.
Pooling exploits the runs' disagreement about how the indices co-vary (piControl
~uncorrelated; u03-hos flips the sign of dT_NS), sharply reducing the within-run
collinearity. All sets use one common sample: the years where every predictor is
present (= AMOC-present years), **551 rows** (historical-ssp585 251 — full 1850–2100
now that AMOC is gap-free — and 100 each from the other three runs).

Ten predictor sets are defined (one multi-panel coefficient map per set, per
predictand). **By default only sets 5 & 10 are produced** (the two used in most
analyses); pass `--all-sets` to any of the regression scripts to produce all ten:

| Set | Predictors |
| --- | --- |
| 1–3 | each index alone: Tglob; dT_NS; AMOC |
| 4–6 | combinations: Tglob+dT_NS; Tglob+AMOC; Tglob+dT_NS+AMOC |
| 7 | orthogonalized, order tas→NS→AMOC: `Tglob`, `dT_NS⊥Tglob`, `AMOC⊥(Tglob,dT_NS)` |
| 8 | orthogonalized, order tas→AMOC→NS: `Tglob`, `AMOC⊥Tglob`, `dT_NS⊥(Tglob,AMOC)` |
| 9 | full quadratic (centered): Tglob, Tglob², AMOC, AMOC², dT_NS, dT_NS², Tglob·AMOC, Tglob·dT_NS, AMOC·dT_NS |
| 10 | Tglob × AMOC interaction (centered): Tglob, AMOC, Tglob·AMOC |

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
- **Set 10** is the global-temperature × AMOC interaction model (reusing the
  centered `add_quadratic_columns` terms): Tglob, AMOC, and Tglob·AMOC. Centering
  the main effects conditions the design and makes each main-effect coefficient
  the response at the *other* index's mean; the interaction coefficient is
  identical to the uncentered form.

Coefficient maps (`src/output.py`) use a diverging colormap with symmetric bounds
(white = 0; `RdBu_r` for tas with warm = red, `RdBu` for prc with wet = blue)
and **stipple cells where p > 0.05**. Each set writes a PDF and a NetCDF of the
coefficient/SE/t/p/R² fields to `data/output/regression/<predictand>/`, plus a
caveats `README.txt`. `scripts/plot_predictor_scatter.py` writes
`data/output/regression/predictor_scatter.pdf`, a 4-panel scatter of the pooled
predictors colored by simulation. `scripts/plot_scalar_timeseries.py` writes
`data/output/regression/predictor_timeseries.pdf`, the predictors as time series
(one panel per simulation): Tglob and ΔT_NS as anomalies from their pooled means on
the left axis (K), AMOC absolute on a right axis (Sv), with the decadal block means
overlaid on the annual lines.

Caveats: p-values are **nominal OLS** (within-run autocorrelation makes them
optimistic — see the decadal variant below, which largely resolves this); the
3-index set retains high collinearity (VIF ≈ 22) and the quadratic set even after
centering (max VIF ≈ 1500), so those coefficients are weakly constrained. Fits are
validated against `statsmodels` (agreement < 1e-6), and the global mean of the
`tas`-on-Tglob coefficient is exactly 1.0 (a consistency check).

### Slow-timescale (decadal) variant

To characterize variability slower than interannual, `run_regressions.py` (and
the EOF script below) produces a **decadal** variant by default (the annual variant
is opt-in via `--do-annuals`). The low-pass is **non-overlapping 10-year block means**
(`data_loader.block_average_on_years`), applied **per run, per contiguous
segment, to both the predictors and the predictand** inside
`regression.build_pooled(block=10)` *before* pooling — so the identical filter
acts on dependent and independent variables, and every downstream step (the
orthogonalized and quadratic columns, the grid OLS, the EOFs) inherits it. Blocks
never span a run boundary or a within-run year gap; with the gap-free
historical-ssp585 AMOC, that run is now one contiguous 1850–2100 segment (25
ten-year blocks; a trailing partial block is dropped). Each block's timestamp is
its midpoint year.

Block averaging is a **decimation**, not a running mean: it collapses each decade
to one ~independent sample (pooled **n ≈ 55**: historical-ssp585 25, others 10),
so the nominal OLS degrees of freedom become honest — this resolves the
annual-variant autocorrelation caveat rather than deferring it, at the cost of
sample size (df ≈ 40 still supports the 9-term set 9). Quadratic and product terms
are formed from the *filtered* bases (filter-then-square), i.e. the genuinely
low-frequency response surface. Decadal results go to
`data/output/regression/<predictand>/decadal10/`; the annual outputs (produced only
with `--do-annuals`) use the same filenames in the top-level `<predictand>/`.

### EOF / principal-component analysis (additive path)

`scripts/run_eof_regressions.py` is an **additive** companion to the direct
per-grid-point maps (it does not replace `run_regressions.py`). It decomposes each
gridded field into empirical orthogonal functions (EOFs) and examines how the
leading principal-component (PC) time series — the EOF *weightings over time* —
behave and relate to the predictors. It is produced for both the annual and the
decadal variant.

Method (`src/eof.py`):

- **Anomalies** are taken about each cell's **grand temporal mean over all pooled
  samples** (not per-run means) — this retains the between-run forced variability
  the predictors are meant to explain.
- **Area-weighted covariance EOF:** anomalies are multiplied by √(zonal-band area
  weight) before an economy SVD and the patterns divided by it afterward, so the
  EOFs are in physical units. No per-cell standard-deviation normalization.
- **Truncation:** two rules combined, the more restrictive winning — keep leading
  modes until cumulative variance reaches **≥ 95 %**, but never keep a mode that
  individually explains **< 1 %** of variance (the per-mode floor drops the long
  low-variance noise tail). Counts are reported per field/variant. `tas` is highly
  low-rank (**2 modes**, EOF1 a global warming pattern, EOF2 an
  AMOC/interhemispheric dipole — the 95 % rule binds). For `prc` the 1 % floor
  binds (the convective-precip field is not low-rank); the retained-mode count is
  reported per variant at run time. `eof_patterns.pdf` maps up to the leading 9
  modes.
- **PC regression:** the retained PCs are regressed on the same predictor sets
  1–10 (including the orthogonalized, quadratic, and interaction columns) with an
  intercept; the PC-space coefficients (coef/SE/t/p) are saved.

Outputs per predictand and variant (`data/output/eof/<predictand>/` and
`…/decadal10/`):

- `eof_patterns.pdf` — the leading EOF spatial patterns + a variance scree.
- `pc_timeseries.pdf` — the **EOF weightings (PCs) over time**, one panel per
  simulation; lines break across genuine year gaps but stay connected across
  regular decadal steps (the historical-ssp585 AMOC is now gap-free).
- `pc_regression_set{N}_*.nc` — OLS of the PCs on each predictor set (PC-space
  coef/SE/t/p and per-mode R²), plus a caveats `README.txt`.

The **decadal** variant additionally renders the PC-on-scalar regression — the EOF
analog of the 2D coefficient maps, with the discrete EOF-mode index replacing the
(lat, lon) grid:

- `pc_regression.pdf` — one **page per predictor set**; each page has one panel per
  retained EOF mode, with a bar per predictor showing the **standardized**
  coefficient β·σ(xⱼ)/σ(PCₘ) (z-scoring predictors and the PC, so bars are
  comparable across modes — raw coefficients scale with each PC's amplitude) and a
  ±SE whisker. Non-significant bars (p > 0.05) are faded; the panel title reports
  R² and the mode's variance share. t/p are scale-invariant and match the per-set
  `pc_regression_set{N}_*.nc`.
- `pc_prediction.pdf` — one **page per set** (the full 3-index set 6, the quadratic
  set 9, and the interaction set 10): the fitted X·β overlaid on the actual PC over
  time, one panel per simulation — a direct view of how well the scalars predict
  each EOF weighting.

The spatial **fingerprint** maps (Σₖ βₖ·EOFₖ, the PC regression projected back to
the grid) are intentionally **not** generated — the EOFs and PC weightings are the
wanted deliverables. The capability remains in `eof.reconstruct_fingerprint`
(verified: with all modes retained it reproduces the direct field regression's
coefficient *and* p-value to Δcoef ~ 1e-10, Δp ~ 1e-8) should maps be wanted later.

### Scenario prediction: where AMOC slowdown exacerbates vs. ameliorates CO₂ change

`scripts/predict_scenarios.py` uses the **decadal** set 5 (Tglob + AMOC) and set 10
(Tglob + AMOC + Tglob·AMOC) coefficient maps to predict end-of-century field changes
and isolate the AMOC-slowdown contribution. It addresses: *where does AMOC slowdown
exacerbate CO₂-induced changes in surface temperature and precipitation, and where
does it ameliorate them?*

The high-CO₂ world **with** AMOC slowdown (SSP585, 2091–2100) is compared against two
counterfactuals **without** slowdown (AMOC restored to the control value), which
bracket the unknown global-mean-temperature effect of the slowdown:

| condition | Tglob (K) | AMOC (Sv) | meaning |
| --- | --- | --- | --- |
| piControl | 287.207 | 17.44 | preindustrial baseline (years 700–799) |
| SSP585 | 293.090 | 7.34 | end-of-century, with slowdown |
| SSP585-adj1 | 294.665 | 17.44 | assm. 1: slowdown cooled by 0.1558 K/Sv × 10.10 Sv ≈ 1.575 K, added back |
| SSP585-adj2 | 293.090 | 17.44 | assm. 2: slowdown had no global-mean-T effect; only AMOC restored |

The 0.1558 ± 0.0042 K/Sv slope is the OLS of global-mean tas on AMOC in the u03-hos
hosing run (a within-experiment correlation, not a transferable causal sensitivity).
For the **`pr`** predictand (no piControl run), the baseline is instead the
historical-ssp585 **1850–1900 mean** (≈287.18 K / 17.84 Sv — essentially the piControl
state). The set-10 centering means are read per predictand from the coef file's
`centering_mean_*` attributes, so the 2-run `pr` fit uses its own centering.

The predicted change between two conditions is `coef · (predictor(X) − predictor(R))`
(the intercept cancels; set 10 evaluates its centered columns and interaction with the
fit's centering means). Each predictand is a **three-page** PDF
(`data/output/scenarios/predicted_change_{tas,prc,pr}.pdf`), rows = set 5 and set 10:
page 1 (2×3) is the change relative to piControl — **SSP585 − piControl**,
**adj1 − piControl**, **adj2 − piControl**; page 2 (2×2) is the AMOC-slowdown effect —
**SSP585 − adj1**, **SSP585 − adj2**; page 3 (2×2) expresses that effect as the
**fractional increase(+)/decrease(−) in the response caused by the slowdown**, i.e.
`(SSP585 − adjN) / (adjN − piControl) × 100` — the AMOC effect divided by the
*no-slowdown CO₂-only* change. Under adj2 global-mean tas is held fixed, so
`SSP585 − adj2` is pure spatial redistribution (global mean exactly 0). Comparing the
sign of the AMOC effect (page 2) against the CO₂-only change (`adjN − piControl`, page 1)
shows where the slowdown adds to (exacerbates) or opposes (ameliorates) the CO₂ response
— e.g. for tas the subpolar North Atlantic cold blob, for prc an ITCZ-shift dipole.
On page 3 the ratio explodes where the denominator crosses zero, so every page-3 panel
uses a common fixed **±100 %** color scale (values beyond saturate) to keep panels
directly comparable.

### ITCZ-position regressions (scalar response)

`scripts/run_itcz_regressions.py` regresses the **scalar** ITCZ index — the
precipitation-mass centroid latitude, for two tropical bands
(`precip_centroid_lat_20`, `precip_centroid_lat_30`) — on the same scalar indices
(Tglob, dT_NS, AMOC), using the same ten predictor sets and the same `annual` /
`decadal10` pooling as the gridded regressions. Because the response is a single
series per simulation-year (not a gridded field or PCs), it uses
`regression.build_pooled_scalar` and `regression.fit_scalar_ols` (a 1-D OLS with
the same normal-equations math as the gridded fit, validated against `statsmodels`,
plus 95 % confidence intervals). The pooled common sample is the same AMOC-complete
500 years (50 decadal blocks). Outputs go to
`data/output/itcz/{band20,band30}/[decadal10/]`:

- `coef_table_{annual,decadal10}.csv` — coef, SE, t, p, 95 % CI per parameter,
  with R² and n, for every set (the scalar analog of the gridded coefficient maps).
- `itcz_fit_set{1..10}_{labels}.nc` — the per-set fit Datasets.
- `itcz_timeseries.pdf` — the centroid latitude per simulation, annual + decadal
  overlay (`scripts/plot_itcz_regressions.py`; band level only).
- `itcz_scatter.pdf` — ITCZ latitude vs each single predictor with the OLS line,
  95 % CI band, and slope ± SE / R² / p annotated.
- `itcz_predicted_vs_observed.pdf` — predicted vs observed centroid latitude for
  the multi-predictor sets 5, 6, 10, with the 1:1 line and R² (shows how well the
  *joint* regression reproduces the ITCZ across runs).
- `itcz_coefficients.pdf` — partial-slope (coef ± SE) bar charts for sets 5, 6, 10,
  blue/red by sign and hatched where not significant.

The three regression figures (`itcz_scatter`, `itcz_predicted_vs_observed`,
`itcz_coefficients`) are written for both the annual sample (in the band directory)
and the decadal10 sample (in the `decadal10/` subdir).

The interhemispheric temperature difference is the strongest single predictor of
the centroid (annual R² ≈ 0.72): the ITCZ shifts toward the warmer hemisphere. AMOC
adds substantial skill (single-predictor R² ≈ 0.4–0.5), and the three-predictor set
explains the bulk of the variance (annual R² ≈ 0.82–0.89, decadal ≈ 0.96–0.97).

## Reproducing the results

After placing the input files in `data/input/` (see [Data](#data)) and installing
dependencies, run, in order:

```bash
python scripts/make_annual_means.py        # data/processed/{tas,prc,pr}_annual_*.nc
python scripts/make_scalar_timeseries.py   # data/processed/scalars_annual_*.nc
python scripts/run_regressions.py          # data/output/regression/{tas,prc,pr}/[decadal10/]coef_set*.{pdf,nc}
python scripts/plot_predictor_scatter.py   # data/output/regression/predictor_scatter.pdf
python scripts/plot_scalar_timeseries.py   # data/output/regression/predictor_timeseries.pdf
python scripts/run_eof_regressions.py      # data/output/eof/{tas,prc,pr}/[decadal10/]{eof_patterns,pc_timeseries}.pdf, pc_regression_set*.nc
python scripts/predict_scenarios.py        # data/output/scenarios/predicted_change_{tas,prc,pr}.pdf
python scripts/run_itcz_regressions.py     # data/output/itcz/{band20,band30}/[decadal10/]coef_table_*.csv, itcz_fit_set*.nc
python scripts/plot_itcz_regressions.py    # data/output/itcz/{band20,band30}/{itcz_timeseries,itcz_scatter}.pdf

# Monthly (per-calendar-month) analysis (optional; larger intermediate files):
python scripts/make_monthly_means.py          # data/processed/{tas,prc,pr}_monthly_*.nc (year, month, lat, lon)
python scripts/run_monthly_regressions.py     # data/output/regression_monthly/{tas,prc,pr}/[decadal10/]coef_set*.{pdf,nc}
python scripts/predict_scenarios_monthly.py   # data/output/scenarios_monthly/predicted_change_{tas,prc,pr}.pdf
```

The four set-fitting scripts (`run_regressions.py`, `run_eof_regressions.py`,
`run_itcz_regressions.py`, `plot_itcz_regressions.py`) default to **only sets 5 & 10**
and **only the decadal10** (slow-timescale) smoothing; add `--all-sets` to produce all
ten sets and `--do-annuals` to also produce the annual (interannual) variant (the flags
compose). `predict_scenarios.py` uses sets 5 & 10 from the decadal10 run, so it needs
only the default run.

**Monthly regressions.** `run_monthly_regressions.py` fits the same per-grid-point OLS
**separately for each calendar month** (Jan…Dec): the response is that month's gridded
tas/prc/pr field, the predictors are the *annual* indices (Tglob, AMOC, …) of the same
year, and the decadal variant uses the 10-year mean of each calendar month. The 12 months
are stacked into one PDF and one NetCDF per predictand×set — NetCDF dims
`(month, param, lat, lon)` (plus `nobs(month)`, and `centering_mean_{Tglob,AMOC}(month)`
for the cross-product set 10). It honors `--all-sets` / `--do-annuals` like the others,
and writes **rasterized** map fields inside the PDF by default (small files); pass
`--vector` for full vector PDFs. `predict_scenarios_monthly.py` then maps the predicted
monthly-mean change for the SSP5-8.5 scenarios (12-month grids). Note the monthly
intermediate files in `data/processed/` are ~12× the size of the annual ones (zlib
compressed).

Each script is a thin wrapper over `src/` and prints what it writes. All outputs
land under `data/` (git-ignored) and are fully regenerable from the inputs.

Utility: `python scripts/split_pdf_into_pages.py <file.pdf>` splits a multi-page
figure PDF into one file per page (`<file>_page01.pdf`, `_page02.pdf`, … in the
same directory) for pasting individual panels into LaTeX. Add `--png` (optionally
`--dpi N`, default 300) to rasterize the pages to high-resolution PNGs instead —
much lighter for a LaTeX engine to load than the vector PDFs.

## Status

Preprocessing (`scripts/make_annual_means.py`,
`scripts/make_scalar_timeseries.py`), pooled per-grid-point regression analysis
(`scripts/run_regressions.py`, sets 1–10 for tas, prc, and pr), predictor scatter
and time series (`scripts/plot_predictor_scatter.py`,
`scripts/plot_scalar_timeseries.py`), and the additive EOF / principal-component
path (`scripts/run_eof_regressions.py`, built on `src/eof.py`), and the decadal
scenario-prediction maps (`scripts/predict_scenarios.py`), all built on
`src/data_loader.py`, `src/regression.py`, and `src/output.py`. The regression and
EOF analyses run in a **decadal10** (10-year block-mean, slow-timescale) variant by
default; the **annual** (interannual) variant is opt-in via `--do-annuals`.
