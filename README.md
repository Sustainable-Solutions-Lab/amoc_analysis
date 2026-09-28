# AMOC Analysis

Statistical analysis (primarily linear regression) of climate model output from
the **CESM** model (currently CESM1 NAHosMIP runs; formerly CESM2), with a
focus on the Atlantic Meridional Overturning Circulation (AMOC) and its
relationships to other climate variables.

## Goals

- Extract AMOC-relevant diagnostics from CESM output (e.g. AMOC streamfunction
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
- **xarray**, **netCDF4**, **dask** — reading CESM NetCDF output
- **scipy**, **statsmodels** — linear regression and statistics
  (closed-form vectorized OLS for the per-grid-cell maps; `statsmodels`
  validates it and is available for single fits)
- **matplotlib**, **cartopy** — figures and coastline maps (Cartopy downloads
  Natural Earth coastline data on first use, which needs network access)

## Repository layout

```
amoc_analysis/
├── data/
│   ├── input/      # CESM model output (read-only — do not modify)
│   └── output/     # generated results, tables, figures (git-ignored)
├── src/            # analysis modules (data loading, regression, plotting)
├── scripts/        # thin command-line wrappers around src modules
├── requirements.txt
├── README.md
└── CLAUDE.md       # coding conventions for this project
```

## Data

Model output lives in `./data/input/` (NetCDF). Treat this directory as read-only
reference data (see `CLAUDE.md`). The files are too large for git and are not
tracked, so **a colleague has to obtain them separately** and put them in
`data/input/` under the exact names below. The names are hard-coded in
`src/data_loader.py` (`EXPERIMENTS`, `AMOC_FILE`).

> **Model version.** These runs come from **CESM1**, not CESM2. The evidence: the
> `B1850CN` compset, the `f19g16` grid, a 26-level CAM initial file
> (`cami_0000-01-01_1.9x2.5_L26`), and the upstream path `…/CESM1_AMOC_Data/…` in
> each file's `history`. Processed files are therefore tagged `CESM1`
> (`data_loader.SOURCE_ID`). The earlier CESM2 inputs (monthly CMORized and CAM
> files) are kept for reference in `data/input/old_data/`, and the code no longer
> reads them.

> **TODO (data source):** record where these files come from (archive/DOI/URL or
> internal path) so that the inputs can be reproduced. Per their `history`
> attribute, they were extracted with CDO from
> `/oak/stanford/groups/sjdavis/Mahendra/CESM1_AMOC_Data/Output_Extracted/`.

### Input file format

There is one file per simulation, named `<CESM case>_annual_mean.nc`
(netCDF 64-bit offset, CF-1.0, written by CDO 2.1.1):

- **Annual means only.** The upstream processing already averaged the fields to
  one value per year, so no monthly data is available. `cdo seltimestep` trimmed
  each run to its first 100/105/301 years.
- **Dimensions** are `time` (unlimited, 1 step per year) × `lat` = 96 × `lon` = 144.
  This is the CAM finite-volume 1.9° × 2.5° grid. `lat` runs from −90 to 90 and
  includes the half-width polar cells; `lon` runs from 0 to 357.5 °E. All fields
  are `float32` `(time, lat, lon)`. None contains `NaN`.
- **Time axis.** `time` is `double` with `units = "years since YYYY-7-2 00:00:00"`
  and `calendar = "365_day"`, values 0, 1, 2, …. `cdo settaxis` wrote these labels
  (mid-year stamps); they are not the model's own dates. CF `years` units cannot
  be decoded by xarray/cftime, so the loader opens files with
  `decode_times=False` and converts `time` into an integer calendar `year` equal
  to the reference year plus the offset.
- **Variables.** Every file holds the same 39 raw CAM history fields, all
  `cell_methods = "time: mean"`, with native CAM names and units:

| Group | Variables | Units |
| --- | --- | --- |
| Near-surface state | `TREFHT` (2 m temperature), `TREFMNAV`/`TREFMXAV` (mean daily min/max of TREFHT), `QREFHT` (specific humidity), `RHREFHT` (relative humidity) | K; kg/kg; fraction |
| Precipitation | `PRECT` (total), `PRECC` (convective), `PRECL` (large-scale), `PRECSH` (shallow convective), `PRECSC`/`PRECSL` (convective / large-scale snow, water equiv.) | m/s |
| Surface water & heat fluxes | `QFLX` (surface water flux), `LHFLX` (latent), `SHFLX` (sensible) | kg/m²/s; W/m² |
| Surface radiation | `FSDS`, `FSDSC`, `FSNS`, `FSNSC`, `FLDS`, `FLDSC`, `FLNS`, `FLNSC`, `SRFRAD` | W/m² |
| TOA radiation | `SOLIN`, `FSNT`, `FSNTC`, `FLNT`, `FLNTC`, `FLUT`, `FLUTC` | W/m² |
| Clouds | `CLDTOT`, `CLDLOW`, `CLDMED`, `CLDHGH` (vertically integrated cloud fraction); `SWCF`, `LWCF` (cloud forcing) | fraction; W/m² |
| Other | `TMQ` (precipitable water), `SNOWHICE`/`SNOWHLND` (snow depth over ice / land, water equiv.) | kg/m²; m |

(Suffix `C` = clear-sky. `S` = surface and `T` = top of model. `N` = net,
`D` = downwelling and `U` = upwelling.)

**AMOC is not in these files.** It comes from a separate file (see
[below](#amoc-strength-time-series)).

### Available simulations

Nine runs: a 3 × 3 matrix of CO₂ level (1×, 2×, 4× preindustrial) × North
Atlantic freshwater hosing (NAHosMIP protocol: −0.3, 0, +0.3 Sv). Each run's
**case name** is built from those two forcings (`data_loader.case_name`):

- `[124]xCO2` for runs without hosing, e.g. `2xCO2`. `1xCO2` is the
  preindustrial control.
- `[124]xCO2_[pm]03Sv` for hosed runs, where `p` = +0.3 Sv (freshwater added) and
  `m` = −0.3 Sv (freshwater removed), e.g. `4xCO2_m03Sv`.

The case name labels the run everywhere in the code and in processed file names.
The table follows the case-grid layout, with CO₂ level as rows and hosing as
columns:

| Case name | File | CO₂ | Hosing | Years (n) |
| --- | --- | --- | --- | --- |
| `1xCO2_m03Sv` | `B1850CN_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc` | 1× | −0.3 Sv | 2051–2155 (105) |
| `1xCO2` | `B1850CN_f19g16_GCC_piCtrl300yr_annual_mean.nc` | 1× | 0 | 1850–2150 (301) |
| `1xCO2_p03Sv` | `B1850CN_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc` | 1× | +0.3 Sv | 2051–2155 (105) |
| `2xCO2_m03Sv` | `B1850CN_2xCO2_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc` | 2× | −0.3 Sv | 2051–2155 (105) |
| `2xCO2` | `B1850CN_2xCO2_noh_f19g16_yr200_annual_mean.nc` | 2× | 0 | 2051–2150 (100) |
| `2xCO2_p03Sv` | `B1850CN_2xCO2_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc` | 2× | +0.3 Sv | 2051–2155 (105) |
| `4xCO2_m03Sv` | `B1850CN_4xCO2_neghos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc` | 4× | −0.3 Sv | 2051–2155 (105) |
| `4xCO2` | `B1850CN_4xCO2_noh_f19g16_yr200_annual_mean.nc` | 4× | 0 | 2051–2150 (100) |
| `4xCO2_p03Sv` | `B1850CN_4xCO2_hos0p3Sv_f19g16_NAHosMIP_v5_annual_mean.nc` | 4× | +0.3 Sv | 2051–2150 (100) |

Figures that map several cases use this grid, available as
`data_loader.CASE_GRID`: rows are CO₂ = 1, 2, 4× (top to bottom) and columns are
hosing = `m03Sv`, 0 Sv, `p03Sv` (left to right).

**Caveats (inferred from the files; please confirm):**

- **Forcing values come from the CESM case names in the file names.** `hos0p3Sv`
  is read as +0.3 Sv of freshwater added to the North Atlantic, `neghos0p3Sv` as
  an equal amount removed (−0.3 Sv), and `noh` as no hosing.
- **Branch point.** Every run except the `1xCO2` control is labeled from 2051, which is control
  year 201 on the control's 1850-based labels. This matches the `yr200` in the
  no-hosing file names. The year labels come from `cdo settaxis`, so an
  experiment's year and the control's year only correspond if the runs really
  branched there.
- **Unequal lengths.** Run lengths differ: 100 years for `2xCO2`, `4xCO2` and
  `4xCO2_p03Sv`, and 105 years for the other hosing runs. Pooled regressions
  use whatever years each run has.
- **Mislabeled `case` attribute.** The `2xCO2` file's global `case` attribute reads
  `B1850CN_f19g16_GCC_piCtrl300yr`, but its data warm steadily (global-mean
  `TREFHT` 287.3 → 288.8 K over 100 years), unlike the control (~286.9 K). The file
  name, not the attribute, identifies the run.

### Variable mapping and units (`src/data_loader.py`)

The analysis uses CMIP variable names. `data_loader.VARIABLES` maps them to
their CAM sources and converts the units:

| Analysis var | CAM source | Conversion | Output units |
| --- | --- | --- | --- |
| `tas` | `TREFHT` | none | K |
| `prc` (convective precip) | `PRECC` | × 1000 kg m⁻³ | kg m⁻² s⁻¹ |
| `pr` (total precip) | `PRECT` (= `PRECC` + `PRECL`) | × 1000 kg m⁻³ | kg m⁻² s⁻¹ |

CAM precipitation is a liquid-water-equivalent rate, and in these files the
`m/s` label is correct. The global-mean `PRECT` (2.86 mm day⁻¹) equals the
global-mean evaporation `QFLX`, so multiplying by the density of water gives the
CMIP mass flux. The old CESM2 CAM files behaved differently: their precipitation
was labeled `m/s` but already held kg m⁻² s⁻¹. Every run supplies all three
variables, so the `prc` and `pr` analyses both pool all nine runs.

Loader entry points:

- `open_experiment(case)` returns the raw Dataset, with all 39 CAM fields, on an
  integer `year` dimension. Use it for any field not in `VARIABLES`.
- `load_annual_field(case, var)` returns a `(year, lat, lon)` DataArray in CMIP
  names and units, with provenance attributes (`source_file`, `source_variable`,
  `original_units`, `conversion_factor`).
- `amoc_strength_on_years(case, years)` returns the AMOC series aligned to `years`
  (NaN where not covered).

### AMOC strength time series

**Required, not yet supplied:** `data/input/AMOC_CESM1_B1850CN_f19g16.nc`
(`data_loader.AMOC_FILE`). The loader expects **one variable per case name**
from the table above (`1xCO2`, `2xCO2_p03Sv`, …), each a 1-D series in **Sv** on
an integer calendar `year` coordinate that uses the same labels as the gridded
files. `amoc_strength_on_years` aligns each series by year. Years the file does
not cover become NaN and are dropped from regressions (complete-case deletion,
see `CLAUDE.md`). If the delivered file's layout differs, adapt the loader to it.
`scripts/make_scalar_timeseries.py` fails with `FileNotFoundError` until the file
exists.

### Processed annual fields (`data/processed/`, git-ignored)

`scripts/make_annual_means.py` writes one file per variable per run,
`{tas,prc,pr}_annual_CESM1_{case}.nc` (27 files). Each is a `(year, lat, lon)`
field in CMIP names and units. The variable carries provenance attributes, and
the file's global attributes record `source_id`, `experiment`, `co2_multiple` and
`hosing_sv`. No averaging happens here, because the inputs are already annual.

### Scalar time series (`data/processed/`, git-ignored)

`scripts/make_scalar_timeseries.py` writes one `scalars_annual_CESM1_{case}.nc` per
run. Each holds these series on the run's year axis:

- `amoc_strength` (Sv), from `AMOC_FILE`
- `tas_global_mean` (K), the area-weighted global annual-mean temperature
- `tas_interhemispheric_diff` (K), the area-weighted NH mean minus the SH mean
- `precip_centroid_lat_20`, `precip_centroid_lat_30` (°N), the **precipitation-mass
  centroid latitude** (an ITCZ-position index) over 20°S–20°N and 30°S–30°N. It is
  the area- and precipitation-weighted mean latitude of the zonal-mean convective
  precipitation (`prc`), `Σ φ·P·a / Σ P·a`. Unlike an argmax, the centroid covers
  *both* branches of a double ITCZ, so it changes continuously instead of jumping
  between them.

Area weighting uses exact zonal-band weights, `sin(edge_N) − sin(edge_S)`, which
handle the FV grid's half-width polar cells at ±90°.

## Analysis

> **Note:** the method descriptions below still apply, but the run names, sample
> sizes, fitted values and scenario numbers were written for the **earlier CESM2
> dataset** (historical-ssp585, abrupt-4xCO2, piControl, u03-hos). They will be
> updated once the analyses are re-run on the CESM1 runs above, which requires the
> AMOC file. The monthly path (`make_monthly_means.py` and its dependents) needs
> monthly input and cannot run on the annual-only CESM1 files.

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
behave and relate to the predictors. Like the gridded regressions it produces the
**decadal10** variant by default (the annual variant is opt-in via `--do-annuals`).

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
- **PC regression:** the retained PCs are regressed on the selected predictor sets
  (5 & 10 by default; all ten with `--all-sets`, including the orthogonalized,
  quadratic, and interaction columns) with an intercept; the PC-space coefficients
  (coef/SE/t/p) are saved.

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
- `pc_prediction.pdf` — one **page per richer set** among the 3-index set 6, the
  quadratic set 9, and the interaction set 10 that was actually fit (only set 10 by
  default; all three with `--all-sets`): the fitted X·β overlaid on the actual PC
  over time, one panel per simulation — a direct view of how well the scalars
  predict each EOF weighting.

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
(Tglob, dT_NS, AMOC), using the same predictor sets (5 & 10 by default, all ten with
`--all-sets`) and the same decadal10 pooling (annual via `--do-annuals`) as the
gridded regressions. Because the response is a single series per simulation-year
(not a gridded field or PCs), it uses `regression.build_pooled_scalar` and
`regression.fit_scalar_ols` (a 1-D OLS with the same normal-equations math as the
gridded fit, validated against `statsmodels`, plus 95 % confidence intervals). The
pooled common sample is the same AMOC-complete 551 years (55 decadal blocks).
Outputs go to `data/output/itcz/{band20,band30}/[decadal10/]`:

- `coef_table_{annual,decadal10}.csv` — coef, SE, t, p, 95 % CI per parameter,
  with R² and n, for every set (the scalar analog of the gridded coefficient maps).
- `itcz_fit_set{1..10}_{labels}.nc` — the per-set fit Datasets.
- `itcz_timeseries.pdf` — the centroid latitude per simulation, annual + decadal
  overlay (`scripts/plot_itcz_regressions.py`; band level only).
- `itcz_scatter.pdf` — ITCZ latitude vs each single predictor with the OLS line,
  95 % CI band, and slope ± SE / R² / p annotated.
- `itcz_predicted_vs_observed.pdf` — predicted vs observed centroid latitude for
  the multi-predictor sets (5 & 10 by default; 5, 6, 10 with `--all-sets`), with the
  1:1 line and R² (shows how well the *joint* regression reproduces the ITCZ across
  runs).
- `itcz_coefficients.pdf` — partial-slope (coef ± SE) bar charts for the same sets,
  blue/red by sign and hatched where not significant.

The three regression figures (`itcz_scatter`, `itcz_predicted_vs_observed`,
`itcz_coefficients`) are written for the decadal10 sample (in the `decadal10/`
subdir) by default; `--do-annuals` additionally writes them for the annual sample
(in the band directory). The `itcz_timeseries.pdf` overview is always written.

The interhemispheric temperature difference is the strongest single predictor of
the centroid (annual R² ≈ 0.54 at 20°, 0.75 at 30°): the ITCZ shifts toward the
warmer hemisphere. AMOC adds substantial skill (single-predictor R² ≈ 0.36–0.45),
and the three-predictor set explains the bulk of the variance (annual R² ≈
0.82–0.89, decadal ≈ 0.96). (Values are for the convective-`prc` centroid over the
AMOC-complete 551-year sample.)

## Reproducing the results

After placing the input files in `data/input/` (see [Data](#data)) and installing
dependencies, run, in order:

```bash
python scripts/make_annual_means.py        # data/processed/{tas,prc,pr}_annual_CESM1_*.nc
python scripts/make_scalar_timeseries.py   # data/processed/scalars_annual_CESM1_*.nc (needs the AMOC file)
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
`scripts/plot_scalar_timeseries.py`), the additive EOF / principal-component path
(`scripts/run_eof_regressions.py`, built on `src/eof.py`), the decadal
scenario-prediction maps (`scripts/predict_scenarios.py`), and the ITCZ-centroid
scalar regressions (`scripts/run_itcz_regressions.py`,
`scripts/plot_itcz_regressions.py`), all built on `src/data_loader.py`,
`src/regression.py`, and `src/output.py`. The regression, EOF, and ITCZ analyses run
in a **decadal10** (10-year block-mean, slow-timescale) variant by default; the
**annual** (interannual) variant is opt-in via `--do-annuals`, and only sets 5 & 10
are fit unless `--all-sets` is given.

A parallel **monthly** (per-calendar-month) path —
`scripts/make_monthly_means.py`, `scripts/run_monthly_regressions.py`,
`scripts/predict_scenarios_monthly.py` — regresses each calendar month's gridded
field on the annual indices and projects the monthly scenario changes, writing
`(month, param, lat, lon)` NetCDFs and rasterized all-months PDFs (`--vector` for
vector output).
