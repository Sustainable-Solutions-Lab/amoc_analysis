# AMOC Analysis

Statistical analysis (primarily linear regression) of climate model output from
the **CESM** model (currently CESM1.2 NAHosMIP runs; formerly CESM2), with a
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
`src/data_loader.py` (`INPUT_FILES`, `AMOC_FILE`).

> **Model version.** These runs come from **CESM1.2**, not CESM2. The files show
> CESM1: the `B1850CN` compset, the `f19g16` grid, a 26-level CAM initial file
> (`cami_0000-01-01_1.9x2.5_L26`), and the upstream path `…/CESM1_AMOC_Data/…` in
> each file's `history`. They do not record the minor version (the `Version`
> attribute is an unfilled `$Name$`); 1.2 is per the data provider. Processed
> files are tagged `CESM1` (`data_loader.SOURCE_ID`). The earlier CESM2 inputs (monthly CMORized and CAM
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

**AMOC is not in these files.** It comes from separate CSV files (see
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

### Analysis variables and units (`src/data_loader.py`)

`data_loader.VARIABLES` defines **46 analysis variables**. Each is available for
all nine runs, and each is a regression predictand. Every variable has a
`definition` (a formula in CAM field names), `units` and a `long_name`, and each
loaded field records these, plus its source file, as attributes.

- **CMIP names (3):** `tas` = `TREFHT` (K); `prc` = `PRECC` (convective
  precipitation); `pr` = `PRECT` (total precipitation, `PRECC` + `PRECL`), both in
  mm day⁻¹.
- **CAM names (36):** every other field in the files keeps its CAM name (table
  above). Water fluxes (`PRECL`, `PRECSH`, `PRECSC`, `PRECSL`, `QFLX`) are converted
  to mm day⁻¹, like `pr` and `prc`. All other fields keep their native units.
- **Derived (7):**

| Variable | Definition | Units | Meaning |
| --- | --- | --- | --- |
| `pr_minus_evap` | `PRECT` × 8.64e7 − `QFLX` × 86400 | mm day⁻¹ | precipitation minus evaporation (P − E) |
| `prsn` | (`PRECSC` + `PRECSL`) × 8.64e7 | mm day⁻¹ | snowfall (water equivalent) |
| `toa_net_down` | `FSNT` − `FLNT` | W m⁻² | net downward radiation at top of model |
| `sfc_net_energy_down` | `FSNS` − `FLNS` − `LHFLX` − `SHFLX` − 3.337e8 × (`PRECSC` + `PRECSL`) | W m⁻² | net downward surface energy flux (radiative + turbulent + melting of snowfall; see below) |
| `cloud_radiative_effect` | `SWCF` + `LWCF` | W m⁻² | net cloud radiative effect at top of model |
| `diurnal_temperature_range` | `TREFMXAV` − `TREFMNAV` | K | mean diurnal temperature range |
| `planetary_albedo` | 1 − `FSNT` / `SOLIN` | 1 | planetary albedo (from annual-mean fluxes) |

As checks, the control's global means are P − E ≈ 0 (−0.0006 mm day⁻¹), and over
1850–2150 a top-of-model imbalance of −0.13 W m⁻² and a surface energy flux of
−0.16 W m⁻², which agree as they must when the atmosphere stores no energy.

**Surface energy flux and snowfall.** CAM's `LHFLX` includes only the latent heat of
vaporization, so `sfc_net_energy_down` also subtracts the latent heat of fusion
needed to melt the snow that falls: L_f ρ_w (`PRECSC` + `PRECSL`), with the CESM
constants L_f = 3.337×10⁵ J kg⁻¹ and ρ_w = 1000 kg m⁻³
(`data_loader.SNOW_MELT_ENERGY_PER_M`). The term is about 0.5–0.7 W m⁻² in the
global mean and concentrated where snow falls. Without it the surface flux runs
0.5–0.7 W m⁻² above the top-of-model flux; with it the two agree to within
0.03 W m⁻² in the 2101–2150 means of the 1×, 2× and 4×CO₂ runs.

**Water-flux units.** All water fluxes (precipitation, snowfall, evaporation,
P − E) are in **mm day⁻¹** of liquid water (`data_loader.WATER_FLUX_UNITS`). CAM
precipitation is a liquid-water-equivalent rate, and in these files the `m/s` label
is correct: m s⁻¹ × 1000 mm m⁻¹ × 86400 s day⁻¹ = × 8.64e7. `QFLX` is a mass flux in
kg m⁻² s⁻¹, and 1 kg m⁻² of water is a 1 mm layer, so it needs only × 86400. As a
check, the control's global-mean `pr` (2.877 mm day⁻¹) equals its global-mean
`QFLX`.

**Corrected source-attribute errors:**

- `RHREFHT` is labeled `fraction`, but its values run from about 20 to 110, so it
  is in **%**.
- `SRFRAD` is labeled "Net radiative flux at surface", but it equals
  `FSNS` + `FLDS` to within 0.04 W m⁻², i.e. absorbed shortwave plus
  *downwelling* longwave, which is not a net flux. The net surface radiation is
  `FSNS` − `FLNS`.

None of the 39 fields contains missing values. Four fields have cells whose
value never changes in any run over 2051–2150:

- `SNOWHICE` (9740 cells): no sea ice ever forms there.
- `SNOWHLND` (9450 cells): ocean, plus ice-sheet cells held at the land model's
  1 m snow cap.
- `PRECSC` (5530 cells, 46°S–37°N): no convective snow.
- `CLDLOW` (356 cells): probably high terrain such as Tibet and Antarctica, where
  CAM defines no low cloud (not checked cell by cell).

At those cells the fit is exact, so every slope coefficient and its standard error
are 0, and the t- and p-values are undefined (NaN). They show as unstippled
white on the maps.

Loader entry points:

- `open_experiment(case)` returns the raw Dataset, with all 39 CAM fields, on an
  integer `year` dimension. Use it for any field not in `VARIABLES`.
- `load_annual_field(case, var)` returns a `(year, lat, lon)` DataArray of any
  `VARIABLES` entry, in the analysis units. Its attributes are `units`,
  `long_name`, `definition` and `source_file`.
- `amoc_strength_on_years(case, years)` returns the AMOC series aligned to `years`
  (NaN where not covered).

### AMOC strength time series

Two CSV files in `data/input/` hold AMOC strength (Sv) at **26.5°N**, the
latitude of the RAPID array. They have one column per run, labeled as below.
The mapping from column label to case name is `INPUT_FILES[...]["amoc_column"]`
in `src/data_loader.py`.

- **`amoc_timeseries_26p5N_9experiments_v5_annual.csv`** (`data_loader.AMOC_FILE`)
  contains the annual-mean series used in the analysis. It has a `year` column
  (2051–2150, 100 years, no gaps) and one column per run, with no missing values.
- **`amoc_timeseries_26p5N_9experiments_v5_summary.csv`** is a reference file
  with one row per run: the CESM case name (`experiment`), `label`, `mean_Sv`,
  `std_Sv` (population, ddof = 0), `yr1_Sv`, `yr100_Sv` and
  `trend_Sv_per_century` (OLS). The code does not read it. The annual file
  reproduces all of its values, the year-100 values to within 0.001 Sv.

| Case name | CSV column label | mean ± std (Sv) | year 1 → year 100 (Sv) | trend (Sv/century) |
| --- | --- | --- | --- | --- |
| `1xCO2_m03Sv` | `1x CO2, -0.3 Sv` | 25.8 ± 1.8 | 21.6 → 25.3 | +3.4 |
| `1xCO2` | `piControl` | 19.8 ± 0.7 | 21.6 → 20.3 | −0.1 |
| `1xCO2_p03Sv` | `1x CO2, +0.3 Sv` | 9.0 ± 4.5 | 21.8 → 5.2 | −13.8 |
| `2xCO2_m03Sv` | `2x CO2, -0.3 Sv` | 23.1 ± 1.0 | 22.2 → 23.3 | +1.9 |
| `2xCO2` | `2x CO2, no hosing` | 15.9 ± 1.7 | 22.6 → 14.7 | −4.6 |
| `2xCO2_p03Sv` | `2x CO2, +0.3 Sv` | 9.2 ± 5.1 | 22.5 → 5.1 | −13.5 |
| `4xCO2_m03Sv` | `4x CO2, -0.3 Sv` | 20.1 ± 0.7 | 22.4 → 20.1 | +0.9 |
| `4xCO2` | `4x CO2, no hosing` | 11.8 ± 2.9 | 22.3 → 10.3 | −7.7 |
| `4xCO2_p03Sv` | `4x CO2, +0.3 Sv` | 8.1 ± 4.1 | 21.9 → 5.9 | −10.8 |

**Transient AMOC recovery in `2xCO2_p03Sv`.** In this run AMOC jumps from
9.5 Sv (2077) to 26.8 Sv (2082) and falls back to about 6 Sv by 2086. The same
run's gridded subpolar North Atlantic temperature (45–65°N, 60–10°W) warms by
about 4 K over 2077–2081. The two datasets agree, so this looks like a genuine
model event, not a data error, and it is kept. It dominates that run's
2071–2080 and 2081–2090 decadal blocks.

**How AMOC years line up with the gridded years.** The AMOC years use the same
labels as the gridded files, and `amoc_strength_on_years` aligns the two by year.
AMOC covers the first 100 years of every perturbation run. The runs lasting 105
years get NaN AMOC for 2151–2155. The control's AMOC covers only 2051–2150 of its
1850–2150 gridded record. Regressions use complete-case deletion (see
`CLAUDE.md`), so every run contributes **100 years (900 pooled; 90 decadal
blocks)**. The best evidence that the control's AMOC years are the same years as
its gridded years is year 1: the control's 2051 value (21.63 Sv) almost equals
the first-year values of the 1×CO₂ hosing runs (21.64 and 21.75 Sv). That is
expected if 2051 is where they branched from the control. Correlating the
control's AMOC with North Atlantic temperature gives no clear answer, because
the control's AMOC variability is small.

### Scalar time series (`data/processed/`, git-ignored)

`scripts/make_scalar_timeseries.py` writes one `scalars_annual_CESM1_{case}.nc` per
run. Each holds these series on the run's year axis:

- `amoc_strength` (Sv), from `AMOC_FILE`
- `tas_global_mean` (K), the area-weighted global annual-mean temperature
- `tas_interhemispheric_diff` (K), the area-weighted NH mean minus the SH mean
- `precip_centroid_lat_20`, `precip_centroid_lat_30` (°N), the **precipitation-mass
  centroid latitude** (an ITCZ-position index) over 20°S–20°N and 30°S–30°N. It is
  the area- and precipitation-weighted mean latitude of the zonal-mean total
  precipitation (`pr`), `Σ φ·P·a / Σ P·a`. Unlike an argmax, the centroid covers
  *both* branches of a double ITCZ, so it changes continuously instead of jumping
  between them.

Area weighting uses exact zonal-band weights, `sin(edge_N) − sin(edge_S)`, which
handle the FV grid's half-width polar cells at ±90°.

## Analysis

> **Status of this section (CESM1.2 data).** Every analysis below (gridded
> regressions, EOFs, scenario prediction, ITCZ regressions) has been run on the nine
> CESM1.2 runs, and all numbers quoted are CESM1.2 results, with water fluxes in
> mm day⁻¹. The earlier CESM2 regression outputs are kept for reference in
> `data/output/old_cesm2/regression/`.

### Pooled per-grid-point regressions

`scripts/run_regressions.py` regresses a gridded annual-mean **predictand** (one
time series per grid cell) on the scalar indices `tas_global_mean` (Tglob, K),
`tas_interhemispheric_diff` (dT_NS, K) and `amoc_strength` (AMOC at 26.5°N, Sv).
It runs for **every analysis variable** (46 predictands, see
[Analysis variables](#analysis-variables-and-units-srcdata_loaderpy)). Each
predictand's field is read directly from the input files.

The years of all nine CESM1.2 runs (the 3 × 3 CO₂ × hosing matrix) are **pooled
into one fit per grid cell**, with a single common intercept and no per-run fixed
effects. The design largely decouples the two main predictors, because CO₂ sets
global temperature while hosing sets AMOC. Across the pooled decadal samples, Tglob
spans 285.0–291.6 K and AMOC 5.4–27.0 Sv, with **corr(Tglob, AMOC) = 0.01**. All
sets use one common sample: the years where every predictor is present, which are
the AMOC years 2051–2150. That is **900 annual rows (100 per run)**. The control's
gridded years before 2051 and the 105-year runs' years 2151–2155 have no AMOC and
are dropped (complete-case deletion).

Ten predictor sets are defined (one multi-panel coefficient map per set, per
predictand). **By default only sets 5 & 10 are produced**; pass `--all-sets` to any
of the regression scripts to produce all ten:

| Set | Predictors |
| --- | --- |
| 1–3 | each index alone: Tglob; dT_NS; AMOC |
| 4–6 | combinations: Tglob+dT_NS; Tglob+AMOC; Tglob+dT_NS+AMOC |
| 7 | orthogonalized, order tas→NS→AMOC: `Tglob`, `dT_NS⊥Tglob`, `AMOC⊥(Tglob,dT_NS)` |
| 8 | orthogonalized, order tas→AMOC→NS: `Tglob`, `AMOC⊥Tglob`, `dT_NS⊥(Tglob,AMOC)` |
| 9 | full quadratic (centered): Tglob, Tglob², AMOC, AMOC², dT_NS, dT_NS², Tglob·AMOC, Tglob·dT_NS, AMOC·dT_NS |
| 10 | Tglob × AMOC interaction (centered): Tglob, AMOC, Tglob·AMOC |

- **Sets 4–6** use full multiple OLS, so each map is that predictor's **partial**
  coefficient (its effect with the other predictors held fixed).
- **Sets 7–8** are Gram–Schmidt orthogonalizations (`add_orthogonalized_columns`).
  Each residual column is the index with the earlier ones regressed out. The
  columns are therefore mutually orthogonal (VIF = 1) and give a hierarchical
  decomposition, whose attribution depends on the chosen order (compare 7 vs 8).
- **Set 9** is the full quadratic response surface (`add_quadratic_columns`). The
  three base indices are **centered on their pooled means** before squares and
  products are formed. Its 9 term coefficients are mapped on a 3×3 grid.
- **Set 10** is the global-temperature × AMOC interaction model: Tglob, AMOC and
  Tglob·AMOC, reusing the centered `add_quadratic_columns` terms. Because the main
  effects are centered, each main-effect coefficient is the response at the
  *other* index's pooled mean. The CESM1.2 centering means are Tglob = 288.48 K and
  AMOC = 15.86 Sv, stored as `centering_mean_*` attributes in the NetCDF. The
  interaction coefficient is the same as in the uncentered form.

Coefficient maps (`src/output.py`) use the Equal Earth projection (longitudes
relabeled to −180…180 via `output.centered_lon`, so the grid's wrap point falls on
the map edge, not at 0°) and a diverging
colormap with symmetric bounds (white = 0). Water-related fields use `RdBu`, so
wetter or moister is blue (`regression.WET_IS_BLUE`). All other fields use
`RdBu_r`, so positive is red. The maps **stipple cells where p > 0.05**.

**Case styling.** Every figure that distinguishes cases uses one convention. Hosing
sets the color: −0.3, 0, +0.3 Sv = red, black, blue. CO₂ sets the line style (1×,
2×, 4× = solid, dashed, dotted; `output.case_line_style`) or, in scatter plots, the
marker (circle, triangle, square; `output.case_marker_style`). Scatter markers are
filled when each simulation contributes ≤ 10 points (e.g. decadal means) and open
outlines when it contributes more (e.g. ~100 annual values).

**Shared axis ranges.** Unless a plot specifies otherwise, panels of a multi-panel
figure whose axes carry the same units share the same axis range on those axes, so
they compare directly by eye: e.g. the per-simulation time-series panels (one K
range and one Sv range throughout), the per-mode PC-regression bars, and the ITCZ
panels (one latitude range). Temperature differences and anomalies (ΔT) count as a
different unit from absolute temperatures, so an absolute global-mean temperature
axis (~287 K) and an interhemispheric-difference axis (a few K) keep their own
ranges, as in `predictor_scatter.pdf` and the ITCZ scatter page.

**Shared color scales.** The same rule applies to color: map panels of one figure
that show the same quantity in the same units share one color scale (one symmetric
bound over all of them for diverging maps). This covers the nine cases of a case
grid, the EOF patterns of one field, and coefficient maps with the same units;
panels in different units (per-K vs per-Sv coefficients) keep their own scales.

**Drawing maps.** Gridded fields are drawn with `output.draw_field`, which projects
the cell corners itself. Cartopy's `pcolormesh(transform=…)` spent ~1.3 s per map
checking for cells that wrap the map edge; `draw_field` takes ~0.01 s. PDF books
are kept to **one per variable**, because matplotlib's `PdfPages` holds every page
in memory until the book is closed.

**Outputs.**

Flat in `data/output/regression/`, with a shared caveats `README.txt`:

- `<var>_coef.pdf` — one PDF book per predictand, one page per predictor set.
- `<var>_coef.nc` — the coefficient/SE/t/p/R² fields, one NetCDF group per
  predictor set (`set5`, `set10`, …; `regression.set_group`). Read a set with
  `xr.open_dataset(path, group="set10")`; set 10's group attributes carry the
  centering means for its centered (`q_`) terms.
- `scripts/plot_predictor_scatter.py`, `scripts/plot_scalar_timeseries.py` and
  `scripts/plot_tglob_vs_amoc.py` write the predictor scatter, time-series and
  AMOC-vs-Tglob plots to `data/output/regression/`.

**Collinearity.** The design is well conditioned for the default sets: VIF =
1.0002 for set 5 (Tglob, AMOC) and ≤ 1.19 for set 10. The three-index union is
more collinear (decadal VIF: Tglob 9.3, AMOC 13.2, dT_NS 21.8), because dT_NS is
largely a linear function of Tglob and AMOC. The partial coefficients of set 6
(Tglob + dT_NS + AMOC) are therefore weakly constrained.

Caveats: p-values are **nominal OLS**. Fits are validated against `statsmodels`
(agreement < 1e-6). Two consistency checks come out exact. Global-mean `tas` *is*
Tglob, so in any set containing Tglob the global mean of the `tas`-on-Tglob
coefficient is exactly 1. Every other `tas` coefficient then has an area-mean of
exactly 0, which also makes its NH and SH means equal and opposite.

### Results: CESM1.2, decadal means, sets 5 & 10

Pooled n = 90 decadal blocks (10 per run); df = 87 for set 5 and 86 for set 10.
In the table, "global" is the area-weighted global mean of the coefficient map.
"Typical SE" is the area mean of the per-cell standard error. "p < 0.05" is the
fraction of global area where the coefficient is significant (nominal). Regions
are area-weighted boxes: subpolar North Atlantic = 45–65°N, 60–10°W; Sahel =
10–20°N, 20°W–40°E. Precipitation is in mm day⁻¹ throughout.

**Set 5: predictand ~ Tglob + AMOC**

| Predictand | Coefficient | Global | Typical SE | NH / SH | Regional | p < 0.05 | R² (area mean) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tas` | Tglob (K K⁻¹) | 1.000 (exact) | 0.018 | 1.23 / 0.77 | subpolar N Atl. 1.36; 0–10°N 0.71 | 100 % | 0.97 |
| `tas` | AMOC (K Sv⁻¹) | 0 (exact) | 0.005 | +0.071 / −0.071 | subpolar N Atl. **+0.36** | 97 % | |
| `prc` | Tglob (mm day⁻¹ K⁻¹) | +0.033 | 0.008 | +0.044 / +0.022 | Sahel +0.079; 0–10°N +0.076 | 92 % | 0.62 |
| `prc` | AMOC (mm day⁻¹ Sv⁻¹) | +0.002 | 0.002 | +0.011 / −0.008 | 0–10°S −0.017; Sahel +0.018 | 83 % | |
| `pr` | Tglob (mm day⁻¹ K⁻¹) | +0.040 | 0.011 | +0.053 / +0.028 | Sahel +0.091; 0–10°N +0.076 | 91 % | 0.58 |
| `pr` | AMOC (mm day⁻¹ Sv⁻¹) | +0.001 | 0.003 | +0.012 / −0.010 | 0–10°S −0.020; Sahel +0.027 | 77 % | |

**Set 10: predictand ~ Tglob + AMOC + Tglob·AMOC (centered)**

The main effects hardly change from set 5 (for example, `tas`/AMOC in the
subpolar North Atlantic is +0.35 K Sv⁻¹), and area-mean R² rises only slightly
(tas 0.98, prc 0.64, pr 0.59). The interaction term:

| Predictand | Tglob·AMOC, global | Typical SE | Regional | p < 0.05 |
| --- | --- | --- | --- | --- |
| `tas` (K K⁻¹ Sv⁻¹) | 0 (exact) | 0.003 | subpolar N Atl. −0.022 | 48 % |
| `prc` (mm day⁻¹ K⁻¹ Sv⁻¹) | 0.000 | 0.001 | subpolar N Atl. +0.001 | 32 % |
| `pr` (mm day⁻¹ K⁻¹ Sv⁻¹) | −0.000 | 0.002 | subpolar N Atl. −0.002 | 26 % |

**Interpretation.**

- **Temperature.** With global temperature held fixed, AMOC mostly *moves heat
  around* rather than changing the global mean. A 1 Sv stronger AMOC warms the
  subpolar North Atlantic box by about 0.36 K on average. The largest effect,
  0.90 K, is at 65°N, 10°W near Iceland, and values above 0.6 K Sv⁻¹ extend from
  52°N to 79°N, reaching northeast toward the Barents Sea. It warms the NH by
  0.07 K on average and cools the SH by the same amount. Per kelvin of global
  warming, the NH warms 1.23 K and the SH 0.77 K. North of 70°N the coefficient
  averages 2.7 K K⁻¹, peaking at 3.6 (polar amplification).
- **Interaction.** The negative `tas` interaction in the subpolar North Atlantic
  means AMOC's local warming effect weakens as the climate warms, by about 6 % of
  its value per K of global warming (−0.022 / 0.354).
- **Precipitation.** A stronger AMOC shifts tropical rain northward. Rainfall
  increases north of the equator and over the Sahel (+0.027 mm day⁻¹ Sv⁻¹ for
  `pr`) and decreases in the 0–10°S band (−0.020 mm day⁻¹ Sv⁻¹). This is the ITCZ
  moving toward the hemisphere that AMOC warms. Global-mean total precipitation
  increases by 0.040 mm day⁻¹ K⁻¹, about 1.4 % K⁻¹ of the 2.88 mm day⁻¹ control
  mean. The Tglob maps show the familiar tropical wet-get-wetter pattern.
- **Interaction significance.** The Tglob·AMOC interaction is significant (nominal
  p < 0.05) over a quarter to half of the globe. Its magnitude is small, however,
  so the additive set 5 already captures most of the response.

### Decadal means

Every regression analysis (gridded, EOF, ITCZ, scenarios) uses **decadal means**,
to characterize variability slower than interannual. The low-pass is
**non-overlapping 10-year block means** (`regression.DECADAL_BLOCK`,
`data_loader.block_average_on_years`). It is applied **per run, per
contiguous segment, to both the predictors and the predictand** inside
`regression.build_pooled(block=regression.DECADAL_BLOCK)`, *before* pooling. The same filter therefore
acts on the dependent and independent variables, and every downstream step (the
orthogonalized and quadratic columns, the grid OLS, the EOFs) inherits it. Blocks
never span a run boundary or a gap within a run. Each block's timestamp is its
midpoint year. For CESM1.2, each run's AMOC years 2051–2150 form one contiguous
segment of exactly ten blocks.

Block averaging is a **decimation**, not a running mean. It collapses each decade
to one roughly independent sample (pooled **n = 90**, 10 per run), so the nominal
OLS degrees of freedom are far more honest than with annual data (successive
decades of a run still drift together, so p-values remain somewhat optimistic). Quadratic and product terms are
formed from the *filtered* bases (filter, then square), which gives the genuinely
low-frequency response surface.

### EOF / principal-component analysis (additive path)

`scripts/run_eof_regressions.py` is an **additive** companion to the direct
per-grid-point maps (it does not replace `run_regressions.py`). It decomposes each
gridded field into empirical orthogonal functions (EOFs) and examines how the
leading principal-component (PC) time series — the EOF *weightings over time* —
behave and relate to the predictors. It works on **decadal means** only (10-year
block means per run, pooled n = 90), and like the other per-variable scripts takes
`--variables`.

Method (`src/eof.py`):

- **Anomalies** are taken about each cell's **grand temporal mean over all pooled
  samples** (not per-run means) — this retains the between-run forced variability
  the predictors are meant to explain.
- **Area-weighted covariance EOF:** anomalies are multiplied by √(zonal-band area
  weight) before an economy SVD and the patterns divided by it afterward. No
  per-cell standard-deviation normalization.
- **Normalization:** each EOF is a dimensionless pattern scaled to an
  **area-weighted RMS of 1** over the grid, so its values (order 1) do not depend
  on model resolution. The PCs carry the field's units: |PCₖ(t)| is the
  area-weighted RMS anomaly (e.g. K) that mode k contributes at time t, and
  PCₖ × EOFₖ is that mode's anomaly field. The pattern maps of one field share a
  color scale; the PC-regression bars are standardized and so dimensionless.
- **Truncation:** two rules combined, the more restrictive winning — keep leading
  modes until cumulative variance reaches **≥ 95 %**, but never keep a mode that
  individually explains **< 1 %** of variance (the per-mode floor drops the long
  low-variance noise tail). Counts are reported per field. The patterns page maps
  up to the leading 9 modes.
- **PC regression:** the retained PCs are regressed on the selected predictor sets
  (5 & 10 by default; all ten with `--all-sets`, including the orthogonalized,
  quadratic, and interaction columns) with an intercept; the PC-space coefficients
  (coef/SE/t/p) are saved.

Outputs are flat in `data/output/eof/`, two files per variable plus a shared
caveats `README.txt`:

- `<var>_pc.nc` — OLS of the PCs on every predictor set fit, in raw PC units, one
  NetCDF group per set (`set5`, `set10`, …): `coef`, `se`, `tstat`, `pvalue` on
  `(param, mode)` and `r2` on `mode`.
- `<var>_pc.pdf`, in page order:
  1. the leading EOF spatial patterns + a variance scree;
  2. the PC-on-scalar regression — the EOF analog of the 2D coefficient maps, with
     the discrete EOF-mode index replacing the (lat, lon) grid. One **page per
     predictor set**; each page has one panel per retained EOF mode, with a bar per
     predictor showing the **standardized** coefficient β·σ(xⱼ)/σ(PCₘ) (z-scoring
     predictors and the PC, so bars are comparable across modes — raw coefficients
     scale with each PC's amplitude) and a ±SE whisker. Non-significant bars
     (p > 0.05) are faded; the panel title reports R² and the mode's variance share.
     t/p are scale-invariant and match `<var>_pc.nc`;
  3. one **page per richer set** among the 3-index set 6, the quadratic set 9, and
     the interaction set 10 that was actually fit (only set 10 by default; all three
     with `--all-sets`): the fitted X·β overlaid on the actual PC over time, one
     panel per simulation — a direct view of how well the scalars predict each EOF
     weighting.

**Results (CESM1.2, decadal means, n = 90).**

| Field | Modes kept (cum. var.) | Leading modes (% var.) | Set 5 R² of PC1, PC2 |
| --- | --- | --- | --- |
| `tas` | 2 (97.9 %) | 81.9, 15.9 | 1.00, 0.95 |
| `prc` | 7 (87.0 %) | 47.4, 21.6, 9.5, 4.2, 1.8, 1.4 | 0.98, 0.84 |
| `pr` | 7 (84.5 %) | 46.5, 19.3, 9.7, 4.5, 1.8, 1.5 | 0.98, 0.79 |

- **`tas` is almost exactly two-dimensional.** The 95 % rule binds at two modes.
  PC1 is essentially Tglob (set 5 t = +197 for Tglob, +39 for AMOC). PC2 is the
  AMOC pattern (t = −38 for AMOC, +8.5 for Tglob).
- **Precipitation is not low-rank**, so the 1 % floor binds at seven modes. As for
  `tas`, PC1 is the warming mode (t_Tglob ≈ +60) and PC2 the AMOC mode
  (t_AMOC ≈ +18 to +21). Together they explain two thirds of the variance. The
  scalar predictors explain almost none of PCs 3–7 (R² ≤ 0.09), which are
  unforced decadal variability.

The spatial **fingerprint** maps (Σₖ βₖ·EOFₖ, the PC regression projected back to
the grid) are intentionally **not** generated — the EOFs and PC weightings are the
wanted deliverables. The capability remains in `eof.reconstruct_fingerprint`
(verified: with all modes retained it reproduces the direct field regression's
coefficient *and* p-value to Δcoef ~ 1e-10, Δp ~ 1e-8) should maps be wanted later.

### Scenario prediction: warming × AMOC decline

`scripts/predict_scenarios.py` uses the **decadal** set 5 (Tglob + AMOC) and set 10
(Tglob + AMOC + Tglob·AMOC) coefficient maps to predict field changes for 3 K of
global warming relative to the 1×CO₂ control, with and without an AMOC decline from
20 to 6 Sv. It addresses: *where does AMOC decline exacerbate the response to warming,
and where does it ameliorate it?* Four states form a 2 × 2 factorial:

| state | Tglob (K) | AMOC (Sv) |
| --- | --- | --- |
| reference | T0 = 286.91 (1×CO₂ control, mean over 2051–2150) | 20 |
| weak | T0 | 6 |
| warm | T0 + 3 | 20 |
| warm-weak | T0 + 3 | 6 |

Each output is one page with a **3 × 3 grid** of maps. The corners are the four
states' changes from the reference; each edge is the difference of its two
neighbouring corners:

| | 20 Sv | AMOC 20 → 6 Sv effect | 6 Sv |
| --- | --- | --- | --- |
| **+0 K** | reference (≡ 0) | weak − reference | weak − reference |
| **warming effect** | warm − reference | *interaction* (set 10) | warm-weak − weak |
| **+3 K** | warm − reference | warm-weak − warm | warm-weak − reference |

The centre is the **interaction**: the AMOC effect at +3 K minus that at +0 K
(equivalently, the warming effect at 6 Sv minus that at 20 Sv). It is identically
zero for the additive set 5, so set 5 leaves it blank. For `tas` in set 10 it
weakens the AMOC-decline cooling of the subpolar North Atlantic by about 0.9 K
(−0.022 K K⁻¹ Sv⁻¹ × 3 K × −14 Sv) out of about 5 K.

Global-mean tas is fixed along each row, so the AMOC effects are pure spatial
redistributions of tas; the global means of prc and pr can still shift. All four
states lie inside the sampled predictor space (between the ~20 Sv runs and the
+0.3 Sv runs at ~5.5 Sv), so the predictions are interpolations. The predicted change
between two states is `coef · (predictor(X) − predictor(R))`; the intercept cancels,
and set 10 evaluates its centered columns and interaction with the fit's centering
means.

Outputs: `data/output/scenarios/<predictand>_scenarios.pdf`, one page per set
(5, then 10). All panels for a predictand, **in both sets**, share one symmetric color scale
(99th percentile of |change|), so set 5 and set 10 compare directly. Each panel title
gives its area-weighted global mean.

**Results (set 5; set 10 differs by < 0.003 mm day⁻¹ in the global means).**
Regional means use the boxes of the regression results above:

| Region | tas: warming only (K) | tas: AMOC effect (K) | pr: warming only (mm day⁻¹) | pr: AMOC effect (mm day⁻¹) |
| --- | --- | --- | --- | --- |
| global | +3.00 | 0.00 (exact) | +0.121 | −0.015 |
| subpolar N Atlantic | +4.09 | **−5.09** | +0.032 | −0.250 |
| Sahel | +2.82 | +0.66 | +0.274 | **−0.372** |
| 0–10°N | +2.14 | +0.65 | +0.227 | −0.085 |
| 0–10°S | +2.14 | +0.69 | +0.048 | +0.285 |

- **Temperature.** The AMOC decline more than cancels the local warming in the
  subpolar North Atlantic. The box mean goes from +4.1 K to −1.0 K, and the
  strongest cell cools by 12.6 K relative to the warming-only case. Because the
  global mean is held at +3 K, most of the rest of the world warms somewhat more,
  including the tropics (about +0.7 K).
- **Precipitation.** The decline shifts tropical rain south. It reverses the
  Sahel's warming-driven wetting (+0.27 → −0.10 mm day⁻¹), wets 0–10°S, and dries
  the subpolar North Atlantic. Globally it takes back 12 % of the warming-driven
  increase in total precipitation (−0.015 of +0.121 mm day⁻¹) and 23 % of the
  increase in convective precipitation (−0.022 of +0.098 mm day⁻¹).

### ITCZ-position regressions (scalar response)

`scripts/run_itcz_regressions.py` regresses the **scalar** ITCZ index — the
precipitation-mass centroid latitude, for two tropical bands
(`precip_centroid_lat_20`, `precip_centroid_lat_30`) — on the same scalar indices
(Tglob, dT_NS, AMOC), using the same predictor sets (5 & 10 by default, all ten with
`--all-sets`) and the same decadal-mean pooling as the gridded regressions. Because the response is a single series per simulation-year
(not a gridded field or PCs), it uses `regression.build_pooled_scalar` and
`regression.fit_scalar_ols` (a 1-D OLS with the same normal-equations math as the
gridded fit, validated against `statsmodels`, plus 95 % confidence intervals). The
pooled common sample is the same AMOC-complete 900 years (90 decadal blocks).
Outputs are flat in `data/output/itcz/`, one pair per band (`band20`, `band30`),
with a shared caveats `README.txt`:

- `<band>_coef_table.csv` — coef, SE, t, p, 95 % CI per parameter, with R² and n,
  for every set (the scalar analog of the gridded coefficient maps).
- `<band>_itcz.pdf` (`scripts/plot_itcz_regressions.py`), in page order:
  1. the centroid latitude per simulation, annual with the decadal means overlaid;
  2. ITCZ latitude vs each single predictor with the OLS line, 95 % CI band, and
     slope ± SE / R² / p annotated;
  3. predicted vs observed centroid latitude for the multi-predictor sets (5 & 10
     by default; 5, 6, 10 with `--all-sets`), with the 1:1 line and R² (shows how
     well the *joint* regression reproduces the ITCZ across runs);
  4. partial-slope (coef ± SE) bar charts for the same sets, blue/red by sign and
     hatched where not significant.

**Results (CESM1.2; total-`pr` centroid).** R² by predictor set, band20 /
band30:

| Set | Predictors | Annual (n = 900) | Decadal (n = 90) |
| --- | --- | --- | --- |
| 1 | Tglob | 0.03 / 0.06 | 0.05 / 0.07 |
| 2 | dT_NS | 0.54 / 0.66 | 0.75 / 0.79 |
| 3 | AMOC | 0.55 / 0.65 | 0.78 / 0.80 |
| 5 | Tglob + AMOC | 0.58 / 0.70 | 0.83 / 0.86 |
| 6 | Tglob + dT_NS + AMOC | 0.67 / 0.78 | 0.92 / 0.94 |
| 9 | full quadratic | 0.70 / 0.81 | 0.96 / 0.97 |
| 10 | Tglob × AMOC | 0.58 / 0.70 | 0.83 / 0.86 |

- **AMOC and the interhemispheric temperature difference are about equally strong
  single predictors** of the ITCZ position (R² within 0.03 of each other). Global
  temperature alone explains almost nothing (R² ≤ 0.07).
- **Set 5 slopes (decadal):** AMOC +0.045 ± 0.002 ° Sv⁻¹ (band20) and
  +0.052 ± 0.002 ° Sv⁻¹ (band30); Tglob +0.042 ± 0.009 ° K⁻¹ and
  +0.057 ± 0.009 ° K⁻¹. A stronger AMOC moves the ITCZ north, toward the hemisphere
  it warms.
- **Scenario scale:** an AMOC decline from 20 to 6 Sv moves the centroid about
  0.6° (band20) to 0.7° (band30) south. That is four to five times the northward
  shift from 3 K of warming (+0.13° / +0.17°).
- **No interaction:** the set-10 Tglob·AMOC term is not significant (p = 0.62 /
  0.24), so R² is unchanged from set 5.
- **Total vs convective precipitation.** The centroid used to be computed from
  convective `prc`, a holdover from the CESM2 data, where total `pr` was missing
  for some runs. With `pr` the fits are slightly weaker (set 5 decadal R² 0.83 / 0.86, against
  0.83 / 0.87 with `prc`), and the AMOC slope in the wider band is smaller (0.052,
  against 0.059 ° Sv⁻¹).

## Reproducing the results

After placing the input files in `data/input/` (see [Data](#data)) and installing
dependencies, run, in order:

```bash
python scripts/make_scalar_timeseries.py   # data/processed/scalars_annual_CESM1_*.nc
python scripts/run_regressions.py          # data/output/regression/<var>_coef.{pdf,nc}
python scripts/plot_predictor_scatter.py   # data/output/regression/predictor_scatter.pdf
python scripts/plot_scalar_timeseries.py   # data/output/regression/predictor_timeseries.pdf
python scripts/plot_tglob_vs_amoc.py       # data/output/regression/tglob_vs_amoc.pdf (AMOC vs Tglob, 9 cases)
python scripts/plot_case_grid_book.py      # data/output/case_grid/<var>_2101-2150.pdf (3x3 case maps, one book per variable)
python scripts/run_eof_regressions.py      # data/output/eof/<var>_pc.{pdf,nc}
python scripts/predict_scenarios.py        # data/output/scenarios/<var>_scenarios.pdf
python scripts/run_itcz_regressions.py     # data/output/itcz/<band>_coef_table.csv
python scripts/plot_itcz_regressions.py    # data/output/itcz/<band>_itcz.pdf
```

The four set-fitting scripts (`run_regressions.py`, `run_eof_regressions.py`,
`run_itcz_regressions.py`, `plot_itcz_regressions.py`) default to **only sets 5 & 10**
on decadal means; add `--all-sets` to produce all ten sets. `predict_scenarios.py`
uses sets 5 & 10, so it needs only the default run.

**Variable sets (`--variables`).** Every script that makes per-variable output --
`run_regressions.py`, `plot_case_grid_book.py`, `run_eof_regressions.py`,
`predict_scenarios.py` -- takes `--variables` with one or
more named sets (`data_loader.VARIABLE_SETS`) and/or variable names, default `all`.
`run_eof_regressions.py` and `predict_scenarios.py` read the matching `run_regressions.py` output, so run that with the same set first.
The ITCZ scripts work on the scalar precipitation-centroid latitude, not per-variable
fields, so they have no `--variables`.

| Set | Variables | `run_regressions.py` / `plot_case_grid_book.py` time |
| --- | --- | --- |
| `minimal` | `tas`, `pr` | ~20 s / ~10 s |
| `key` | 13: `tas`, `diurnal_temperature_range`, `pr`, `prc`, `pr_minus_evap`, `prsn`, `RHREFHT`, `TMQ`, `CLDTOT`, `cloud_radiative_effect`, `toa_net_down`, `sfc_net_energy_down`, `planetary_albedo` | ~2 min / ~40 s |
| `all` (default) | all 46 | ~6 min / ~2 min |

```bash
python scripts/run_regressions.py --variables minimal
python scripts/plot_case_grid_book.py --variables key SHFLX   # sets and names mix
```

Every output is per variable, so a subset run simply rewrites that subset's files
and leaves the others alone. A full run of everything on all 46
variables takes roughly 20 minutes, mostly figure rendering (the fits take ~1 s
per variable), with peak memory under 2 GB.

Each script is a thin wrapper over `src/` and prints what it writes. All outputs
land under `data/` (git-ignored) and are fully regenerable from the inputs.

Utility: `python scripts/split_pdf_into_pages.py <file.pdf>` splits a multi-page
figure PDF into one file per page (`<file>_page01.pdf`, `_page02.pdf`, … in the
same directory) for pasting individual panels into LaTeX. Add `--png` (optionally
`--dpi N`, default 300) to rasterize the pages to high-resolution PNGs instead —
much lighter for a LaTeX engine to load than the vector PDFs.

## Status

Preprocessing (`scripts/make_scalar_timeseries.py`), pooled per-grid-point
regression analysis (`scripts/run_regressions.py`, sets 1–10 for all 46 analysis
variables), predictor scatter
and time series (`scripts/plot_predictor_scatter.py`,
`scripts/plot_scalar_timeseries.py`), the additive EOF / principal-component path
(`scripts/run_eof_regressions.py`, built on `src/eof.py`), the decadal
scenario-prediction maps (`scripts/predict_scenarios.py`), and the ITCZ-centroid
scalar regressions (`scripts/run_itcz_regressions.py`,
`scripts/plot_itcz_regressions.py`), all built on `src/data_loader.py`,
`src/regression.py`, and `src/output.py`. The regression, EOF, and ITCZ analyses all
use **decadal means** (10-year block means), and only sets 5 & 10 are fit unless
`--all-sets` is given.

## Legacy CESM1 code

`src/amoc_cesm/` and its scripts come from the former `amoc-cesm` repository
(CESM1 CO₂ × hosing factorial); the data they target is outmoded. See
[`src/amoc_cesm/README.md`](src/amoc_cesm/README.md).
