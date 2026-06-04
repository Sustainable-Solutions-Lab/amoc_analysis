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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies:

- **numpy**, **pandas** — numerical arrays and tabular data
- **xarray**, **netCDF4** — reading CESM2 NetCDF output
- **scipy**, **statsmodels** — linear regression and statistics
  (`statsmodels` for OLS with full inference; `scipy.stats` for quick fits)
- **matplotlib** — figures

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
commit to git and are not tracked in this repository.

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

## Status

Project scaffolding. Analysis modules to follow.
