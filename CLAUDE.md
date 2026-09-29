# Claude Code Style Guide for AMOC Analysis

## Project Goal
This project performs statistical analysis — primarily linear regression — on
climate model output from the **CESM1** model (NAHosMIP 3×3 CO₂ × hosing runs), focused on the Atlantic
Meridional Overturning Circulation (AMOC) and its relationships to other climate
variables and forcings.

Typical tasks:

1. **Extract diagnostics** from CESM1 NetCDF output (AMOC strength, North
   Atlantic temperature/salinity, surface fluxes, etc.)
2. **Regress** AMOC and related quantities against other variables and time to
   characterize trends and sensitivities.
3. **Quantify uncertainty** in fitted slopes and intercepts (standard errors,
   confidence intervals, p-values).
4. **Visualize** relationships and trends with publication-quality figures.

When implementing new analysis features:
- Use established statistical routines (`statsmodels`, `scipy.stats`) rather than
  hand-rolling regression math, so inference (SEs, CIs, p-values) comes for free.
- Keep variable extraction (NetCDF → arrays/DataFrames) separate from the
  statistics so each step is independently testable.
- Document units, time ranges, and any spatial/temporal averaging applied.
- Be explicit about the regression model: dependent variable, predictors, and
  whether time or other covariates are included.

## Coding Philosophy
This project prioritizes elegant, fail-fast code that surfaces errors quickly
rather than hiding them.

### Root Cause Analysis
- Always investigate and understand the root cause of problems before
  implementing solutions.
- Avoid band-aid fixes that mask symptoms without addressing underlying issues.
- When unexpected behavior occurs, trace it back to its source rather than
  applying quick patches.
- Document the reasoning behind fixes to prevent similar issues.

## Core Style Requirements

### Error Handling
- No input validation on function parameters (except for command-line interfaces).
- No defensive programming — let exceptions bubble up naturally.
- Fail fast — prefer code that crashes immediately on invalid inputs rather than
  continuing with bad data.
- No try-catch blocks unless absolutely necessary for program logic (not error
  suppression).
- Assume complete data — do not check for missing data fields. If required data
  is missing, let the code fail with natural Python errors. (Note: genuine
  missing values in climate fields, e.g. land mask `NaN`s, are data — handle
  them explicitly with documented masking, not defensive guards.)

### Code Elegance
- Minimize conditional statements — prefer functional approaches, mathematical
  expressions, and numpy/xarray vectorization.
- Favor mathematical clarity over defensive checks.
- Use numpy/xarray operations instead of loops and conditionals where possible.
- Compute once, use many times — move invariant calculations outside loops and
  create centralized helper functions.
- No backward compatibility — do not add conditional logic to support deprecated
  field names or old configurations. Update all code to use current conventions.
- Use standard packages — prefer established numerical methods from scipy, numpy,
  statsmodels, and xarray rather than implementing custom numerical algorithms.

### Code Organization
- All imports at the top of the file — no imports inside functions or scattered
  throughout the code.
- Source code belongs in `src/` with clear module responsibilities, e.g.:
  - `data_loader.py` — read CESM1 NetCDF, select variables, apply averaging
  - `regression.py` — OLS / linear-regression fitting and inference
  - `output.py` — results tables and plots
- Scripts belong in `scripts/` and should be thin wrappers around src modules.

### Protected Directories
- **Never modify files in `./data/input/`** — this directory contains CESM1
  reference data that must remain unchanged.
- Generated results, tables, and figures go in `./data/output/` (git-ignored).

### Naming Conventions
- Descriptive names preferred — long, clear names are better than short,
  ambiguous ones.
- Use consistent names for fitted quantities: `slope`, `intercept`, `stderr`,
  `pvalue`, `r_squared`, `conf_int`.
- Keep CAM variable names recognizable (e.g. `TREFHT`, `PRECT`, `QFLX`) when
  mapping to analysis variables, and document the mapping.

### Function Design
- Functions should assume valid inputs and focus on their core
  mathematical/logical purpose.
- Let Python's natural error messages guide debugging rather than custom error
  handling.
- Clean fail-fast approach — if required arguments are not supplied, the code
  should fail immediately with a clear error.

## Legacy CESM1 code (`src/amoc_cesm/`)
- `src/amoc_cesm/` and its scripts (`scripts/extract_postproc.py`,
  `inventory.py`, `make_pair_compare_book.py`, `make_sss_maps.py`,
  `make_steady_state_book.py`, `regrid_salt.py`) were merged in from the former
  `amoc-cesm` repository (CESM1 CO₂ × hosing factorial). The data they were
  written against is outmoded; only the CESM1 NAHosMIP data in `data/input/` used
  by the rest of this repository is current. See `src/amoc_cesm/README.md` for its original docs.

## Version Control
- Do not concern yourself with committing or pushing to the remote repository.
  The user manages git commits and pushes; do not offer to commit/push, ask
  whether to, or run `git commit`/`git push` unless explicitly instructed to in a
  specific request.

## Plotting Conventions
- **File format**: Save figures as PDF. Use
  `fig.savefig(path, dpi=300, bbox_inches='tight')`. Build multi-page PDFs with
  `output.PdfBook`, never matplotlib's `PdfPages`: `PdfPages` holds every page in
  memory until it is closed (a 46-page map book reached ~13 GB).
- **Regression plots**: Show the data (scatter or line), the fitted line, and a
  shaded confidence band; report slope ± standard error and the relevant
  statistic (R², p-value) in the legend or annotation.
- **Diverging colormaps**: For difference/anomaly maps or any plot using a
  diverging colormap (e.g. `RdBu_r` where white is the midpoint), always use
  symmetric bounds with equal magnitude and opposite sign so white represents
  zero. Example: if data ranges from -0.03 to 0.05, use bounds (-0.05, 0.05),
  not the raw data range.
- **Map projection**: Use the Equal Earth projection for all maps by default
  (following UN guidance): `ccrs.EqualEarth()`, available as
  `output.PROJECTION`. Draw gridded data with `transform=ccrs.PlateCarree()`
  (`output.DATA_CRS`), and draw coastlines with `output.draw_coastlines(ax)`
  (Natural Earth 110m coastline simplified to 1°, keeping vector PDFs small)
  rather than `ax.coastlines()`. Example:
  ```python
  import matplotlib.pyplot as plt

  from output import PROJECTION, draw_coastlines

  fig = plt.figure()
  ax = plt.axes(projection=PROJECTION)
  draw_coastlines(ax)
  ```
- **Multi-case map figures**: Any figure showing maps for several cases uses a
  CO₂ × hosing grid, available as `data_loader.CASE_GRID`. Rows are CO₂ level
  1, 2, 4× (top to bottom); columns are hosing `m03Sv` (−0.3 Sv), none (0 Sv),
  `p03Sv` (+0.3 Sv) (left to right). Case names are `[124]xCO2` (no hosing) and
  `[124]xCO2_[pm]03Sv`.
- **Line plots of cases**: CO₂ level sets the line style and hosing sets the
  color: 1×, 2×, 4×CO₂ = solid, dashed, dotted; −0.3, 0, +0.3 Sv = red, black,
  blue. Use `output.case_line_style(case)` (constants `output.CO2_LINESTYLE`,
  `output.HOSING_COLOR`).
- **Scatter plots of cases**: hosing sets the color exactly as for lines
  (−0.3, 0, +0.3 Sv = red, black, blue); CO₂ level sets the filled marker shape:
  1×, 2×, 4×CO₂ = circle, triangle, square (the triangle, least prominent, marks the
  least-emphasized 2×CO₂ level). Use `output.case_marker_style(case)`
  (constant `output.CO2_MARKER`). Every figure that distinguishes cases uses these
  two conventions — never a generic color cycle.
- **Units and labels**: Always label axes with variable name and units; state the
  time range and any spatial averaging in the title or caption.

## Mathematical Conventions
- State the regression model explicitly (dependent variable, predictors,
  treatment of time).
- Report uncertainty on every fitted parameter (standard error or confidence
  interval), not just the point estimate.
- Be explicit about units and any normalization or anomaly referencing applied.
- **Missing data in regressions**: If the dependent variable or any predictor
  has missing data (`NaN`) for some years, drop those years from the fit
  (listwise / complete-case deletion) rather than imputing or interpolating.
  For example, AMOC covers only 2051–2150, so the control's earlier years and the
  105-year hosing runs' 2151–2155 are omitted; a regression using AMOC simply uses
  each run's AMOC-present years. State in the output how
  many years were used after dropping incomplete cases.
