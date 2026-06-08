# Claude Code Style Guide for AMOC Analysis

## Project Goal
This project performs statistical analysis — primarily linear regression — on
climate model output from the **CESM2** model, focused on the Atlantic
Meridional Overturning Circulation (AMOC) and its relationships to other climate
variables and forcings.

Typical tasks:

1. **Extract diagnostics** from CESM2 NetCDF output (AMOC strength, North
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
  - `data_loader.py` — read CESM2 NetCDF, select variables, apply averaging
  - `regression.py` — OLS / linear-regression fitting and inference
  - `output.py` — results tables and plots
- Scripts belong in `scripts/` and should be thin wrappers around src modules.

### Protected Directories
- **Never modify files in `./data/input/`** — this directory contains CESM2
  reference data that must remain unchanged.
- Generated results, tables, and figures go in `./data/output/` (git-ignored).

### Naming Conventions
- Descriptive names preferred — long, clear names are better than short,
  ambiguous ones.
- Use consistent names for fitted quantities: `slope`, `intercept`, `stderr`,
  `pvalue`, `r_squared`, `conf_int`.
- Keep CESM2 variable names recognizable (e.g. `MOC`, `TEMP`, `SALT`, `SHF`) when
  mapping to analysis variables, and document the mapping.

### Function Design
- Functions should assume valid inputs and focus on their core
  mathematical/logical purpose.
- Let Python's natural error messages guide debugging rather than custom error
  handling.
- Clean fail-fast approach — if required arguments are not supplied, the code
  should fail immediately with a clear error.

## Version Control
- Do not concern yourself with committing or pushing to the remote repository.
  The user manages git commits and pushes; do not offer to commit/push, ask
  whether to, or run `git commit`/`git push` unless explicitly instructed to in a
  specific request.

## Plotting Conventions
- **File format**: Save figures as PDF. Use
  `fig.savefig(path, dpi=300, bbox_inches='tight')`.
- **Regression plots**: Show the data (scatter or line), the fitted line, and a
  shaded confidence band; report slope ± standard error and the relevant
  statistic (R², p-value) in the legend or annotation.
- **Diverging colormaps**: For difference/anomaly maps or any plot using a
  diverging colormap (e.g. `RdBu_r` where white is the midpoint), always use
  symmetric bounds with equal magnitude and opposite sign so white represents
  zero. Example: if data ranges from -0.03 to 0.05, use bounds (-0.05, 0.05),
  not the raw data range.
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
  For example, the combined historical+ssp585 AMOC series is missing 1950–2000;
  a regression using it simply omits those years. State in the output how many
  years were used after dropping incomplete cases.
