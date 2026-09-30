"""EOF / principal-component analysis of pooled decadal-mean gridded fields.

For each predictand (``--variables``, default all): compute area-weighted
covariance EOFs of the grand-mean anomalies of the pooled decadal-mean sample
(10-year block means per run), plot the leading EOF patterns, and regress the
retained PCs on each selected predictor set. Complements scripts/run_regressions.py.
By default only sets 5 & 10 are run (pass ``--all-sets`` for all ten).

Outputs, flat in ``data/output/eof/``:

- ``<var>_pc.pdf`` -- EOF patterns + scree, then one PC-regression page per
  predictor set, then one fitted-vs-actual PC page per richer set (6, 9, 10) fit.
- ``<var>_pc.nc`` -- the PC-space regression of every set fit, one NetCDF group
  per set (``set5``, ``set10``, ...; ``regression.set_group``): ``coef``, ``se``,
  ``tstat``, ``pvalue`` on ``(param, mode)`` and ``r2`` on ``mode``.
- ``README.txt`` -- the caveats below (shared by all variables).

    python scripts/run_eof_regressions.py [--all-sets] [--variables minimal | tas pr ...]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from matplotlib.backends.backend_pdf import PdfPages

import data_loader as dl
import eof
import regression as reg
from output import (
    plot_eof_patterns,
    plot_pc_prediction,
    plot_pc_regression,
)

OUT_DIR = os.path.join(dl._REPO_ROOT, "data", "output", "eof")
# Retain leading EOFs until cumulative variance reaches VARIANCE_THRESHOLD, but
# never keep a mode explaining less than MIN_VARIANCE_FRACTION (drops the noise
# tail; the more restrictive rule wins).
VARIANCE_THRESHOLD = 0.95
MIN_VARIANCE_FRACTION = 0.01


CAVEATS = """EOF / principal-component analysis of decadal-mean fields.

- Sample: each run's AMOC-present years (2051-2150 for all nine CO2 x hosing runs)
  are reduced to non-overlapping 10-year block means, and the decadal means of all
  runs are pooled (9 runs x 10 decades = 90 samples). The EOFs are area-weighted
  covariance EOFs of the anomalies from the pooled grand mean.
- Mode truncation (the more restrictive rule wins): keep modes until cumulative
  variance >= 95%, but never keep a mode explaining < 1% of variance. For noisy
  fields the 1% floor stops first, so fewer than 95% of the variance is retained
  (e.g. pr: 7 modes, ~85%); the retained count and variance are printed at run
  time and shown in the scree panel.
- Files (flat in this directory, one pair per variable):
    * <var>_pc.pdf -- page 1: the leading EOF patterns (at most 9 mapped) and a
      scree of variance explained. Then one page per predictor set: the
      PC-on-scalar regression, one panel per retained mode, a bar per predictor
      showing the STANDARDIZED coefficient beta*sigma(x)/sigma(PC) with +/-SE;
      faded bars are not significant (p > 0.05); panel titles give R^2 and % var.
      Standardizing makes bars comparable across modes (raw coefs scale with each
      PC's amplitude). Then one page per richer set among (6, 9, 10) that was fit
      (only set 10 by default; all three with --all-sets): fitted (X*beta) vs
      actual PC over time per run.
    * <var>_pc.nc -- the raw (unstandardized) PC-space OLS of every set fit, one
      NetCDF group per set (set5, set10, ...): coef/se/tstat/pvalue on
      (param, mode) and r2 on mode. Read with xr.open_dataset(path, group="set5").
      t and p are scale-invariant, so they match the standardized bars.
- Normalization: each EOF is a dimensionless pattern with area-weighted RMS = 1
  over the grid (independent of resolution); the PCs carry the field's units, so
  |PC| is the area-weighted RMS anomaly the mode contributes and PC x EOF is the
  mode's anomaly field. Raw coefs are therefore in [field units] / [predictor
  units] of that RMS amplitude.
- The spatial fingerprint maps (Sum_k beta_k * EOF_k) are intentionally NOT
  produced; scripts/run_regressions.py maps the per-grid-point regressions.
- p-values are nominal OLS on the 90 decadal samples. Decadal means within a run
  are still autocorrelated (the runs drift toward equilibrium), so they are
  somewhat optimistic.
"""


def run_for_predictand(name, all_sets):
    predictand = reg.PREDICTANDS[name]
    predictors, response = reg.build_pooled(predictand=predictand, block=reg.DECADAL_BLOCK)
    predictors = reg.add_orthogonalized_columns(predictors)
    predictors = reg.add_quadratic_columns(predictors)

    eof_ds = eof.compute_eofs(
        response, variance_threshold=VARIANCE_THRESHOLD,
        min_variance_fraction=MIN_VARIANCE_FRACTION,
    )
    var_frac = eof_ds["variance_fraction"].values
    print(f"\n[{name}] pooled n={predictors.sizes['sample']}; {eof_ds.attrs['n_modes']} "
          f"EOF modes retained (cum var {eof_ds.attrs['total_variance_fraction'] * 100:.1f}%); "
          f"leading % = {(var_frac[:6] * 100).round(1)}")

    fits = {set_def["number"]: (eof.fit_pcs(predictors[set_def["predictors"]], eof_ds["pcs"]),
                                set_def["predictors"])
            for set_def in reg.select_predictor_sets(all_sets)}

    pdf_path = os.path.join(OUT_DIR, f"{name}_pc.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_eof_patterns(
            eof_ds, f"EOF patterns: {name} (decadal-mean anomalies)",
            cmap=predictand["cmap"], pdf=pdf,
        )
        for num, (pc_fit, names) in fits.items():
            plabels = ", ".join(reg.PREDICTORS[p]["label"] for p in names)
            plot_pc_regression(
                pc_fit, predictors[names], eof_ds["pcs"],
                f"PC regression: {name} (decadal means) — set {num}: {plabels}",
                variance_fraction=var_frac, pdf=pdf,
            )
        # Fitted vs actual PCs for the richer sets that were fit (full 3-index,
        # quadratic, Tglob×AMOC interaction): only set 10 by default.
        for num in (n for n in (6, 9, 10) if n in fits):
            pc_fit, names = fits[num]
            plot_pc_prediction(
                eof_ds, pc_fit, predictors[names],
                f"PC fitted vs actual: {name} (decadal means) — set {num}",
                predictand["units"], pdf=pdf,
            )
    nc_path = os.path.join(OUT_DIR, f"{name}_pc.nc")
    reg.write_set_fits(nc_path, {
        num: pc_fit[["coef", "se", "tstat", "pvalue", "r2"]].assign_attrs(predictors=", ".join(names))
        for num, (pc_fit, names) in fits.items()})
    print(f"[{name}] wrote {pdf_path}\n[{name}] wrote {nc_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sets", action="store_true",
        help="regress on all ten predictor sets (default: only sets 5 & 10)",
    )
    dl.add_variables_argument(parser)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
        f.write(CAVEATS)
    for name in dl.resolve_variables(args.variables):
        run_for_predictand(name, args.all_sets)


if __name__ == "__main__":
    main()
