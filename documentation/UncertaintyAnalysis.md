# Uncertainty Analysis — Concept and Practical Guide

To complete our analysis framework we need to include uncertainties both yield/normalization uncertainties that might effect the relative yields but most importantly shape uncertainties.

Here I set some guidelines to plan how we include the large number of possible uncertainties:

Purpose
- Provide a compact, actionable plan for identifying, propagating, and summarizing uncertainties in the analysis.
- Focus on reproducible studies that exercise the full pipeline (selection → reconstruction → fit).

1) Types of uncertainties to consider
- Statistical: finite data / MC sample size.
- Experimental systematic: momentum scale/resolution, timing calibration, detector efficiency, alignment.
- Selection/systematic: cut thresholds, veto efficiencies, reconstruction choices.
- Background-modeling: shape and normalization uncertainties (e.g. DIO PDF variants, cosmic/RPC templates).
- Theory/modeling: cross-section or physics-model parameters used in simulation.
- Technical: pile-up, readout windows, event merging, file-reading differences.

2) General strategy
- Enumerate sources and assign a controlled variation for each (±1σ, alternative model, reweighting function).
- Parametrize each source as a nuisance parameter (NP) where possible (linear shift, scale, smearing sigma, template weight).
- Define a baseline pipeline run that produces the main results (yields, fit parameters, pull distributions).

3) Methods to propagate uncertainties
- Refit with shifted inputs: run the full pipeline after applying each systematic shift and record the change in the final parameters.
- Toy Monte Carlo (pseudo-experiments): generate ensembles with varied nuisance parameters and fit each; measure bias, variance, and coverage.
- Profile (frequentist) approach: include NPs in the fit (with constraints) and profile over them to produce confidence intervals.
- Bayesian marginalization: include priors on NPs and marginalize to obtain posterior distributions for parameters of interest.
- Reweighting / event-by-event variations: apply weights instead of regenerating samples when appropriate.
- Covariance propagation: compute analytic or numeric derivatives and propagate the covariance to final observables when re-fits are too expensive.

4) Practical workflow (recommended)
- Step A — Inventory: make a short table of sources: name, typical size, how to vary (file/parameter/weight), expected direction (see the `uncertainties1 package)
- Step B — Small-scale checks: for each source, produce a quick diagnostic (histogram overlays before/after variation) to detect gross mismodeling.
- Step C — One-at-a-time evaluation: apply each variation individually, run the full pipeline, and capture delta metrics (shift in parameter, change in uncertainty).
- Step D — Combined effects: for correlated NPs, consider joint toys or multi-parameter profiling to capture interplay.
- Step E — Ranking and impact: produce an impact table (absolute and relative shifts) and a waterfall plot showing how uncertainties add in quadrature or via profiling.

5) Metrics and visualizations to produce
- Shift table: parameter shifts vs. baseline for each systematic.
- Pull/coverage plots from pseudo-experiments: distribution of (fitted - true)/fit_error.
- Waterfall/impact plot: show contributions of top N systematics to the total uncertainty.
- Correlation matrix: between fit parameters and NPs.
- Overlayed spectra/histograms: baseline vs. shifted inputs for key distributions (momentum, time).
- Toy summary: bias and RMS as a function of injected NP value.

6) Implementation notes and file layout
- Keep each systematic variation as a small, readable script under `py-fitter/uncertainties/` (or a similar folder). Each script should:
  - accept the same CLI options as `process.py` (filelist, location, cuts) and produce diagnostics into a named output folder.
  - save the derived inputs (e.g. shifted `mom_mag.npz` or pickled filtered arrays) so re-running fits is cheap.
- Create a driver `run_systematics.py` that enumerates variations, dispatches jobs (local/cluster), and collects results into `uncertainties/outputs/`.
- Use consistent filenames: `<basename>__sys-<NAME>__shift-<dir>.npz` and `<basename>__sys-<NAME>__fit.json` to make collection and comparison trivial.

7) Automated tests and reproducibility
- Add a quick smoke test (similar to `run_unit_test.py`) that runs the baseline and a small handful of systematic shifts and stores a summary CSV.
- Store logs under `py-fitter/unit_test/uncertainties/` and require attaching them to PRs that touch core code (per your GettingStarted requirement).

8) Scaling to expensive studies
- If re-fitting is expensive, prefer reweighting or analytic propagation for first-pass ranking.
- For final results, run full toys only for top-ranked systematics or combined nuisance sets.

9) Quick checklist for a single systematic
- Define variation (scale/shift/alternative PDF).
- Produce diagnostic histogram (before/after).
- Run pipeline and save outputs (filtered data, `mom_mag.npz`, fit results JSON/PNG).
- Record metric: shift in parameter and change in fit uncertainty.

10) Provide a README.md with commands to anlayze the specific systematic.


11) Next steps you can ask me to implement
- scaffold `py-fitter/systematics/` with a template `scale_momentum.py` and a `collect_results.py` summarizer.
- add a small smoke test that runs baseline + one systematic and stores logs into `py-fitter/unit_test/systematics/`.

Keep this document near your other analysis docs so contributors can follow the prescribed workflow.
