# Control Regions

This page documents the control-region strategy and how to run the simple
unbinned fits used to estimate shapes/yields for the Cosmic and RPC
components. The control-region helpers live in `py-fitter/control_region.py`.

## Principles

- Control-region fits must use the exact same event selection and object
  definitions as the final analysis fits (same SID, reconstructions, and all
  preselection cuts). This ensures the fitted shapes and yields are directly
  applicable to the main fit or to derive constraints.

- The control-region code is intentionally minimal and component-specific so
  that new per-component fits can be added later with the same interface.

## Cosmic control region

- Source sample: reconstructed OffSpill cosmic sample (the same OffSpill
  selection used in `process.py`). Use the reconstructed `mom_mag` array
  produced by the same pipeline that creates the final data input.

- Fit model: extended Chebyshev polynomial (implemented as
  `ControlRegion.fit_cosmic` in `py-fitter/control_region.py`). This is an
  extended unbinned likelihood fit using `zfit.pdf.Chebyshev`.

- Fit range and parameters:
  - Default momentum fit range: 80–150 MeV/c (adjustable via the method call).
  - Polynomial degree: configurable (default degree = 4).
  - Yield is a free, extended parameter (`N_Cosmic`).

- Usage pattern (conceptual):
  - Load/produce the `mom_mag` array for OffSpill cosmics using the standard
    selection in `process.py`.
  - Instantiate: `cr = ControlRegion(mom_mag_offspill)`
  - Run: `out = cr.fit_cosmic(fit_range=(80,150), degree=4, plot=True)`
  - `out` contains `result` (zfit FitResult), `params` dict, optional
    `figure` and `hist` for diagnostics.

## RPC control region

- Source sample: reconstructed e+ sample (select positive-charge electron
  candidates using the same selection cuts used in the main analysis in
  `process.py`). The RPC fit expects the final per-event reconstructed
  momentum `mom_mag` array for e+.

- Fit model: extended Gaussian (implemented as `ControlRegion.fit_rpc`). The
  fit returns the Gaussian mean and sigma and an extended yield parameter
  `N_RPC`.

- Fit range and parameters:
  - Default momentum fit range: 95–115 MeV/c (adjustable).
  - Initial guesses may be provided; otherwise data-derived estimates are
    used.

- Usage pattern (conceptual):
  - Produce the `mom_mag` array for the e+ selection.
  - Instantiate: `cr = ControlRegion(mom_mag_eplus)`
  - Run: `out = cr.fit_rpc(fit_range=(95,115), plot=True)`
  - `out` contains `result`, `mean`, `sigma`, `norm`, and optional
    `figure` and `hist`.

## General notes and extension

- The control-region functions assume input arrays are produced by the same
  reconstruction and selection code paths used for the main analysis; any
  mismatch will invalidate the transfer of shapes/yields.

- The module is written so additional component-fit methods can be added with
  the same API signature (returning a result dict, optional plotting, and
  a compact `params` summary).

- The fit implementations require `zfit` to be installed in the environment.

## Contact

If you want different default ranges, degrees, or stricter constraints on
parameters (e.g. freeze coefficients), update `py-fitter/control_region.py`
or open an issue describing the desired behaviour.
