Cosmic_test uncertainty package
================================

Contents
- `constraints.json` : example Gaussian priors for `N_Cosmic`, `c1_Cosmic`, and `c2_Cosmic`.
- `templates.npz` : optional shape templates (nominal, up, down) — not included by default; run `make_templates.py` to generate it.
- `make_templates.py` : script to produce `templates.npz` for local testing.

Usage
-----
1. Generate templates (optional):

```bash
python uncertainties/Cosmic_test/make_templates.py
```

2. Run the fitter and point to this directory:

```python
from py_fitter.fit_module import Unbinned_fit_mom
Unbinned_fit_mom(mom_mag, track_cat, count_particle_types, 95, 115, constraints_dir='uncertainties/Cosmic_test', verbose=1)
```

Notes
-----
- `constraints.json` entries use `name` that maps to parameter names created by `MomModel` (e.g., `N_Cosmic`, `c1_Cosmic`).
- The loader in `py-fitter/uncertainty_loader.py` currently supports Gaussian priors only and will create `zfit.constraint.GaussianConstraint` objects for matching parameter names.
