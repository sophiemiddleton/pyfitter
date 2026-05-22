"""
Usage: called from run_sens_scan.py — do not run directly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import awkward as ak
import matplotlib
import multiprocessing


# ---------------------------------------------------------------------------
# Helpers for analytical PDF evaluation (pickleable — no zfit dependency)
# ---------------------------------------------------------------------------

def _eval_poly58(x_arr, a5, a6, a7, a8):
    """Evaluate the DIO poly58 unnormalized PDF over an array of x values."""
    m_mu = 105.194
    m_Al = 25133.0
    delta = np.maximum(m_mu - x_arr - x_arr**2 / (2 * m_Al), 0.0)
    return a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8


def _eval_chebyshev2(x_arr, c1, c2, x_min, x_max):
    """Evaluate the Chebyshev order-2 PDF (Cosmic) over an array of x values.
    
    Maps x -> [-1, 1] then evaluates 1 + c1*T1 + c2*T2.
    """
    x_std = 2.0 * (x_arr - x_min) / (x_max - x_min) - 1.0
    T1 = x_std
    T2 = 2.0 * x_std**2 - 1.0
    vals = 1.0 + c1 * T1 + c2 * T2
    return np.maximum(vals, 0.0)


def precompute_pdf_grids(fit_result, mom_components_dict, fit_range, n_grid=5000):
    """Evaluate each background PDF over a fine grid using the fitted parameter values.

    Returns a dict keyed by process name with entries:
        {'x': ndarray, 'weights': ndarray (normalised to sum=1)}

    CE is intentionally excluded — signal events are injected analytically.

    Parameters
    ----------
    fit_result : zfit FitResult
        The nominal fit result from Unbinned_fit_mom.
    mom_components_dict : dict
        The mom_components dictionary from mom_components.py.
    fit_range : tuple (float, float)
        (low, high) momentum fit range.
    n_grid : int
        Number of grid points for PDF evaluation.
    """
    x = np.linspace(fit_range[0], fit_range[1], n_grid)
    grids = {}

    params = fit_result.params  # dict: name -> {'value': float, ...}

    def _get(name, fallback):
        return float(params[name]['value']) if name in params else fallback

    for proc, cfg in mom_components_dict.items():
        if proc == 'CE':
            continue  # signal injected separately
        pdf_name = cfg['pdf']
        pars_cfg = cfg.get('pars', {})

        if pdf_name == 'poly58':
            a5 = _get('a5_' + proc, pars_cfg.get('a5', (8.97879e-17,))[0])
            a6 = _get('a6_' + proc, pars_cfg.get('a6', (1.17169e-17,))[0])
            a7 = _get('a7_' + proc, pars_cfg.get('a7', (-1.06599e-19,))[0])
            a8 = _get('a8_' + proc, pars_cfg.get('a8', (8.14251e-20,))[0])
            vals = _eval_poly58(x, a5, a6, a7, a8)

        elif pdf_name in ('poly2', 'poly5'):
            c1 = _get('c1_' + proc, pars_cfg.get('c1', (0.0,))[0])
            c2 = _get('c2_' + proc, pars_cfg.get('c2', (0.0,))[0])
            vals = _eval_chebyshev2(x, c1, c2, fit_range[0], fit_range[1])

        elif pdf_name == 'uniform':
            vals = np.ones(n_grid)

        else:
            # Fallback: uniform
            vals = np.ones(n_grid)

        # Normalize to a proper probability weight
        total = vals.sum()
        if total <= 0:
            vals = np.ones(n_grid)
            total = float(n_grid)
        weights = vals / total

        grids[proc] = {'x': x.copy(), 'weights': weights}

    return grids


def extract_fitted_yields(fit_result, mom_components_dict):
    """Extract fitted yield (N) for each process from the fit result.

    Falls back to the default_norms value in mom_components if not found.

    Returns dict: {proc: float}
    """
    from momentum_pdf_builder import mom_default_norms as default_norms
    params = fit_result.params
    yields = {}
    for proc in mom_components_dict:
        key = 'N_' + proc
        if key in params:
            yields[proc] = float(params[key]['value'])
        elif proc in default_norms:
            yields[proc] = float(default_norms[proc])
        else:
            yields[proc] = 0.0
    return yields


def extract_ce_shape_params(fit_result, mom_components_dict):
    """Return the EXPECTED (default) CE shape parameters from mom_components.

    We intentionally do NOT use the initial fit result for CE shape because the
    initial fit is performed on background-only data that has no CE signal, so
    the DSCB shape floats to a spurious minimum (e.g. mu_CE=100).  Instead we
    always use the central values from the mom_components configuration, which
    correspond to the physically expected CE signal shape (mu~104.97 MeV/c etc).

    Returns dict of param_name -> value for all CE shape params.
    """
    ce_cfg = mom_components_dict.get('CE', {})
    pars_cfg = ce_cfg.get('pars', {})

    shape_params = {}
    for p_name, val in pars_cfg.items():
        shape_params[p_name] = float(val[0]) if hasattr(val, '__len__') else float(val)

    return shape_params


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------

def single_toy_task_v2(args):
    """Worker function for one toy in the improved parallel scan.

    Parameters passed as a single tuple (all pickleable):
      mu, fit_range, constraints_dir, verbose,
      bg_yields,          # dict proc -> expected yield (float)
      pdf_grids,          # dict proc -> {'x': ndarray, 'weights': ndarray}
      ce_shape_params,    # dict param_name -> fixed value
      plot_toys, plot_dir, toy_idx
    """
    (mu, fit_range, constraints_dir, verbose,
     bg_yields, pdf_grids, ce_shape_params,
     plot_toys, plot_dir, toy_idx) = args

    import numpy as np
    import awkward as ak
    import copy
    import zfit
    from fit_module import Unbinned_fit_mom
    from results_module import ResultsClass

    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # 1. Generate background events by sampling from precomputed PDF grids
    #    with Poisson-fluctuated yields.
    # ------------------------------------------------------------------
    mom_parts = []
    for proc, grid in pdf_grids.items():
        expected = bg_yields.get(proc, 0.0)
        if expected <= 0:
            continue
        n = rng.poisson(expected)
        if n > 0:
            sampled = rng.choice(grid['x'], size=n, replace=True, p=grid['weights'])
            mom_parts.append(sampled)

    # ------------------------------------------------------------------
    # 2. Inject Poisson(mu) CE signal events (Gaussian at fixed shape).
    # ------------------------------------------------------------------
    ce_mean = ce_shape_params.get('mu', 104.97)
    ce_sigma = ce_shape_params.get('sigma', 0.5)
    n_sig = rng.poisson(mu)
    if n_sig > 0:
        mom_sig = rng.normal(loc=ce_mean, scale=ce_sigma, size=n_sig)
        mom_parts.append(mom_sig)

    if len(mom_parts) == 0:
        print(f'[toy mu={mu} idx={toy_idx}] Empty toy — returning nan')
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'Empty toy'}

    mom = np.concatenate(mom_parts)
    # Keep only events inside the fit range
    mom = mom[(mom >= fit_range[0]) & (mom <= fit_range[1])]

    if len(mom) == 0:
        print(f'[toy mu={mu} idx={toy_idx}] No events in fit range — returning nan')
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'No events in fit range'}

    n_bkg = len(mom) - n_sig
    print(f'[toy mu={mu} idx={toy_idx}] n_bkg≈{n_bkg}, n_sig={n_sig}, total={len(mom)}')

    # ------------------------------------------------------------------
    # 3. Optional toy plot
    # ------------------------------------------------------------------
    if plot_toys > 0 and toy_idx < plot_toys:
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.hist(mom, bins=50, alpha=0.7, color='C0')
        plt.xlabel('Momentum [MeV/c]')
        plt.title(f'Toy mu={mu}, toy={toy_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'toy_mu{mu}_toy{toy_idx}.png'))
        plt.close()

    mom_ak = ak.Array(mom)

    # ------------------------------------------------------------------
    # 4. Build the toy fit with CE shape FIXED and background yields
    #    initialised from the nominal fit (not default_norms).
    #    Temporarily override mom_components inside this worker only.
    # ------------------------------------------------------------------
    from physics_components import mom_components

    # Snapshot current state then mutate the live dict in-place.
    # fit_module uses 'from physics_components import mom_components' which binds
    # a reference to the dict object, not the module attribute — so we must
    # mutate the same object rather than rebinding.
    orig_snapshot = copy.deepcopy(mom_components)
    fixed_components = copy.deepcopy(orig_snapshot)

    # Fix CE shape: set each shape parameter to its fitted value, treat as fixed.
    # N_CE is NOT listed here — it is created by the 'N' special key path in
    # MomModel and will float freely with a negative lower bound (see momPDF_module).
    if 'CE' in fixed_components:
        fixed_ce_pars = {}
        for p_name, fitted_val in ce_shape_params.items():
            fixed_ce_pars[p_name] = (fitted_val,)  # single-element tuple → fixed
        fixed_components['CE']['pars'] = fixed_ce_pars
        fixed_components['CE']['treat_params'] = 'fix'
        # Widen mu_CE bounds so hepstats UL scan does not clip when profiling
        # (hepstats internally tries values slightly outside the fitted range).
        # With treat_params='fix' these bounds are only used if zfit falls back
        # to a floating parameter; make them wide enough to never constrain.
        fixed_components['CE']['pars']['mu'] = (
            ce_shape_params.get('mu', 104.0), 90.0, 115.0)
        fixed_components['CE']['pars']['alphaR'] = (
            ce_shape_params.get('alphaR', 2.227), 0.0, 200.0)

    # Initialise background yields from nominal fit so the minimizer starts
    # near the expected value rather than the hardcoded default_norms values.
    for proc, comp in fixed_components.items():
        if proc == 'CE':
            continue
        fitted_n = bg_yields.get(proc, None)
        if fitted_n is not None and fitted_n > 0:
            # Inject as 'N' key so MomModel uses it as the yield starting value.
            pars_copy = dict(comp.get('pars', {}))
            pars_copy['N'] = (fitted_n, 0.0, max(fitted_n * 10, 1e4))
            fixed_components[proc]['pars'] = pars_copy

    # Apply overrides in-place
    mom_components.clear()
    mom_components.update(fixed_components)

    try:
        fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
            mom_ak, [], [],
            fit_range[0], fit_range[1],
            False, verbose,
            minos=False, plot_NLL=False, plot_results=False,
            constraints_dir=constraints_dir,
        )
    except Exception as e:
        mom_components.clear()
        mom_components.update(orig_snapshot)
        print(f'[toy mu={mu} idx={toy_idx}] Fit exception: {e}')
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': str(e)}
    finally:
        mom_components.clear()
        mom_components.update(orig_snapshot)

    # ------------------------------------------------------------------
    # 5. Validity check and POI identification
    # ------------------------------------------------------------------
    try:
        valid = fitresult.valid
    except Exception:
        valid = None
    poi_name = getattr(par, 'name', None)
    print(f'[toy mu={mu} idx={toy_idx}] fit valid={valid}, POI={poi_name}')

    if poi_name is None or not ('CE' in poi_name.upper() or poi_name.startswith('N_')):
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True,
                'error': f'POI "{poi_name}" is not a signal yield'}

    # ------------------------------------------------------------------
    # 6. Compute upper limit via hepstats AsymptoticCalculator
    # ------------------------------------------------------------------
    rc = ResultsClass(mom_ak, fitresult, verbose=verbose)
    ul_obj = rc.GetUL(par, loss, nlls, combine_pdf, constraints,
                      fit_range[0], fit_range[1],
                      sig_yield=0, CL=0.90, opt='asym')

    # upperlimit() returns dict[str, float] with keys:
    #   'observed', 'expected', 'expected_p1', 'expected_m1', 'expected_p2', 'expected_m2'
    # GetUL attaches the result as ul_obj.limits_result; fall back to calling it directly.
    import traceback as _tb
    ul_value = None
    if hasattr(ul_obj, 'limits_result') and ul_obj.limits_result is not None:
        try:
            ul_value = float(ul_obj.limits_result['observed'])
        except Exception:
            ul_value = None

    if ul_value is None:
        try:
            ul_dict = ul_obj.upperlimit(alpha=0.05, CLs=True)
            ul_value = float(ul_dict['observed'])
        except Exception as e:
            print(f'[toy mu={mu} idx={toy_idx}] upperlimit() exception: {type(e).__name__}: {e}')
            print(_tb.format_exc())
            ul_value = None

    if ul_value is None:
        print(f'[toy mu={mu} idx={toy_idx}] upperlimit() failed — returning nan')
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True,
                'error': 'upperlimit() returned None'}

    print(f'[toy mu={mu} idx={toy_idx}] UL={ul_value:.3f}')
    return {'mu': mu, 'ul': float(ul_value)}


# ---------------------------------------------------------------------------
# Parallel scan driver
# ---------------------------------------------------------------------------

def parallel_toy_scan_v2(mu_grid, ntoys, fit_result, mom_components_dict, fit_range,
                          constraints_dir=None, verbose=0,
                          plot_toys=0, plot_dir='toy_plots',
                          n_workers=4):
    """Run a parallelised toy sensitivity scan with improved toy generation.

    Parameters
    ----------
    mu_grid : array-like
        Injected signal yields to scan (N_CE values).
    ntoys : int
        Number of toy experiments per mu point.
    fit_result : zfit FitResult
        Nominal background-only fit result (used to extract PDF shapes and yields).
    mom_components_dict : dict
        The mom_components dict from mom_components.py.
    fit_range : tuple (float, float)
        Momentum fit range.
    constraints_dir : str or None
        Path to constraints JSON directory.
    verbose : int
    plot_toys : int
        Save plots for the first N toys per mu point.
    plot_dir : str
    n_workers : int
        Number of parallel worker processes.

    Returns
    -------
    dict mapping mu -> {'values', 'median', 'p16', 'p84', 'n_success', 'n_failed'}
    """
    # Pre-compute everything the workers need (all pickleable)
    pdf_grids = precompute_pdf_grids(fit_result, mom_components_dict, fit_range)
    bg_yields = extract_fitted_yields(fit_result, mom_components_dict)
    ce_shape_params = extract_ce_shape_params(fit_result, mom_components_dict)

    print('Fitted background yields:', {k: f'{v:.1f}' for k, v in bg_yields.items() if k != 'CE'})
    print('Fixed CE shape params:', {k: f'{v:.4f}' for k, v in ce_shape_params.items()})

    tasks = []
    for mu in mu_grid:
        for toy_idx in range(ntoys):
            tasks.append((
                float(mu), tuple(fit_range),
                constraints_dir, verbose,
                bg_yields, pdf_grids, ce_shape_params,
                plot_toys, plot_dir, toy_idx,
            ))

    with multiprocessing.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        raw_results = pool.map(single_toy_task_v2, tasks)

    # Aggregate
    out = {float(mu): [] for mu in mu_grid}
    n_failed = {float(mu): 0 for mu in mu_grid}
    for res in raw_results:
        mu_key = float(res['mu'])
        ul = res['ul']
        if np.isfinite(ul):
            out[mu_key].append(ul)
        else:
            n_failed[mu_key] += 1

    summary = {}
    for mu in mu_grid:
        mu_f = float(mu)
        arr = np.array(out[mu_f], dtype=float)
        n_ok = len(arr)
        n_fail = n_failed[mu_f]
        if n_ok > 0:
            summary[mu_f] = {
                'values': list(arr),
                'median': float(np.median(arr)),
                'p16': float(np.percentile(arr, 16)),
                'p84': float(np.percentile(arr, 84)),
                'n_success': n_ok,
                'n_failed': n_fail,
            }
        else:
            summary[mu_f] = {
                'values': [],
                'median': float('nan'),
                'p16': float('nan'),
                'p84': float('nan'),
                'n_success': 0,
                'n_failed': n_fail,
            }
        print(f'mu={mu_f}: n_ok={n_ok}, n_fail={n_fail}, '
              f'median UL={summary[mu_f]["median"]:.3f}, '
              f'1σ=({summary[mu_f]["p16"]:.3f}, {summary[mu_f]["p84"]:.3f})')

    return summary
