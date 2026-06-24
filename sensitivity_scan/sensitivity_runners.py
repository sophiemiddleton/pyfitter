"""
Usage: called from run_sens_scan.py — do not run directly.
Adapted for 2D (Momentum and Time) parallel toy scans.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyutils.pylogger import Logger
from config import GLOBAL_VERBOSITY
import numpy as np
import awkward as ak
import matplotlib
import multiprocessing

# Module logger
logger = Logger(print_prefix='[sensitivity_runners_2d] ', verbosity=GLOBAL_VERBOSITY)


def extract_fitted_yields(fit_result):
    """Extract fitted yield (N) for each process from the 2D fit result."""
    params = fit_result.params
    yields = {}
    for param_obj in params:
        p_name = param_obj.name
        if p_name.startswith('N_'):
            proc = p_name.replace('N_', '')
            yields[proc] = float(params[param_obj]['value'])
    return yields


def single_toy_task(args):
    """Worker function for one 2D toy in the parallel scan.
    
    Parameters passed as a single tuple (all pickleable):
      mu, fit_range_mom, fit_range_time, constraints_dir, verbose,
      bg_yields, component_cats, plot_toys, plot_dir, toy_idx
    """
    (mu, fit_range_mom, fit_range_time, constraints_dir, verbose,
     bg_yields, component_cats, plot_toys, plot_dir, toy_idx) = args

    import numpy as np
    import awkward as ak
    import copy
    import zfit
    from fit_module import Unbinned_2d_fit_mom_time
    from results_module import ResultsClass

    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # 1 & 2. Generate 2D Background & Signal events using a fresh zfit model
    # ------------------------------------------------------------------
    # We instantiate a dummy 2D fit setup to get access to the true combined PDF
    # and sample from it cleanly.
    dummy_mom = np.linspace(fit_range_mom[0], fit_range_mom[1], 10)
    dummy_time = np.linspace(fit_range_time[0], fit_range_time[1], 10)
    
    # Temporarily set up a background-only yield snapshot
    from model.physics_components import mom_components
    orig_snapshot = copy.deepcopy(mom_components)
    fixed_components = copy.deepcopy(orig_snapshot)
    
    # Fix background starting points near their nominal fitted values
    for proc, comp in fixed_components.items():
        if proc == 'CE':
            continue
        fitted_n = bg_yields.get(proc, 0.0)
        if fitted_n > 0:
            pars_copy = dict(comp.get('pars', {}))
            pars_copy['N'] = (fitted_n, 0.0, max(fitted_n * 10, 1e4))
            fixed_components[proc]['pars'] = pars_copy

    # Inject the injected signal yield (mu) into the generation phase
    if 'CE' in fixed_components:
        pars_copy = dict(fixed_components['CE'].get('pars', {}))
        pars_copy['N'] = (float(mu), 0.0, max(float(mu) * 10, 100.0))
        fixed_components['CE']['pars'] = pars_copy

    mom_components.clear()
    mom_components.update(fixed_components)

    try:
        # Build the model context
        _, _, _, combine_pdf, _ = Unbinned_2d_fit_mom_time(
            ak.Array(dummy_mom), ak.Array(dummy_time), component_cats,
            fit_range_mom, fit_range_time, plot_truth=False, verbose=0,
            plot_NLL=False, plot_results=False
        )
        
        # Sample directly from the 2D zfit Space using its native sampler
        sampler = combine_pdf.create_sampler()
        sampler.resample()
        toy_array = sampler.numpy()
        
        mom_toy = toy_array[:, 0]
        time_toy = toy_array[:, 1]
        
    except Exception as e:
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': f'Generation failed: {e}'}
    finally:
        mom_components.clear()
        mom_components.update(orig_snapshot)

    if len(mom_toy) == 0:
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'Empty toy generated'}

    # ------------------------------------------------------------------
    # 3. Optional toy plot (2D Scatter)
    # ------------------------------------------------------------------
    if plot_toys > 0 and toy_idx < plot_toys:
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6, 5))
        plt.scatter(mom_toy, time_toy, alpha=0.5, s=2, color='C1')
        plt.xlabel('Momentum [MeV/c]')
        plt.ylabel('Time [ns]')
        plt.title(f'2D Toy mu={mu}, toy={toy_idx}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'toy_mu{mu}_toy{toy_idx}.png'))
        plt.close()

    # ------------------------------------------------------------------
    # 4. Fit the 2D Toy Data
    # ------------------------------------------------------------------
    mom_ak = ak.Array(mom_toy)
    time_ak = ak.Array(time_toy)

    try:
        fitresult, poi, loss, combine_pdf, _ = Unbinned_2d_fit_mom_time(
            mom_ak, time_ak, component_cats,
            fit_range_mom, fit_range_time,
            plot_truth=False, verbose=verbose,
            plot_NLL=False, plot_results=False
        )
    except Exception as e:
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': f'Fit exception: {e}'}

    # ------------------------------------------------------------------
    # 5. Extract Upper Limit via ResultsClass 
    # ------------------------------------------------------------------
    rc = ResultsClass(mom_ak, fitresult, verbose=verbose)
    
    # Isolate the concrete POI object
    signal_param = None
    for param in combine_pdf.get_params():
        if 'ce' in param.name.lower() or 'cem' in param.name.lower():
            signal_param = param
            break

    if signal_param  is None:
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'POI signal parameter missing'}

    ul_obj = rc.GetUL(signal_param, loss, [], combine_pdf, None,
                      fit_range_mom[0], fit_range_mom[1],
                      sig_yield=0, CL=0.90, opt='asym')

    ul_value = None
    if hasattr(ul_obj, 'limits_result') and ul_obj.limits_result is not None:
        try:
            ul_value = float(ul_obj.limits_result['observed'])
        except Exception:
            ul_value = None

    if ul_value is None:
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'Limit calculation returned None'}

    return {'mu': mu, 'ul': float(ul_value)}


def parallel_toy_scan_2d(mu_grid, ntoys, fit_result, component_cats, fit_range_mom, fit_range_time,
                         constraints_dir=None, verbose=0, plot_toys=0, plot_dir='toy_plots_2d', n_workers=4):
    """Run a parallelised 2D toy sensitivity scan."""
    bg_yields = extract_fitted_yields(fit_result)
    logger.log(f'Fitted background yields for toy base: {bg_yields}', 'info')

    tasks = []
    for mu in mu_grid:
        for toy_idx in range(ntoys):
            tasks.append((
                float(mu), tuple(fit_range_mom), tuple(fit_range_time),
                constraints_dir, verbose,
                bg_yields, component_cats,
                plot_toys, plot_dir, toy_idx,
            ))

    with multiprocessing.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        raw_results = pool.map(single_toy_task, tasks)

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
        logger.log(f'mu={mu_f}: n_ok={n_ok}, n_fail={n_fail}, '
                   f'median UL={summary[mu_f]["median"]:.3f}', 'info')

    return summary