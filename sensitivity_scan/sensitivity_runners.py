"""Example fit_runner wrappers for `ResultsClass.SensitivityFromMocks`.

These wrappers adapt the project's existing fit functions to the simple
callable interface expected by `SensitivityFromMocks(mock_samples, fit_runner)`.
Each runner returns a dict containing the numeric metric under key `'ul'` by
default so it can be consumed without further unpacking.

Notes:
- The wrappers are intentionally defensive: they try to extract a numeric
  upper limit from the `UpperLimit` object produced by `ResultsClass.GetUL`.
  If that fails, they fall back to a proxy value (median of POI scan) so the
  sensitivity scan can continue.
"""

import numpy as np
import awkward as ak
import os
import matplotlib.pyplot as plt
import multiprocessing

from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
from results_module import ResultsClass
from pyutils.pyprocess import Processor


def _to_numeric_array(x, verbose=0):
    """Coerce input `x` into a 1D numeric numpy array.

    Handles object-dtype arrays by concatenating contained arrays/lists
    or by elementwise float conversion with NaN fallback.
    Returns numpy.ndarray or raises ValueError if coercion fails.
    """
    arr = np.asarray(x)

    # If arr.dtype is object, attempt smarter coercions
    if arr.dtype == object:
        # try concatenating nested arrays/lists
        pieces = [np.asarray(u).ravel() for u in arr]
        if len(pieces) == 0:
            return np.asarray([], dtype=float)
        arr2 = np.concatenate(pieces)
        if verbose:
            print('[sensitivity_runners] Coerced object-array by concatenation; new shape', arr2.shape)
        return arr2.astype(float)

    # For numeric arrays, just ensure float dtype and flatten
    return np.asarray(arr, dtype=float).ravel()


def fit_runner_1d_ul(sample, fit_range=(95.0, 115.0), constraints_dir='uncertainties/Cosmic_test', verbose=0):
    """Run the 1D momentum fitter on a mock sample and return an estimated UL.

    Parameters
    ----------
    sample : array-like
      1D array-like of momentum values (numpy or awkward)
    fit_range : tuple
      (low, high) fit range for momentum
    constraints_dir : str
      directory containing `constraints.json` / `templates.npz` (optional)
    verbose : int
      verbosity forwarded to fit functions

    Returns
    -------
    dict with key 'ul' (numeric upper limit) and 'fitresult' (zfit result object)
    """
    # Coerce sample to numeric numpy array then to awkward
    mom_np = _to_numeric_array(sample, verbose=verbose)
    mom_mag = ak.Array(mom_np)
    fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
        mom_mag,
        [],        # track_cat (not used for mocks)
        [],        # count_particle_types
        fit_range[0],
        fit_range[1],
        False,
        verbose,
        minos=False,
        plot_NLL=False,
        plot_results=False,
        constraints_dir=constraints_dir,
    )

    # Use ResultsClass to produce an UpperLimit object and extract a numeric UL
    rc = ResultsClass(mom_mag, fitresult, verbose=verbose)
    # defensive check: ensure returned POI looks like a signal yield (e.g. "N_CE").
    poi_name = getattr(par, 'name', None)
    if poi_name is None or not (poi_name.startswith('N_') or 'CE' in poi_name.upper() or 'SIG' in poi_name.upper()):
        return {'ul': float('nan'), 'fitresult': fitresult, 'ul_failed': True, 'error': f'Returned POI "{poi_name}" does not look like a signal yield'}
    ul_obj = rc.GetUL(par, loss, nlls, combine_pdf, constraints, fit_range[0], fit_range[1], sig_yield=0, CL=0.90, opt='asym')

    # Try to get a numeric upper limit from the returned object
    ul_value = None
    for meth in ('upperlimit', 'upper_limit', 'upperLimit', 'limit'):
        if hasattr(ul_obj, meth):
            try:
                ul_value = float(getattr(ul_obj, meth)(alpha=0.05, CLs=True))
                break
            except Exception:
                try:
                    ul_value = float(getattr(ul_obj, meth)())
                    break
                except Exception:
                    ul_value = None
    if ul_value is None:
        # try to use a proxy (median of POI scan) if available
        ul_value = float(np.median(ul_obj.poinull.values)) if hasattr(ul_obj, 'poinull') and hasattr(ul_obj.poinull, 'values') else float('nan')

    return {'ul': float(ul_value), 'fitresult': fitresult, 'ul_obj': ul_obj, 'par': par, 'loss': loss, 'nlls': nlls, 'combine_pdf': combine_pdf, 'constraints': constraints}


def fit_runner_2d_ul(mom_sample, time_sample, fit_range_mom=(95.0, 115.0), fit_range_time=(400.0, 1695.0), constraints_dir='uncertainties/Cosmic_test', verbose=0):
    """Run the 2D fitter on mock momentum+time samples and return an estimated UL.

    Parameters
    ----------
    mom_sample : array-like
      1D array-like of momentum values
    time_sample : array-like
      1D array-like of time values (same length as flattened mom_sample)
    fit_range_mom, fit_range_time : tuple
      fit ranges for mom and time
    constraints_dir : str
      uncertainties package directory
    verbose : int

    Returns
    -------
    dict with key 'ul' and 'fitresult'
    """
    mom_np = _to_numeric_array(mom_sample, verbose=verbose)
    time_np = _to_numeric_array(time_sample, verbose=verbose)
    mom_mag = ak.Array(mom_np)
    times = ak.Array(time_np)
    res, par, loss, combine_pdf, norms = Unbinned_2d_fit_mom_time(
        mom_mag,
        times,
        [],  # track_cat
        [],  # count_particle_types
        [fit_range_mom[0], fit_range_mom[1]],
        [fit_range_time[0], fit_range_time[1]],
        False,
        verbose,
        plot_results=False,
        constraints_dir=constraints_dir,
    )

    fitresult = res
    # Use ResultsClass with flattened 1D mom sample as data
    rc = ResultsClass(mom_mag, fitresult, verbose=verbose)
    ul_obj = rc.GetUL(par, loss, [], combine_pdf, [], fit_range_mom[0], fit_range_mom[1], sig_yield=0, CL=0.90, opt='asym')

    ul_value = None
    try:
        ul_value = ul_obj.upperlimit(alpha=0.05, CLs=True)
    except Exception:
        ul_value = float(np.median(ul_obj.poinull.values)) if hasattr(ul_obj, 'poinull') and hasattr(ul_obj.poinull, 'values') else float('nan')

    return {'ul': float(ul_value), 'fitresult': fitresult, 'ul_obj': ul_obj}


def toy_scan_from_model(combine_pdf, par, fit_runner, mu_grid, ntoys=100, n_per_toy=1000, fit_runner_args=(), fit_runner_kwargs=None, verbose=0, plot_first_n=0, plot_dir=None, compute_sigmas=False, sig_calc_opt='asym'):
    """Run a toy-based sensitivity scan by sampling from `combine_pdf` at
    several injected signal strengths `mu_grid`.

    Parameters
    ----------
    combine_pdf : zfit.pdf.ZPDF
      Extended model PDF (must provide `create_sampler()` or `sample()`)
    par : zfit.Parameter or parameter-like
      The parameter to set as the injected signal (POI). Pass the zfit
      parameter object used by the model.
    fit_runner : callable
      Callable that accepts a single mock sample (1D array) and returns a
      numeric metric or a dict containing key 'ul'. For 2D fits, provide a
      fit_runner that accepts a tuple (mom, time) or two args — the function
      will call it with a single array if it expects one argument.
    mu_grid : iterable
      Values of injected signal strength to scan.
    ntoys : int
      Number of toys per grid point.
    n_per_toy : int
      Number of events to sample per toy. If the sampler provides variable
      extended sampling, this may be ignored.
    fit_runner_args, fit_runner_kwargs : additional args forwarded to fit_runner
    verbose : int
      Verbosity

    Returns
    -------
    dict mapping mu -> list of numeric metrics (one per toy) and summary stats
    """
    if fit_runner_kwargs is None:
        fit_runner_kwargs = {}

    results = {}

    # create a sampler; prefer combine_pdf.create_sampler() if available
    sampler = None
    if hasattr(combine_pdf, 'create_sampler'):
        sampler = combine_pdf.create_sampler()

    for mu in mu_grid:
        mu_vals = []
        errors = []
        for itoy in range(int(ntoys)):
            try:
                # attempt to draw a toy dataset
                if sampler is not None:
                    # robustly try multiple sampler APIs
                    data = None
                    # try common sample method names
                    try_methods = [
                        ('sample', lambda s, n: s.sample(n)),
                        ('draw', lambda s, n: s.draw(n)),
                        ('__call__', lambda s, n: s(n)),
                    ]
                    for name, fn in try_methods:
                        if hasattr(sampler, name) or (name == '__call__' and callable(sampler)):
                            try:
                                data = fn(sampler, n_per_toy)
                                break
                            except Exception:
                                data = None
                    # try resample(+sample/draw) to set POI then sample
                    if data is None and hasattr(sampler, 'resample'):
                        try:
                            sampler.resample({par: mu})
                            for name, fn in try_methods:
                                if hasattr(sampler, name) or (name == '__call__' and callable(sampler)):
                                    try:
                                        data = fn(sampler, n_per_toy)
                                        break
                                    except Exception:
                                        data = None
                        except Exception:
                            data = None
                    # final fallback to combine_pdf.sample if available
                    if data is None:
                        if hasattr(combine_pdf, 'sample'):
                            try:
                                data = combine_pdf.sample(n_per_toy)
                            except Exception as e:
                                raise RuntimeError(f"Sampler failed to produce toy: {e}")
                        else:
                            raise RuntimeError('No sampler available on model; cannot generate toys')
                else:
                    # try combine_pdf.sample if available
                    if hasattr(combine_pdf, 'sample'):
                        data = combine_pdf.sample(n_per_toy)
                    else:
                        raise RuntimeError('No sampler available on model; cannot generate toys')

                # Convert sampled data into a plain numpy array expected by fit_runner
                # Try several common sampler/data accessors (zfit.Data, SamplerData, awkward, numpy)
                arr = None
                try:
                    if hasattr(data, 'to_numpy'):
                        arr = np.asarray(data.to_numpy())
                    elif hasattr(data, 'numpy'):
                        arr = np.asarray(data.numpy())
                    elif hasattr(data, 'value'):
                        arr = np.asarray(data.value())
                    elif hasattr(data, 'samples'):
                        arr = np.asarray(data.samples)
                    else:
                        arr = np.asarray(data)
                except Exception:
                    try:
                        arr = np.asarray(list(data))
                    except Exception:
                        # give up and pass raw object through (fit runner wrappers should handle)
                        arr = data
                # If we got an object-dtype array, try to coerce numeric contents
                try:
                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        arr = np.asarray([np.asarray(x) for x in arr])
                except Exception:
                    pass

                # Optional: save toy plots for the first N toys per mu
                try:
                    if plot_dir and plot_first_n and itoy < int(plot_first_n):
                        odir = os.path.join(plot_dir, f"mu_{mu}")
                        os.makedirs(odir, exist_ok=True)
                        fname = os.path.join(odir, f"toy_{itoy}.png")
                        plt.figure()
                        # If 2D array with two columns, make a scatter; else histogram
                        try:
                            if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.shape[1] == 2:
                                plt.scatter(arr[:, 0], arr[:, 1], s=4)
                                plt.xlabel('mom')
                                plt.ylabel('time')
                                plt.title(f'mu={mu} toy={itoy} (scatter)')
                            else:
                                plt.hist(np.asarray(arr).ravel(), bins=60, histtype='stepfilled', alpha=0.7)
                                plt.xlabel('Momentum [MeV/c]')
                                plt.title(f'mu={mu} toy={itoy} (hist)')
                            plt.tight_layout()
                            plt.savefig(fname)
                            plt.close()
                        except Exception:
                            try:
                                plt.close()
                            except Exception:
                                pass
                except Exception:
                    pass

                # choose how to call fit_runner depending on its signature
                try:
                    # If 2D array with two columns and runner expects two args, try splitting
                    if hasattr(arr, 'ndim') and arr.ndim == 2 and arr.shape[1] == 2:
                        # try calling with two separate arrays
                        res = fit_runner(arr[:, 0], arr[:, 1], *fit_runner_args, **fit_runner_kwargs)
                    else:
                        res = fit_runner(arr, *fit_runner_args, **fit_runner_kwargs)
                except TypeError:
                    # fallback: try single-argument call
                    res = fit_runner(arr, *fit_runner_args, **fit_runner_kwargs)

                # extract numeric value and record any diagnostic errors returned by fit_runner
                if isinstance(res, dict):
                    ul_raw = res.get('ul', None)
                    try:
                        v = float(ul_raw)
                    except Exception:
                        v = float('nan')
                    # if the runner flagged failure or produced NaN, record its error message
                    if res.get('ul_failed', False) or (isinstance(v, float) and np.isnan(v)):
                        err_msg = res.get('error', f'Invalid UL value returned: {ul_raw}')
                        errors.append(err_msg)
                        if verbose:
                            print(f"Toy produced invalid UL for mu={mu} toy={itoy}; recorded error: {err_msg}")
                        continue
                    # optionally compute discovery significance for this toy if fit details are available
                    sigma_val = None
                    if compute_sigmas:
                        try:
                            fitres = res.get('fitresult', None)
                            par_res = res.get('par', None)
                            loss_res = res.get('loss', None)
                            if fitres is not None and par_res is not None and loss_res is not None:
                                rc = ResultsClass(arr, fitres, verbose=0)
                                try:
                                    sig = rc.GetSignifcance(par_res, loss_res, opt=sig_calc_opt)
                                    # sig is (pvalue, sigma)
                                    sigma_val = float(sig[1]) if sig is not None and len(sig) > 1 else float('nan')
                                except Exception:
                                    sigma_val = float('nan')
                        except Exception:
                            sigma_val = float('nan')
                    else:
                        sigma_val = None
                else:
                    try:
                        v = float(res)
                    except Exception:
                        v = float('nan')
                    if isinstance(v, float) and np.isnan(v):
                        err_msg = f'Non-numeric fit_runner return: {res!r}'
                        errors.append(err_msg)
                        sigma_val = None
                        if verbose:
                            print(f"Toy produced non-numeric result for mu={mu} toy={itoy}; recorded error: {err_msg}")
                        continue

                # record sigma if computed
                if compute_sigmas:
                    if 'sigmas' not in locals():
                        sigmas = []
                    sigmas.append(sigma_val)

            except Exception as e:
                import traceback
                err = traceback.format_exc()
                errors.append(err)
                if verbose:
                    print(f"Toy generation/fit failed for mu={mu} toy={itoy}: {e}")
                    print(err)
                continue

            mu_vals.append(v)

        if len(mu_vals) == 0:
            results[mu] = {'values': [], 'median': np.nan, 'p16': np.nan, 'p84': np.nan, 'errors': errors}
        else:
            arr = np.array(mu_vals, dtype=float)
            results[mu] = {
                'values': mu_vals,
                'median': float(np.median(arr)),
                'p16': float(np.percentile(arr, 16)),
                'p84': float(np.percentile(arr, 84)),
                'errors': errors,
            }
        # attach sigmas if computed
        if compute_sigmas:
            try:
                sig_arr = np.array(sigmas, dtype=float)
                results[mu]['sigmas'] = list(sig_arr)
                results[mu]['sigma_median'] = float(np.nanmedian(sig_arr))
                results[mu]['sigma_p16'] = float(np.nanpercentile(sig_arr, 16))
                results[mu]['sigma_p84'] = float(np.nanpercentile(sig_arr, 84))
            except Exception:
                results[mu]['sigmas'] = []
                results[mu]['sigma_median'] = float('nan')
                results[mu]['sigma_p16'] = float('nan')
                results[mu]['sigma_p84'] = float('nan')

        # clear sigmas for next mu
        if 'sigmas' in locals():
            del sigmas

        if verbose:
            print(f"mu={mu}: n_success={len(mu_vals)}, median={results[mu]['median']}")

    return results


# --- Parallel toy scan utilities using pyutils.pyprocess ---
def single_toy_task(args):
    """
    Worker function for a single toy. Arguments should include all needed inputs.
    """
    # args: (mu, n_per_toy, fit_range, constraints_dir, fit_runner_name, verbose,
    #        nominal_data, plot_toys, plot_dir, toy_idx)
    mu, n_per_toy, fit_range, constraints_dir, fit_runner_name, verbose, nominal_data, plot_toys, plot_dir, toy_idx = args
    import numpy as np
    import awkward as ak
    import zfit
    from fit_module import Unbinned_fit_mom
    from results_module import ResultsClass

    rng = np.random.default_rng()

    # Generate background by resampling from nominal data (background-only sample).
    # Optionally inject Poisson(mu) CE signal events on top.
    if nominal_data is not None and len(nominal_data) > 0:
        mom_bkg = rng.choice(nominal_data, size=n_per_toy, replace=True)
    else:
        mom_bkg = rng.uniform(fit_range[0], fit_range[1], n_per_toy)

    # Inject signal (CE): Poisson(mu) events, Gaussian at 104.97 MeV
    n_sig = rng.poisson(mu)
    if n_sig > 0:
        ce_mean = 104.97
        ce_sigma = 0.2  # MeV
        mom_sig = rng.normal(loc=ce_mean, scale=ce_sigma, size=n_sig)
        mom = np.concatenate([mom_bkg, mom_sig])
    else:
        mom = mom_bkg

    # Plot and save toy data if requested
    if plot_toys > 0 and toy_idx < plot_toys:
        import matplotlib
        matplotlib.use('Agg')  # for headless environments
        import matplotlib.pyplot as plt
        import os
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(6,4))
        plt.hist(mom, bins=50, alpha=0.7, color='C0')
        plt.title(f'Toy sample (mu={mu}, toy={toy_idx})')
        plt.xlabel('Momentum')
        plt.ylabel('Entries')
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f'toy_mu{mu}_toy{toy_idx}.png')
        plt.savefig(plot_path)
        plt.close()
    # If we ended up with 0 events, the fit will fail; bail out early.
    if len(mom) == 0:
        print(f'[toy mu={mu} idx={toy_idx}] Empty toy (0 events) — returning nan UL')
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'Empty toy dataset'}

    mom_ak = ak.Array(mom)
    print(f'[toy mu={mu} idx={toy_idx}] n_bkg={len(mom_bkg)}, n_sig={n_sig}, total={len(mom)}')

    if fit_runner_name == 'fit_runner_1d_ul':
        try:
            fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
                mom_ak, [], [],
                fit_range[0], fit_range[1],
                False, verbose,
                minos=False, plot_NLL=False, plot_results=False,
                constraints_dir=constraints_dir,
            )
        except Exception as e:
            print(f'[toy mu={mu} idx={toy_idx}] Fit raised exception: {e}')
            return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': str(e)}

        rc = ResultsClass(mom_ak, fitresult, verbose=verbose)
        poi_name = getattr(par, 'name', None)
        if poi_name is None or not (poi_name.startswith('N_') or 'CE' in poi_name.upper() or 'SIG' in poi_name.upper()):
            return {'mu': mu, 'ul': float('nan'), 'ul_failed': True,
                    'error': f'POI "{poi_name}" does not look like a signal yield'}

        # Log validity of fit before computing UL
        try:
            valid = fitresult.valid
        except Exception:
            valid = None
        print(f'[toy mu={mu} idx={toy_idx}] fit valid={valid}, poi={poi_name}')

        ul_obj = rc.GetUL(par, loss, nlls, combine_pdf, constraints,
                          fit_range[0], fit_range[1], sig_yield=0, CL=0.90, opt='asym')

        ul_value = None
        for meth in ('upperlimit', 'upper_limit', 'upperLimit', 'limit'):
            if hasattr(ul_obj, meth):
                try:
                    ul_value = float(getattr(ul_obj, meth)(alpha=0.05, CLs=True))
                    break
                except Exception:
                    try:
                        ul_value = float(getattr(ul_obj, meth)())
                        break
                    except Exception:
                        ul_value = None

        # Do NOT fall back to median(poinull.values) — that always gives a
        # meaningless midpoint of the scan grid (historically produced 25.0).
        if ul_value is None:
            print(f'[toy mu={mu} idx={toy_idx}] upperlimit() failed — returning nan')
            return {'mu': mu, 'ul': float('nan'), 'ul_failed': True, 'error': 'upperlimit() returned None'}

        print(f'[toy mu={mu} idx={toy_idx}] ul={ul_value}')
        return {'mu': mu, 'ul': float(ul_value)}
    else:
        return {'mu': mu, 'ul': float('nan'), 'ul_failed': True,
                'error': f'Unknown fit_runner_name {fit_runner_name}'}

# For parallel toy scans, use parallel_toy_scan_with_multiprocessing (native Python, no pyutils dependency).

def parallel_toy_scan_with_multiprocessing(combine_pdf, par, fit_runner, mu_grid, ntoys=100, n_per_toy=1000, fit_runner_args=(), fit_runner_kwargs=None, n_workers=16):
    """
    Parallel toy scan using Python's multiprocessing.Pool.

    For each mu in mu_grid, generates `ntoys` toy datasets by resampling
    from nominal_data (background) and injecting Poisson(mu) CE signal events.
    Fits each toy with the full model and computes an upper limit on N_CE.
    Returns a summary dict per mu.
    """
    import numpy as np
    import multiprocessing
    fit_range = (95.0, 115.0)
    constraints_dir = 'uncertainties/Cosmic_test'
    fit_runner_name = 'fit_runner_1d_ul'
    verbose = 0
    nominal_data = None
    if hasattr(fit_runner, '__name__') and fit_runner.__name__ == 'fit_runner_1d_ul':
        fit_runner_name = 'fit_runner_1d_ul'
    plot_toys = 0
    plot_dir = 'toy_plots'
    if fit_runner_kwargs is not None:
        fit_range = fit_runner_kwargs.get('fit_range', fit_range)
        constraints_dir = fit_runner_kwargs.get('constraints_dir', constraints_dir)
        verbose = fit_runner_kwargs.get('verbose', verbose)
        nominal_data = fit_runner_kwargs.get('nominal_data', None)
        plot_toys = fit_runner_kwargs.get('plot_toys', 0)
        plot_dir = fit_runner_kwargs.get('plot_dir', 'toy_plots')
    tasks = []
    for mu in mu_grid:
        for toy_idx in range(ntoys):
            tasks.append((mu, n_per_toy, fit_range, constraints_dir, fit_runner_name, verbose, nominal_data, plot_toys, plot_dir, toy_idx))
    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(single_toy_task, tasks)
    out = {mu: [] for mu in mu_grid}
    for res in results:
        out[res['mu']].append(res['ul'])
    # Convert to summary dict per mu
    summary = {}
    for mu in mu_grid:
        arr = np.array(out[mu], dtype=float)
        summary[mu] = {
            'values': list(arr),
            'median': float(np.median(arr)) if len(arr) > 0 else float('nan'),
            'p16': float(np.percentile(arr, 16)) if len(arr) > 0 else float('nan'),
            'p84': float(np.percentile(arr, 84)) if len(arr) > 0 else float('nan'),
            'errors': [],  # Could be extended to collect errors
        }
        print(f"mu={mu}: median UL={summary[mu]['median']}, 1σ=({summary[mu]['p16']}, {summary[mu]['p84']})")
    return summary
