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

from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
from results_module import ResultsClass


def _to_numeric_array(x, verbose=0):
    """Coerce input `x` into a 1D numeric numpy array.

    Handles object-dtype arrays by concatenating contained arrays/lists
    or by elementwise float conversion with NaN fallback.
    Returns numpy.ndarray or raises ValueError if coercion fails.
    """
    try:
        arr = np.asarray(x)
    except Exception:
        # try iterating
        try:
            arr = np.asarray(list(x))
        except Exception:
            raise ValueError('Cannot convert input to numpy array')

    # If object dtype, attempt smarter coercions
    if arr.dtype == object:
        # try concatenating nested arrays/lists
        try:
            pieces = [np.asarray(u).ravel() for u in arr]
            if len(pieces) == 0:
                return np.asarray([], dtype=float)
            arr2 = np.concatenate(pieces)
            if verbose:
                print('[sensitivity_runners] Coerced object-array by concatenation; new shape', arr2.shape)
            return arr2.astype(float)
        except Exception:
            # fallback: try elementwise float conversion
            out = []
            for u in arr:
                try:
                    out.append(float(u))
                except Exception:
                    out.append(np.nan)
            if verbose:
                print('[sensitivity_runners] Coerced object-array elementwise to float (NaNs possible)')
            return np.asarray(out, dtype=float)

    # For numeric arrays, just ensure float dtype and flatten
    try:
        return np.asarray(arr, dtype=float).ravel()
    except Exception:
        raise ValueError('Cannot coerce array to numeric')


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
    try:
        mom_np = _to_numeric_array(sample, verbose=verbose)
    except Exception as e:
        return {'ul': float('nan'), 'fitresult': None, 'ul_failed': True, 'error': f'coercion failed: {e}'}
    mom_mag = ak.Array(mom_np)
    try:
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
    except Exception as e:
        raise RuntimeError(f"Unbinned_fit_mom failed: {e}")

    # Use ResultsClass to produce an UpperLimit object and extract a numeric UL
    rc = ResultsClass(mom_mag, fitresult, verbose=verbose)
    # defensive check: ensure returned POI looks like a signal yield (e.g. "N_CE").
    poi_name = getattr(par, 'name', None)
    if poi_name is None or not (poi_name.startswith('N_') or 'CE' in poi_name.upper() or 'SIG' in poi_name.upper()):
        return {'ul': float('nan'), 'fitresult': fitresult, 'ul_failed': True, 'error': f'Returned POI "{poi_name}" does not look like a signal yield'}
    try:
        ul_obj = rc.GetUL(par, loss, nlls, combine_pdf, constraints, fit_range[0], fit_range[1], sig_yield=0, CL=0.90, opt='asym')
    except Exception:
        # If UL construction fails, return NaN and include the error for diagnostics
        import traceback
        err = traceback.format_exc()
        return {'ul': float('nan'), 'fitresult': fitresult, 'ul_failed': True, 'error': err}

    # Try to get a numeric upper limit from the returned object
    ul_value = None
    # try several common interfaces for upper-limit objects
    try:
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
    except Exception:
        ul_value = None
    if ul_value is None:
        # try to use a proxy (median of POI scan) if available
        try:
            ul_value = float(np.median(ul_obj.poinull.values))
        except Exception:
            ul_value = float('nan')

    return {'ul': float(ul_value), 'fitresult': fitresult, 'ul_obj': ul_obj}


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
    try:
        mom_np = _to_numeric_array(mom_sample, verbose=verbose)
        time_np = _to_numeric_array(time_sample, verbose=verbose)
    except Exception as e:
        return {'ul': float('nan'), 'fitresult': None, 'ul_failed': True, 'error': f'coercion failed: {e}'}
    mom_mag = ak.Array(mom_np)
    times = ak.Array(time_np)
    try:
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
    except Exception as e:
        raise RuntimeError(f"Unbinned_2d_fit_mom_time failed: {e}")

    fitresult = res
    # Use ResultsClass with flattened 1D mom sample as data
    rc = ResultsClass(mom_mag, fitresult, verbose=verbose)
    try:
        ul_obj = rc.GetUL(par, loss, [], combine_pdf, [], fit_range_mom[0], fit_range_mom[1], sig_yield=0, CL=0.90, opt='asym')
    except Exception:
        import traceback
        err = traceback.format_exc()
        return {'ul': float('nan'), 'fitresult': fitresult, 'ul_failed': True, 'error': err}

    ul_value = None
    try:
        ul_value = ul_obj.upperlimit(alpha=0.05, CLs=True)
    except Exception:
        ul_value = None
    if ul_value is None:
        try:
            ul_value = float(np.median(ul_obj.poinull.values))
        except Exception:
            ul_value = float('nan')

    return {'ul': float(ul_value), 'fitresult': fitresult, 'ul_obj': ul_obj}


def toy_scan_from_model(combine_pdf, par, fit_runner, mu_grid, ntoys=100, n_per_toy=1000, fit_runner_args=(), fit_runner_kwargs=None, verbose=0, plot_first_n=0, plot_dir=None):
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
    try:
        sampler = combine_pdf.create_sampler()
    except Exception:
        sampler = None

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
                                plt.xlabel('momentum')
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
                else:
                    try:
                        v = float(res)
                    except Exception:
                        v = float('nan')
                    if isinstance(v, float) and np.isnan(v):
                        err_msg = f'Non-numeric fit_runner return: {res!r}'
                        errors.append(err_msg)
                        if verbose:
                            print(f"Toy produced non-numeric result for mu={mu} toy={itoy}; recorded error: {err_msg}")
                        continue

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

        if verbose:
            print(f"mu={mu}: n_success={len(mu_vals)}, median={results[mu]['median']}")

    return results
