"""Control-region helpers focused on the Cosmic component.

This module provides a `ControlRegion` class which is initialised with
`mom_mag` (and optional `time`) arrays and exposes `fit_cosmic` that
fits an extended Chebyshev polynomial to the momentum distribution.

Design notes:
- Accepts numpy or awkward arrays (similar to `compare.py`).
- Uses zfit ExtendedUnbinnedNLL + `zfit.pdf.Chebyshev` to model Cosmic.
- Returns the zfit `FitResult` and a compact summary of fit parameters.

The implementation is intentionally focused and easy to extend for
additional components later.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import awkward as ak

try:
    import zfit
except Exception:
    zfit = None


def _to_numpy_flat(arr: any) -> np.ndarray:
    if isinstance(arr, ak.Array) or hasattr(arr, 'layout'):
        a = ak.to_numpy(ak.flatten(ak.drop_none(arr), axis=None))
    else:
        a = np.asarray(arr)
    if a.size:
        a = a[~np.isnan(a)]
    return a.astype(float)


class ControlRegion:
    """Encapsulate control-region arrays and fits.

    Parameters
    ----------
    mom_mag : array-like
        Reconstructed momentum magnitudes (numpy or awkward).
    times : array-like, optional
        Time values (kept for API compatibility / future use).
    verbose : int
        Verbosity level (0 = quiet).
    """

    def __init__(self, mom_mag, times=None, verbose: int = 0):
        self.mom_mag = mom_mag
        self.times = times
        self.verbose = verbose

    def fit_cosmic(self,
                   fit_range: Tuple[float, float] = (80.0, 150.0),
                   degree: int = 4,
                   init_N: Optional[float] = None,
                   plot: bool = False,
                   bins: int = 50) -> Dict[str, Any]:
        """Fit a Chebyshev polynomial (extended) to `mom_mag`.

        Returns a dictionary with keys: `result` (zfit FitResult),
        `params` (mapping of param -> value), `hesse` (errors), and
        `hist` (histogram arrays) if requested.
        """
        if zfit is None:
            raise RuntimeError('zfit is required for fitting (install zfit).')

        mom_np = _to_numpy_flat(self.mom_mag)
        lo, hi = float(fit_range[0]), float(fit_range[1])
        obs_mom = zfit.Space('x', limits=(lo, hi))

        # zfit Data
        data = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

        # Yield parameter
        N_init = init_N if init_N is not None else max(1.0, float(mom_np.size))
        N_Cosmic = zfit.Parameter('N_Cosmic', N_init, floating=True)

        # Chebyshev coefficients (c1..c_degree)
        coeffs = []
        for i in range(1, degree + 1):
            p = zfit.Parameter(f'c{i}', 0.0, -5.0, 5.0)
            coeffs.append(p)

        poly = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Cosmic)

        loss = zfit.loss.ExtendedUnbinnedNLL(model=poly, data=data)
        minimizer = zfit.minimize.Minuit()
        result = minimizer.minimize(loss=loss)

        # gather results
        params_out: Dict[str, float] = {}
        try:
            hesse = result.hesse()
        except Exception:
            hesse = {}

        params_out['N_Cosmic'] = float(result.params[N_Cosmic]['value']) if N_Cosmic in result.params else float(N_Cosmic.value())
        for p in coeffs:
            try:
                params_out[p.name] = float(result.params[p]['value'])
            except Exception:
                try:
                    params_out[p.name] = float(p.value())
                except Exception:
                    params_out[p.name] = None

        out: Dict[str, Any] = {
            'result': result,
            'params': params_out,
            'hesse': hesse,
            'model': poly,
        }

        # optional plotting / histogram diagnostics
        if plot:
            import matplotlib.pyplot as plt
            fit_range_vals = (lo, hi)
            bin_width = (hi - lo) / bins
            hist_vals, edges = np.histogram(mom_np, bins=bins, range=fit_range_vals)
            centers = 0.5 * (edges[:-1] + edges[1:])
            # evaluate model at bin centers
            eval_x = centers.reshape(-1, 1)
            model_vals = zfit.run(poly.pdf(eval_x) * params_out['N_Cosmic'] * bin_width)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
            ax1.hist(mom_np, bins=bins, range=fit_range_vals, histtype='step', label='data')
            ax1.plot(centers, model_vals.flatten(), '--', label='chebyshev fit')
            ax1.set_ylabel('Events')
            ax1.legend()

            ratio = hist_vals / (model_vals.flatten() + 1e-9)
            ax2.errorbar(centers, ratio, yerr=np.sqrt(hist_vals) / (model_vals.flatten() + 1e-9), fmt='.')
            ax2.axhline(1.0, color='gray', linestyle='--')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylabel('Data/Fit')

            plt.tight_layout()
            out['hist'] = (hist_vals, edges)
            out['figure'] = fig

        return out

    def fit_rpc(self,
                fit_range: Tuple[float, float] = (95.0, 115.0),
                init_mu: Optional[float] = None,
                init_sigma: Optional[float] = None,
                init_N: Optional[float] = None,
                plot: bool = False,
                bins: int = 50) -> Dict[str, Any]:
        """Fit a Gaussian (extended) to RPC momentum band.

        Returns a dictionary with keys: `result`, `mean`, `mean_err`,
        `sigma`, `sigma_err`, `norm`, and optional `figure`/`hist`.
        """
        if zfit is None:
            raise RuntimeError('zfit is required for fitting (install zfit).')

        mom_np = _to_numpy_flat(self.mom_mag)
        lo, hi = float(fit_range[0]), float(fit_range[1])
        obs_mom = zfit.Space('x', limits=(lo, hi))

        # initial guesses
        mu0 = init_mu if init_mu is not None else (float(np.mean(mom_np)) if mom_np.size else 100.0)
        sig0 = init_sigma if init_sigma is not None else (float(np.std(mom_np)) if mom_np.size else 3.0)
        N0 = init_N if init_N is not None else max(1.0, float(mom_np.size))

        mu = zfit.Parameter('mu_rpc', mu0, mu0 - 10.0, mu0 + 10.0)
        sigma = zfit.Parameter('sigma_rpc', max(0.1, sig0), 0.01, max(0.1, (hi - lo)))
        N_RPC = zfit.Parameter('N_RPC', N0, max(0.0, N0 * 0.1), max(1.0, N0 * 10.0))

        gauss = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N_RPC)
        data = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)

        loss = zfit.loss.ExtendedUnbinnedNLL(model=gauss, data=data)
        minimizer = zfit.minimize.Minuit()
        result = minimizer.minimize(loss=loss)

        try:
            hesse = result.hesse()
        except Exception:
            hesse = {}

        out: Dict[str, Any] = {
            'result': result,
            'mean': float(result.params[mu]['value']) if mu in result.params else float(mu.value()),
            'mean_err': float(hesse[mu]['error']) if mu in hesse else None,
            'sigma': float(result.params[sigma]['value']) if sigma in result.params else float(sigma.value()),
            'sigma_err': float(hesse[sigma]['error']) if sigma in hesse else None,
            'norm': float(result.params[N_RPC]['value']) if N_RPC in result.params else float(N_RPC.value()),
            'hesse': hesse,
            'model': gauss,
        }

        if plot:
            import matplotlib.pyplot as plt
            fit_range_vals = (lo, hi)
            bin_width = (hi - lo) / bins
            hist_vals, edges = np.histogram(mom_np, bins=bins, range=fit_range_vals)
            centers = 0.5 * (edges[:-1] + edges[1:])
            eval_x = centers.reshape(-1, 1)
            model_vals = zfit.run(gauss.pdf(eval_x) * out['norm'] * bin_width)

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))
            ax1.hist(mom_np, bins=bins, range=fit_range_vals, histtype='step', label='data')
            ax1.plot(centers, model_vals.flatten(), '--', label='gaussian fit')
            ax1.set_ylabel('Events')
            ax1.legend()

            ratio = hist_vals / (model_vals.flatten() + 1e-9)
            ax2.errorbar(centers, ratio, yerr=np.sqrt(hist_vals) / (model_vals.flatten() + 1e-9), fmt='.')
            ax2.axhline(1.0, color='gray', linestyle='--')
            ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
            ax2.set_ylabel('Data/Fit')

            plt.tight_layout()
            out['hist'] = (hist_vals, edges)
            out['figure'] = fig

        return out


__all__ = ['ControlRegion']


