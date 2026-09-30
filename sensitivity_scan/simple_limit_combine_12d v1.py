"""
Frequentist 1D/2D (Momentum & Optional Time) Asymptotic CLs Limit Setter matching ROOT Combine.

- Select fit dimension on command line (--dim 1d or --dim 2d).
- Dynamically loads observables and yields from YAML datacard.
- Scales POI as signal strength mu (N_CE = mu * reference_rate).
- Implements Log-Normal systematic penalty constraints matching Combine's lnN.
- Uses exact Cowan et al. (2011) asymptotic formulae for 90% CLs upper limits.

Usage:
    python fit_combine_match_cls.py --datacard fit.card.yaml --dim 1d --toys 1000 --seed 42
    python fit_combine_match_cls.py --datacard fit.card.yaml --dim 2d --toys 1000 --seed 42
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import zfit
import tensorflow as tf

import matplotlib.pyplot as plt
import argparse
import logging
import scipy.optimize as opt
import scipy.stats as stats
import multiprocessing as mp
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys_path_parent = Path(__file__).parent.parent
if str(sys_path_parent) not in sys.path:
    sys.path.insert(0, str(sys_path_parent))

from custom_models import poly58
from datacard import DataCard

# ==============================================================================
# CUSTOM PDF DEFINITIONS
# ==============================================================================

class GammaPolyHybrid(zfit.pdf.ZPDF):
    """RMC background PDF: (x-x0)^alpha * (120-x)^beta * exp(-lambda*x)"""
    _PARAMS = ["x0", "alpha", "beta", "lam"]
    
    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        dtype = x.dtype
        
        x0 = tf.cast(self.params["x0"], dtype)
        alpha = tf.cast(self.params["alpha"], dtype)
        beta = tf.cast(self.params["beta"], dtype)
        lam = tf.cast(self.params["lam"], dtype)
        
        epsilon = tf.cast(1e-8, dtype)
        upper_limit = tf.cast(120.0, dtype)
        
        turn_on = tf.pow(tf.maximum(epsilon, x - x0), alpha)
        bulk = tf.pow(tf.maximum(epsilon, upper_limit - x), beta)
        decay = tf.exp(-lam * x)
        
        return turn_on * bulk * decay


# ==============================================================================
# DATACARD LOADER AND MODEL BUILDER (1D AND 2D)
# ==============================================================================

def rebuild_combine_model_from_datacard(datacard_path, dim='2d'):
    """Reconstructs 1D (Momentum) or 2D (Momentum x Time) zfit model from YAML datacard."""
    card = DataCard.from_yaml(datacard_path)
    
    # Extract observable spaces
    obs_ranges = card.observables
    fit_range_mom = tuple(obs_ranges['mom'])
    obs_mom = zfit.Space('mom', limits=fit_range_mom)
    
    if dim == '2d':
        fit_range_time = tuple(obs_ranges['time'])
        obs_time = zfit.Space('time', limits=fit_range_time)
        obs_fit = obs_mom * obs_time
    else:
        obs_fit = obs_mom
    
    # Extract yields
    yields_dict = card.get_expected_yields()
    ref_signal_rate = yields_dict['CE']
    n_dio_nom = yields_dict['DIO']
    n_cosmic_nom = yields_dict['Cosmic']
    
    if 'RPC' in yields_dict:
        n_rpc_nom = yields_dict['RPC']
    else:
        n_rpc_nom = yields_dict.get('rpc_ext', 0.0) + yields_dict.get('rpc_int', 0.0)
    
    if 'RMC' in yields_dict:
        n_rmc_nom = yields_dict['RMC']
    else:
        n_rmc_nom = 0.0
        
    logger.info(f"[Datacard {dim.upper()}] Ref Signal Rate: {ref_signal_rate:.4f}")
    logger.info(f"[Datacard {dim.upper()}] Yields -> Cosmic: {n_cosmic_nom:.2f} | DIO: {n_dio_nom:.2f} | RPC: {n_rpc_nom:.2f} | RMC: {n_rmc_nom:.2f}")

    # POI is Signal Strength mu (N_CE = mu * ref_signal_rate)
    poi_mu = zfit.Parameter('mu', 1.0, lower=-10.0, upper=100.0)
    
    yield_dio = zfit.Parameter('N_DIO', n_dio_nom, lower=0, upper=5000)
    yield_cosmic = zfit.Parameter('N_Cosmic', n_cosmic_nom, lower=0, upper=5000)
    yield_rpc = zfit.Parameter('N_RPC', n_rpc_nom, lower=0, upper=1000)
    yield_rmc = zfit.Parameter('N_RMC', n_rmc_nom, lower=0, upper=1000)
    
    # Extract Shape Parameters
    ce_shapes = card.get_process_shape_info('CE')['params']
    cosmic_shapes = card.get_process_shape_info('Cosmic')['params']
    rpc_shapes = card.get_process_shape_info(
        'RPC' if 'RPC' in card.processes else 'rpc_ext'
    )['params']
    dio_shapes = card.get_process_shape_info('DIO')['params']
    rmc_shapes = card.get_process_shape_info('RMC')['params'] if 'RMC' in card.processes else {}
    
    # 1D PDFs - Momentum
    cosmic_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[cosmic_shapes['c1'], cosmic_shapes['c2']])
    rpc_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[rpc_shapes['c1'], rpc_shapes['c2']])
    
    dio_a5 = zfit.Parameter('a5_DIO', dio_shapes['a5'], floating=False)
    dio_a6 = zfit.Parameter('a6_DIO', dio_shapes['a6'], floating=False)
    dio_a7 = zfit.Parameter('a7_DIO', dio_shapes['a7'], floating=False)
    dio_a8 = zfit.Parameter('a8_DIO', dio_shapes['a8'], floating=False)
    dio_mom_pdf = poly58(obs=obs_mom, a5=dio_a5, a6=dio_a6, a7=dio_a7, a8=dio_a8)
    
    signal_mom_pdf = zfit.pdf.Gauss(mu=ce_shapes['mu'], sigma=ce_shapes['sigma'], obs=obs_mom)
    
    # RMC PDF - GammaPolyHybrid
    rmc_x0 = zfit.Parameter('rmc_x0', rmc_shapes.get('x0', 90.0), floating=False)
    rmc_alpha = zfit.Parameter('rmc_alpha', rmc_shapes.get('alpha', 2.0), floating=False)
    rmc_beta = zfit.Parameter('rmc_beta', rmc_shapes.get('beta', 1.0), floating=False)
    rmc_lam = zfit.Parameter('rmc_lambda', rmc_shapes.get('lam', 0.5), floating=False)
    rmc_mom_pdf = GammaPolyHybrid(obs=obs_mom, x0=rmc_x0, alpha=rmc_alpha, beta=rmc_beta, lam=rmc_lam)
    
    if dim == '1d':
        cosmic_pdf = cosmic_mom_pdf
        rpc_pdf = rpc_mom_pdf
        dio_pdf = dio_mom_pdf
        rmc_pdf = rmc_mom_pdf
        signal_pdf = signal_mom_pdf
    else:
        # 1D PDFs - Time
        tau_dio = zfit.Parameter('tau_DIO', dio_shapes.get('tau', -864.0), floating=False)
        tau_rpc = zfit.Parameter('tau_RPC', rpc_shapes.get('tau', -25.0), floating=False)
        
        cosmic_time_pdf = zfit.pdf.Uniform(obs=obs_time, low=fit_range_time[0], high=fit_range_time[1])
        dio_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_dio, obs=obs_time)
        rpc_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_rpc, obs=obs_time)
        signal_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_dio, obs=obs_time) # CE follows DIO timing structure
        
        # 2D Product PDFs
        cosmic_pdf = zfit.pdf.ProductPDF([cosmic_mom_pdf, cosmic_time_pdf])
        rpc_pdf = zfit.pdf.ProductPDF([rpc_mom_pdf, rpc_time_pdf])
        dio_pdf = zfit.pdf.ProductPDF([dio_mom_pdf, dio_time_pdf])
        rmc_pdf = zfit.pdf.ProductPDF([rmc_mom_pdf, cosmic_time_pdf])  # RMC uses Cosmic timing
        signal_pdf = zfit.pdf.ProductPDF([signal_mom_pdf, signal_time_pdf])
    
    # Extended PDFs
    signal_yield_func = zfit.ComposedParameter('signal_yield', lambda m: m * ref_signal_rate, params=[poi_mu])
    
    dio_ext = dio_pdf.create_extended(yield_dio)
    cosmic_ext = cosmic_pdf.create_extended(yield_cosmic)
    rpc_ext = rpc_pdf.create_extended(yield_rpc)
    rmc_ext = rmc_pdf.create_extended(yield_rmc)
    signal_ext = signal_pdf.create_extended(signal_yield_func)
    
    combined_pdf = zfit.pdf.SumPDF([dio_ext, cosmic_ext, rpc_ext, rmc_ext, signal_ext])
    
    nominals = {
        'ref_signal_rate': ref_signal_rate,
        'dio': n_dio_nom,
        'cosmic': n_cosmic_nom,
        'rpc': n_rpc_nom,
        'rmc': n_rmc_nom
    }
    
    return combined_pdf, poi_mu, obs_fit, nominals, card


# ==============================================================================
# LOG-NORMAL CONSTRAINT HELPER
# ==============================================================================

def build_lognormal_constraint(param, nominal_val, kappa_uncertainty_factor):
    """
    Log-Normal constraint penalty term matching Combine's 'lnN'.
    
    Args:
        param: zfit Parameter to constrain
        nominal_val: Nominal value
        kappa_uncertainty_factor: Uncertainty factor (e.g., 1.20 for 20% uncertainty)
                                  Defines range as [1/kappa, kappa]
    """
    sigma_log = np.log(kappa_uncertainty_factor)
    
    def log_normal_nll():
        val = param
        ratio = val / nominal_val
        safe_ratio = tf.maximum(ratio, 1e-6)
        log_r = tf.math.log(safe_ratio)
        return 0.5 * tf.square(log_r / sigma_log)
    
    return zfit.constraint.SimpleConstraint(log_normal_nll, params=[param])


# ==============================================================================
# MULTIPROCESSING WORKER FUNCTIONS FOR PARALLEL TOYS
# ==============================================================================

_WORKER_CONTEXT = {}

def _worker_init(datacard_path, dim, freeze_nuisances, cl, verbosity):
    """Initializer for multiprocessing pool workers: builds zfit model once per worker process."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    combined_pdf, poi_mu, obs_fit, nominals, card = rebuild_combine_model_from_datacard(datacard_path, dim=dim)
    target_cls = 1.0 - cl

    model_params = {p.name: p for p in combined_pdf.get_params()}
    yield_cosmic = model_params['N_Cosmic']
    yield_dio = model_params['N_DIO']
    yield_rpc = model_params['N_RPC']
    yield_rmc = model_params.get('N_RMC')

    if freeze_nuisances:
        constraints = []
    else:
        kappa_cosmic = 1.20
        kappa_dio = 1.025
        kappa_rpc = 1.27
        kappa_rmc = 1.0

        if hasattr(card, 'systematics') and card.systematics:
            for syst_name, syst_info in card.systematics.items():
                if isinstance(syst_info, dict) and 'effects' in syst_info:
                    effects = syst_info['effects']
                    if 'Cosmic' in effects: kappa_cosmic = effects['Cosmic']
                    if 'DIO' in effects: kappa_dio = effects['DIO']
                    if 'RPC' in effects: kappa_rpc = effects['RPC']
                    if 'RMC' in effects: kappa_rmc = effects['RMC']

        constraints = [
            build_lognormal_constraint(yield_cosmic, nominals['cosmic'], kappa_cosmic),
            build_lognormal_constraint(yield_dio, nominals['dio'], kappa_dio),
            build_lognormal_constraint(yield_rpc, nominals['rpc'], kappa_rpc)
        ]
        if yield_rmc is not None and nominals['rmc'] > 0:
            constraints.append(build_lognormal_constraint(yield_rmc, nominals['rmc'], kappa_rmc))

    yield_dio.set_value(nominals['dio'])
    yield_cosmic.set_value(nominals['cosmic'])
    yield_rpc.set_value(nominals['rpc'])
    if yield_rmc is not None:
        yield_rmc.set_value(nominals['rmc'])
    poi_mu.set_value(0.0)

    initial_parameter_values = {}
    for parameter in combined_pdf.get_params():
        if not hasattr(parameter, 'set_value') or not hasattr(parameter, 'value'):
            continue
        try:
            initial_parameter_values[parameter] = float(parameter.value())
        except (TypeError, ValueError):
            continue

    sampler = combined_pdf.create_sampler()

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        'combined_pdf': combined_pdf,
        'poi_mu': poi_mu,
        'obs_fit': obs_fit,
        'nominals': nominals,
        'constraints': constraints,
        'target_cls': target_cls,
        'initial_parameter_values': initial_parameter_values,
        'sampler': sampler,
        'verbosity': verbosity,
        'freeze_nuisances': freeze_nuisances
    }

def _worker_run_toy(args_tuple):
    """Executes a single asymptotic CLs toy evaluation inside a worker process."""
    toy_idx, seed = args_tuple

    if seed is not None:
        toy_seed = int((seed + toy_idx * 10007) % (2**31 - 1))
        tf.random.set_seed(toy_seed)
        np.random.seed(toy_seed)

    ctx = _WORKER_CONTEXT
    combined_pdf = ctx['combined_pdf']
    poi_mu = ctx['poi_mu']
    obs_fit = ctx['obs_fit']
    nominals = ctx['nominals']
    constraints = ctx['constraints']
    target_cls = ctx['target_cls']
    initial_parameter_values = ctx['initial_parameter_values']
    sampler = ctx['sampler']
    verbosity = ctx['verbosity']
    freeze_nuisances = ctx['freeze_nuisances']

    sampler.resample({poi_mu: 0.0})
    raw_sample = sampler.numpy()

    if np.any(np.isnan(raw_sample)):
        return toy_idx, np.nan, np.nan, 'nan'

    for parameter, value in initial_parameter_values.items():
        try:
            parameter.set_value(value)
        except Exception:
            pass
    poi_mu.set_value(0.0)
    poi_mu.floating = True

    minimizer = zfit.minimize.Minuit()
    data_zfit = zfit.Data.from_numpy(array=raw_sample, obs=obs_fit)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combined_pdf, data=data_zfit, constraints=constraints)

    poi_mu.lower = -10.0
    poi_mu.upper = 100.0
    poi_mu.floating = True

    try:
        global_fit = minimizer.minimize(loss)
        if not global_fit.valid:
            if not freeze_nuisances:
                loss_frozen = zfit.loss.ExtendedUnbinnedNLL(model=combined_pdf, data=data_zfit, constraints=[])
                global_fit = minimizer.minimize(loss_frozen)
                if not global_fit.valid:
                    return toy_idx, np.nan, np.nan, 'fit_failure'
            else:
                return toy_idx, np.nan, np.nan, 'fit_failure'

        nll_best = float(global_fit.fmin)
        mu_hat = float(global_fit.params[poi_mu]['value'])

        try:
            global_fit.hesse()
            sigma_mu = float(global_fit.params[poi_mu]['errors']['hesse']['error'])
        except Exception:
            sigma_mu = np.sqrt(max(1.0, (mu_hat * nominals['ref_signal_rate']) + nominals['cosmic'] + nominals['dio'] + nominals['rpc'])) / nominals['ref_signal_rate']
    except Exception:
        return toy_idx, np.nan, np.nan, 'fit_failure'

    def compute_asymptotic_cls(mu_test):
        if mu_test <= mu_hat:
            return 1.0

        poi_mu.set_value(mu_test)
        poi_mu.floating = False

        try:
            profile_fit = minimizer.minimize(loss)
            if not profile_fit.valid:
                return 1.0
            nll_profiled = float(profile_fit.fmin)
        except Exception:
            return 1.0
        finally:
            poi_mu.floating = True

        q_mu = max(0.0, 2.0 * (nll_profiled - nll_best))
        sqrt_q_mu = np.sqrt(q_mu)

        sqrt_q_mu_A = mu_test / sigma_mu if sigma_mu > 0 else sqrt_q_mu

        if sqrt_q_mu <= sqrt_q_mu_A:
            p_sb = 1.0 - stats.norm.cdf(sqrt_q_mu)
            one_minus_pb = stats.norm.cdf(sqrt_q_mu_A - sqrt_q_mu)
        else:
            p_sb = 1.0 - stats.norm.cdf((q_mu + sqrt_q_mu_A**2) / (2.0 * sqrt_q_mu_A))
            one_minus_pb = 1.0 - stats.norm.cdf((q_mu - sqrt_q_mu_A**2) / (2.0 * sqrt_q_mu_A))

        if one_minus_pb <= 1e-9:
            return 1.0

        return p_sb / one_minus_pb

    def root_func(mu_test):
        return compute_asymptotic_cls(mu_test) - target_cls

    left_bracket = max(0.1, mu_hat)
    right_bracket = max(10.0, mu_hat + 30.0)

    try:
        while root_func(right_bracket) > 0:
            right_bracket += 20.0
            if right_bracket > 300.0:
                return toy_idx, np.nan, np.nan, 'root_failure'

        mu_limit = opt.brentq(root_func, left_bracket, right_bracket, xtol=0.05)
        n_ce_limit = mu_limit * nominals['ref_signal_rate']
        return toy_idx, n_ce_limit, mu_limit, 'success'
    except Exception:
        return toy_idx, np.nan, np.nan, 'root_failure'
    finally:
        poi_mu.floating = True


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='1D/2D Datacard-driven Asymptotic CLs Limit Setter')
    parser.add_argument('--datacard', required=True, help='Path to YAML datacard')
    parser.add_argument('--dim', choices=['1d', '2d'], default='2d', help='Fit dimension: 1d (momentum) or 2d (momentum + time)')
    parser.add_argument('--toys', type=int, default=1000, help='Number of toy experiments')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='Number of parallel worker processes (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cl', type=float, default=0.90, help='Confidence Level (default 0.90)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--freeze-nuisances', action='store_true', help='Freeze nuisance parameters (stat only)')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for debugging failed fits')
    args = parser.parse_args()

    n_ce_limits = []
    mu_limits = []
    n_converged = 0
    n_failed = 0
    n_nan_samples = 0
    n_fit_failures = 0

    logger.info(f"Running {args.toys} background-only {args.dim.upper()} toys from datacard: {args.datacard} using {args.jobs} process(es)...")

    toy_tasks = [(i, args.seed + i if args.seed is not None else None) for i in range(args.toys)]

    if args.jobs > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=args.jobs, initializer=_worker_init, initargs=(args.datacard, args.dim, args.freeze_nuisances, args.cl, args.verbosity)) as pool:
            for idx, (toy_i, n_ce, mu, status) in enumerate(pool.imap_unordered(_worker_run_toy, toy_tasks)):
                if status == 'success':
                    n_ce_limits.append(n_ce)
                    mu_limits.append(mu)
                    n_converged += 1
                elif status == 'nan':
                    n_nan_samples += 1
                    n_failed += 1
                else:
                    n_fit_failures += 1
                    n_failed += 1

                if (idx + 1) % max(1, args.toys // 10) == 0:
                    logger.info(f"Progress: {idx + 1} / {args.toys} toys completed... ({n_converged} converged, {n_failed} failed)")
    else:
        _worker_init(args.datacard, args.dim, args.freeze_nuisances, args.cl, args.verbosity)
        for idx, task in enumerate(toy_tasks):
            toy_i, n_ce, mu, status = _worker_run_toy(task)
            if status == 'success':
                n_ce_limits.append(n_ce)
                mu_limits.append(mu)
                n_converged += 1
            elif status == 'nan':
                n_nan_samples += 1
                n_failed += 1
            else:
                n_fit_failures += 1
                n_failed += 1

            if (idx + 1) % max(1, args.toys // 10) == 0:
                logger.info(f"Progress: {idx + 1} / {args.toys} toys completed... ({n_converged} converged, {n_failed} failed)")

    n_ce_limits = np.array(n_ce_limits)
    mu_limits = np.array(mu_limits)

    # Safety check: handle case where all toys failed
    if len(n_ce_limits) == 0:
        logger.error("\n" + "="*60)
        logger.error("CRITICAL ERROR: ALL TOYS FAILED!")
        logger.error("="*60)
        logger.error(f"NaN Samples:  {n_nan_samples}")
        logger.error(f"Fit Failures: {n_fit_failures}")
        logger.error("\nDiagnostic suggestions:")
        logger.error("  1. Try: --freeze-nuisances (disable all constraints)")
        logger.error("  2. Try: --dim 1d (switch to 1D momentum-only)")
        logger.error("  3. Try: --verbosity 2 (enable detailed debugging)")
        logger.error("  4. Check datacard file for invalid parameter values")
        logger.error("  5. Verify toy sample sizes are reasonable (check expected yields)")
        logger.error("="*60)
        return

    med_n_ce = np.percentile(n_ce_limits, 50.0)
    med_mu = np.percentile(mu_limits, 50.0)
    p16_n_ce = np.percentile(n_ce_limits, 15.87)
    p84_n_ce = np.percentile(n_ce_limits, 84.13)

    logger.info("\n" + "="*60)
    logger.info(f"{args.dim.upper()} DATACARD ASYMPTOTIC CLs RESULTS")
    logger.info("="*60)
    logger.info(f"Toys Requested:       {args.toys}")
    logger.info(f"Toys Converged:       {n_converged} ({100.0*n_converged/args.toys:.1f}%)")
    logger.info(f"Toys Failed:          {n_failed} ({100.0*n_failed/args.toys:.1f}%)")
    logger.info(f"  → NaN Samples:      {n_nan_samples}")
    logger.info(f"  → Fit Failures:     {n_fit_failures}")
    logger.info(f"Median Expected Limit (Signal Strength mu): {med_mu:.2f}")
    logger.info(f"Median Expected Limit (Events N_CE):        {med_n_ce:.2f}")
    logger.info(f"68% Band (Events N_CE):                     [{p16_n_ce:.2f}, {p84_n_ce:.2f}]")
    logger.info("="*60)

    if n_converged < args.toys * 0.5:
        logger.error(f"WARNING: Less than 50% convergence rate ({n_converged}/{args.toys})")
        logger.error("  → Try: --freeze-nuisances to disable constraints")
        logger.error("  → Try: reducing toy count for quick diagnostics")
        logger.error("  → Try: --verbosity 2 to see detailed fit failures")

    # Summary Plot
    plt.figure(figsize=(9, 6))
    plt.hist(n_ce_limits, bins=35, color='mediumpurple', alpha=0.5, edgecolor='indigo', label=f'{args.dim.upper()} Toy Limits')
    plt.axvline(med_n_ce, color='black', linestyle='--', linewidth=2, label=f'Median: {med_n_ce:.2f} events')
    plt.axvspan(p16_n_ce, p84_n_ce, color='green', alpha=0.2, label='1-Sigma Band')

    plt.xlabel('90% CLs Upper Limit on Signal Yield ($N_{CE}$)', fontsize=11)
    plt.ylabel('Toys / Bin', fontsize=11)
    plt.title(f'{args.dim.upper()} Asymptotic $CL_s$ Sensitivity from YAML Datacard', fontsize=12, weight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_plot = out_dir / f"datacard_{args.dim}_cls_limits.png"
    plt.savefig(out_plot, dpi=150)
    logger.info(f"Saved plot to {out_plot}")

if __name__ == '__main__':
    main()
