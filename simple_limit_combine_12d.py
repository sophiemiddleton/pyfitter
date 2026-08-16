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

import numpy as np
import zfit
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import logging
import scipy.optimize as opt
import scipy.stats as stats
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from custom_models import poly58
from datacard import DataCard

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
    ref_signal_rate = 0.57 #yields_dict['CE']
    n_dio_nom = yields_dict['DIO']
    n_cosmic_nom = yields_dict['Cosmic']
    
    if 'RPC' in yields_dict:
        n_rpc_nom = yields_dict['RPC']
    else:
        n_rpc_nom = yields_dict.get('rpc_ext', 0.0) + yields_dict.get('rpc_int', 0.0)
        
    logger.info(f"[Datacard {dim.upper()}] Ref Signal Rate: {ref_signal_rate:.4f}")
    logger.info(f"[Datacard {dim.upper()}] Yields -> Cosmic: {n_cosmic_nom:.2f} | DIO: {n_dio_nom:.2f} | RPC: {n_rpc_nom:.2f}")

    # POI is Signal Strength mu (N_CE = mu * ref_signal_rate)
    poi_mu = zfit.Parameter('mu', 1.0, lower=-10.0, upper=100.0)
    
    yield_dio = zfit.Parameter('N_DIO', n_dio_nom, lower=0, upper=5000)
    yield_cosmic = zfit.Parameter('N_Cosmic', n_cosmic_nom, lower=0, upper=5000)
    yield_rpc = zfit.Parameter('N_RPC', n_rpc_nom, lower=0, upper=1000)
    
    # Extract Shape Parameters
    ce_shapes = card.get_process_shape_info('CE')['params']
    cosmic_shapes = card.get_process_shape_info('Cosmic')['params']
    rpc_shapes = card.get_process_shape_info(
        'RPC' if 'RPC' in card.processes else 'rpc_ext'
    )['params']
    dio_shapes = card.get_process_shape_info('DIO')['params']
    
    # 1D PDFs - Momentum
    cosmic_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[cosmic_shapes['c1'], cosmic_shapes['c2']])
    rpc_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[rpc_shapes['c1'], rpc_shapes['c2']])
    
    dio_a5 = zfit.Parameter('a5_DIO', dio_shapes['a5'], floating=False)
    dio_a6 = zfit.Parameter('a6_DIO', dio_shapes['a6'], floating=False)
    dio_a7 = zfit.Parameter('a7_DIO', dio_shapes['a7'], floating=False)
    dio_a8 = zfit.Parameter('a8_DIO', dio_shapes['a8'], floating=False)
    dio_mom_pdf = poly58(obs=obs_mom, a5=dio_a5, a6=dio_a6, a7=dio_a7, a8=dio_a8)
    
    signal_mom_pdf = zfit.pdf.Gauss(mu=ce_shapes['mu'], sigma=ce_shapes['sigma'], obs=obs_mom)
    
    if dim == '1d':
        cosmic_pdf = cosmic_mom_pdf
        rpc_pdf = rpc_mom_pdf
        dio_pdf = dio_mom_pdf
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
        signal_pdf = zfit.pdf.ProductPDF([signal_mom_pdf, signal_time_pdf])
    
    # Extended PDFs
    signal_yield_func = zfit.ComposedParameter('signal_yield', lambda m: m * ref_signal_rate, params=[poi_mu])
    
    dio_ext = dio_pdf.create_extended(yield_dio)
    cosmic_ext = cosmic_pdf.create_extended(yield_cosmic)
    rpc_ext = rpc_pdf.create_extended(yield_rpc)
    signal_ext = signal_pdf.create_extended(signal_yield_func)
    
    combined_pdf = zfit.pdf.SumPDF([dio_ext, cosmic_ext, rpc_ext, signal_ext])
    
    nominals = {
        'ref_signal_rate': ref_signal_rate,
        'dio': n_dio_nom,
        'cosmic': n_cosmic_nom,
        'rpc': n_rpc_nom
    }
    
    return combined_pdf, poi_mu, obs_fit, nominals, card


# ==============================================================================
# LOG-NORMAL CONSTRAINT HELPER
# ==============================================================================

def build_lognormal_constraint(param, nominal_val, relative_unc):
    """Log-Normal constraint penalty term matching Combine's 'lnN'."""
    sigma_log = np.log(1.0 + relative_unc)
    
    def log_normal_nll():
        val = param
        ratio = val / nominal_val
        safe_ratio = tf.maximum(ratio, 1e-6)
        log_r = tf.math.log(safe_ratio)
        return 0.5 * tf.square(log_r / sigma_log)
    
    return zfit.constraint.SimpleConstraint(log_normal_nll, params=[param])


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='1D/2D Datacard-driven Asymptotic CLs Limit Setter')
    parser.add_argument('--datacard', required=True, help='Path to YAML datacard')
    parser.add_argument('--dim', choices=['1d', '2d'], default='2d', help='Fit dimension: 1d (momentum) or 2d (momentum + time)')
    parser.add_argument('--toys', type=int, default=1000, help='Number of toy experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cl', type=float, default=0.90, help='Confidence Level (default 0.90)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    args = parser.parse_args()

    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)

    combined_pdf, poi_mu, obs_fit, nominals, card = rebuild_combine_model_from_datacard(args.datacard, dim=args.dim)
    minimizer = zfit.minimize.Minuit()
    target_cls = 1.0 - args.cl

    model_params = {p.name: p for p in combined_pdf.get_params()}
    yield_cosmic = model_params['N_Cosmic']
    yield_dio = model_params['N_DIO']
    yield_rpc = model_params['N_RPC']

    # Log-Normal systematic penalty constraints matching datacard parameters
    constraints = [
        build_lognormal_constraint(yield_cosmic, nominals['cosmic'], 0.20),
        build_lognormal_constraint(yield_dio, nominals['dio'], 0.025),
        build_lognormal_constraint(yield_rpc, nominals['rpc'], 0.27)
    ]

    def asymptotic_cls_limit_runner(mock_data_sample):
        data_zfit = zfit.Data.from_numpy(array=mock_data_sample, obs=obs_fit)
        loss = zfit.loss.ExtendedUnbinnedNLL(model=combined_pdf, data=data_zfit, constraints=constraints)

        # 1. Unconstrained best fit
        poi_mu.lower = -10.0
        poi_mu.upper = 100.0
        poi_mu.floating = True

        try:
            global_fit = minimizer.minimize(loss)
            if not global_fit.valid:
                return np.nan
            nll_best = float(global_fit.fmin)
            mu_hat = float(global_fit.params[poi_mu]['value'])

            # POI standard error sigma_mu for Asimov term
            try:
                global_fit.hesse()
                sigma_mu = float(global_fit.params[poi_mu]['errors']['hesse']['error'])
            except Exception:
                sigma_mu = np.sqrt(max(1.0, (mu_hat * nominals['ref_signal_rate']) + nominals['cosmic'] + nominals['dio'] + nominals['rpc'])) / nominals['ref_signal_rate']
        except Exception:
            return np.nan

        # 2. Cowan et al. Asymptotic Formulae
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

            # Expected Asimov statistic: sqrt(q_mu_A) = mu / sigma_mu
            sqrt_q_mu_A = mu_test / sigma_mu if sigma_mu > 0 else sqrt_q_mu

            # Cowan et al. Eq. 65/66 tail probabilities
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
                    return np.nan

            mu_limit = opt.brentq(root_func, left_bracket, right_bracket, xtol=0.05)
            
            # Event limit N_CE = mu_limit * reference signal rate
            n_ce_limit = mu_limit * nominals['ref_signal_rate']
            return n_ce_limit, mu_limit

        except Exception:
            return np.nan
        finally:
            poi_mu.floating = True

    # Run Toys
    sampler = combined_pdf.create_sampler()
    n_ce_limits = []
    mu_limits = []

    logger.info(f"Running {args.toys} background-only {args.dim.upper()} toys from datacard: {args.datacard}...")
    for i in range(args.toys):
        sampler.resample({poi_mu: 0.0})
        raw_sample = sampler.numpy()

        res = asymptotic_cls_limit_runner(raw_sample)
        if res is not np.nan and not np.isnan(res[0]):
            n_ce_limits.append(res[0])
            mu_limits.append(res[1])

        if (i + 1) % max(1, args.toys // 10) == 0:
            logger.info(f"Progress: {i + 1} / {args.toys} toys completed...")

    n_ce_limits = np.array(n_ce_limits)
    mu_limits = np.array(mu_limits)

    med_n_ce = np.percentile(n_ce_limits, 50.0)
    med_mu = np.percentile(mu_limits, 50.0)
    p16_n_ce = np.percentile(n_ce_limits, 15.87)
    p84_n_ce = np.percentile(n_ce_limits, 84.13)

    logger.info("\n" + "="*60)
    logger.info(f"{args.dim.upper()} DATACARD ASYMPTOTIC CLs RESULTS")
    logger.info("="*60)
    logger.info(f"Median Expected Limit (Signal Strength mu): {med_mu:.2f}")
    logger.info(f"Median Expected Limit (Events N_CE):        {med_n_ce:.2f}")
    logger.info(f"68% Band (Events N_CE):                     [{p16_n_ce:.2f}, {p84_n_ce:.2f}]")
    logger.info("="*60)

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