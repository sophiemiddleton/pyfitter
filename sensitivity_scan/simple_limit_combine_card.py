"""Card-driven Combine-style asymptotic and empirical CLs sensitivity runner.

This version strictly follows CMS Combine / LHC Higgs PAG conventions:
- Test statistic q_mu is defined as 2 * (NLL(mu) - NLL(mu_hat)) for mu_hat <= mu, else 0.
- Empirical CLs evaluates CL_sb = P(q >= q_obs | S+B) and CL_b = P(q >= q_obs | B).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU to avoid multiprocessing memory contention

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import gc

import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import tensorflow as tf
import zfit
import matplotlib.pyplot as plt

tf.config.run_functions_eagerly(False)

from simple_limit_combine_12d import (
    build_gaussian_constraint,
    rebuild_combine_model_from_datacard,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONTEXT = {}


def initialize_worker(datacard, dim, freeze_nuisances, cl, n_toys=None, n_scan_points=None, seed=None):
    model, poi, obs, nominals, card = rebuild_combine_model_from_datacard(datacard, dim)
    logger.info('Using datacard: %s', Path(datacard).resolve())
    logger.info('Card yields: %s', card.get_expected_yields())
    constraints = [] if freeze_nuisances else [
        build_gaussian_constraint(parameter)
        for parameter in model.get_params()
        if parameter.name.startswith('theta_')
    ]
    initial_values = {}
    for parameter in model.get_params():
        if hasattr(parameter, 'set_value') and hasattr(parameter, 'value'):
            try:
                initial_values[parameter] = float(parameter.value())
            except (TypeError, ValueError):
                pass
    poi.set_value(0.0)
    sampler = model.create_sampler()
    _CONTEXT.clear()
    _CONTEXT.update(
        model=model,
        poi=poi,
        obs=obs,
        nominals=nominals,
        constraints=constraints,
        initial_values=initial_values,
        sampler=sampler,
        target=1.0 - cl,
        n_toys=n_toys,
        n_scan_points=n_scan_points,
        seed=seed,
    )


def reset_parameters():
    for parameter, value in _CONTEXT['initial_values'].items():
        try:
            parameter.set_value(value)
        except Exception:
            pass
    poi = _CONTEXT['poi']
    poi.set_value(0.0)
    poi.lower = 0.0
    poi.upper = 100.0
    poi.floating = True


def _compute_q_mu(sample, mu_test, minimizer):
    """Computes the one-sided LHC profile likelihood test statistic q_mu."""
    model = _CONTEXT['model']
    poi = _CONTEXT['poi']
    obs = _CONTEXT['obs']
    constraints = _CONTEXT['constraints']

    reset_parameters()
    
    data = zfit.Data.from_numpy(array=sample, obs=obs)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data, constraints=constraints)

    def usable(result):
        return np.isfinite(float(result.fmin)) and (
            result.valid or (getattr(result, 'converged', False) and result.status == 0)
        )

    try:
        # Unconstrained global fit
        global_fit = minimizer.minimize(loss)
        if not usable(global_fit):
            return np.nan
        
        mu_hat = float(global_fit.params[poi]['value'])
        best_nll = float(global_fit.fmin)
        
        # Standard one-sided test statistic: q_mu = 0 if mu_hat > mu_test or mu_hat < 0 (physical boundary)
        if mu_hat >= mu_test:
            return 0.0

        # Profiled fit conditional on mu = mu_test
        poi.set_value(mu_test)
        poi.floating = False
        conditional = minimizer.minimize(loss)
        
        if not usable(conditional):
            return np.nan
        
        q_mu = max(0.0, 2.0 * (float(conditional.fmin) - best_nll))
        return q_mu

    except Exception:
        return np.nan
    finally:
        poi.floating = True


def _empirical_scan_point(grid_index, mu_test):
    """Evaluate empirical CLs at a signal strength following Combine HybridNew logic."""
    ctx = _CONTEXT
    poi = ctx['poi']
    sampler = ctx['sampler']
    n_toys = ctx.get('n_toys', 200)
    seed_base = ctx.get('seed', 42)
    
    logger.info('Empirical scan point %d/%d: mu=%.5g', grid_index + 1, ctx.get('n_scan_points', 33), mu_test)
    minimizer = zfit.minimize.Minuit(tol=0.5, maxiter=5000, verbosity=0)
    
    b_q = []
    sb_q = []
    
    # Generate background-only ensemble (B)
    for toy_idx in range(n_toys):
        seed = int((seed_base + grid_index * 20000 + toy_idx) % (2**31 - 1))
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        poi.set_value(0.0)
        sampler.resample({poi: 0.0})
        sample = sampler.numpy()
        if sample.size == 0 or not np.all(np.isfinite(sample)):
            continue
            
        q = _compute_q_mu(sample, mu_test, minimizer)
        if np.isfinite(q):
            b_q.append(q)
            
        if (toy_idx + 1) % 25 == 0:
            gc.collect()

    # Generate signal+background ensemble (S+B)
    for toy_idx in range(n_toys):
        seed = int((seed_base + grid_index * 20000 + n_toys + toy_idx) % (2**31 - 1))
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        poi.set_value(mu_test)
        sampler.resample({poi: mu_test})
        sample = sampler.numpy()
        if sample.size == 0 or not np.all(np.isfinite(sample)):
            continue
            
        q = _compute_q_mu(sample, mu_test, minimizer)
        if np.isfinite(q):
            sb_q.append(q)
            
        if (toy_idx + 1) % 25 == 0:
            gc.collect()

    b_q = np.asarray(b_q)
    sb_q = np.asarray(sb_q)
    
    if len(b_q) < 10 or len(sb_q) < 10:
        logger.warning('Insufficient converged toys at mu=%.5g (b=%d, sb=%d)', mu_test, len(b_q), len(sb_q))
        return grid_index, np.nan

    # CMS Combine Expected Limit definition:
    # q_obs is defined as the median of q under the background-only hypothesis.
    q_obs = float(np.median(b_q))

    # Combine tail definitions: P(q >= q_obs)
    cl_sb = float(np.mean(sb_q >= q_obs))
    cl_b = float(np.mean(b_q >= q_obs))
    
    # Robust boundary protection for discrete distributions or low toy counts
    cl_b = max(cl_b, 0.5 / len(b_q))
    
    cls = cl_sb / cl_b if cl_b > 0 else 1.0
    
    logger.info('  mu=%.5g: q_obs=%.5g, CL_sb=%.5g, CL_b=%.5g, CLs=%.5g', mu_test, q_obs, cl_sb, cl_b, cls)
    
    del b_q, sb_q
    gc.collect()
    
    return grid_index, cls


def limit_from_toy(toy_index, seed):
    """Asymptotic asymptotic limit computation per toy."""
    if seed is not None:
        toy_seed = int((seed + toy_index * 10007) % (2**31 - 1))
        tf.random.set_seed(toy_seed)
        np.random.seed(toy_seed)

    model = _CONTEXT['model']
    poi = _CONTEXT['poi']
    sampler = _CONTEXT['sampler']
    sampler.resample({poi: 0.0})
    sample = sampler.numpy()
    if sample.size == 0 or not np.all(np.isfinite(sample)):
        logger.warning('Toy %d produced an empty or non-finite sample', toy_index)
        return np.nan

    reset_parameters()
    data = zfit.Data.from_numpy(array=sample, obs=_CONTEXT['obs'])
    loss = zfit.loss.ExtendedUnbinnedNLL(
        model=model, data=data, constraints=_CONTEXT['constraints']
    )
    minimizer = zfit.minimize.Minuit(tol=1e-3, maxiter=10000, verbosity=0)

    def usable(result):
        return np.isfinite(float(result.fmin)) and (
            result.valid or (getattr(result, 'converged', False) and result.status == 0)
        )

    try:
        global_fit = minimizer.minimize(loss)
        if not usable(global_fit):
            return np.nan
        best_nll = float(global_fit.fmin)
        mu_hat = max(0.0, float(global_fit.params[poi]['value']))
        
        try:
            global_fit.hesse()
            sigma = float(global_fit.params[poi]['errors']['hesse']['error'])
        except Exception:
            sigma = np.sqrt(max(1.0, mu_hat * _CONTEXT['nominals']['ref_signal_rate'] + _CONTEXT['nominals']['cosmic'] + _CONTEXT['nominals']['dio'] + _CONTEXT['nominals']['rpc'])) / _CONTEXT['nominals']['ref_signal_rate']
        
        if not np.isfinite(sigma) or sigma <= 0.0:
            return np.nan

        # Cowan et al. asymptotic formula for CLs
        def cls(mu_test):
            if mu_test <= mu_hat:
                return 1.0
            poi.set_value(mu_test)
            poi.floating = False
            try:
                profiled = minimizer.minimize(loss)
                if not usable(profiled):
                    return 1.0
                q = max(0.0, 2.0 * (float(profiled.fmin) - best_nll))
            finally:
                poi.floating = True
            
            sqrt_q = np.sqrt(q)
            sqrt_q_asimov = mu_test / sigma
            
            if sqrt_q <= sqrt_q_asimov:
                p_sb = stats.norm.sf(sqrt_q)
                one_minus_pb = stats.norm.cdf(sqrt_q_asimov - sqrt_q)
            else:
                p_sb = stats.norm.sf((q + sqrt_q_asimov**2) / (2.0 * sqrt_q_asimov))
                one_minus_pb = stats.norm.sf((q - sqrt_q_asimov**2) / (2.0 * sqrt_q_asimov))
            
            return p_sb / one_minus_pb if one_minus_pb > 1e-12 else 1.0

        upper = max(10.0, mu_hat + 20.0)
        while cls(upper) > _CONTEXT['target'] and upper < 1e4:
            upper *= 1.5
        if upper >= 1e4:
            return np.nan
        return opt.brentq(lambda value: cls(value) - _CONTEXT['target'], mu_hat, upper, xtol=0.05)
    except Exception as error:
        logger.warning('Toy %d limit calculation failed: %s', toy_index, error)
        return np.nan
    finally:
        poi.floating = True


def run_empirical_toy_cls(args):
    """Empirical toy CLs scan over mu grid."""
    mu_grid = np.linspace(args.mu_min, args.mu_max, args.mu_points)
    logger.info('Empirical toy CLs scan: %d points from mu=%.3f to %.3f, %d toys per ensemble', len(mu_grid), args.mu_min, args.mu_max, args.toys)
    target_cls = 1.0 - args.cl
    
    initialize_worker(args.datacard, args.dim, args.freeze_nuisances, args.cl, args.toys, args.mu_points, args.seed)
    
    tasks = [(i, mu) for i, mu in enumerate(mu_grid)]
    
    context = mp.get_context('spawn')
    with context.Pool(
        args.jobs,
        initializer=initialize_worker,
        initargs=(args.datacard, args.dim, args.freeze_nuisances, args.cl, args.toys, args.mu_points, args.seed),
    ) as pool:
        results = list(pool.starmap(_empirical_scan_point, tasks))
    
    cls_values = np.asarray([r[1] for r in results])
    valid = np.isfinite(cls_values)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError('Too few valid empirical CLs scan points.')
    
    crossing = np.where((cls_values[:-1] > target_cls) & (cls_values[1:] <= target_cls))[0]
    if len(crossing):
        index = crossing[0]
        limit_mu = np.interp(target_cls, cls_values[index:index + 2][::-1], mu_grid[index:index + 2][::-1])
    else:
        limit_mu = np.nan
        logger.warning('CLs scan did not cross %.5g; try higher --mu-max', target_cls)
    
    signal_rate = _CONTEXT['nominals']['ref_signal_rate']
    limit_events = limit_mu * signal_rate if np.isfinite(limit_mu) else np.nan
    effective_muons = args.n_muons * args.capture_rate
    rmue_limit = limit_events / (effective_muons * args.signal_efficiency) if np.isfinite(limit_events) else np.nan
    
    logger.info('Empirical toy CLs limit: mu=%.6g, N_CE=%.6g events', limit_mu, limit_events)
    logger.info('R_mu_e limit: %.6e', rmue_limit)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f'card_{args.dim}_empirical_cls_scan.npz',
        mu_grid=mu_grid,
        cls=cls_values,
        limit_mu=limit_mu,
        limit_events=limit_events,
        rmue_limit=rmue_limit,
        n_muons=args.n_muons,
        capture_rate=args.capture_rate,
        signal_efficiency=args.signal_efficiency,
        confidence_level=args.cl,
    )
    
    plt.figure(figsize=(9, 6))
    plt.plot(mu_grid, cls_values, 'o-', linewidth=2, markersize=6, label='Empirical CLs')
    plt.axhline(target_cls, color='red', linestyle='--', linewidth=2, label=f'Target CL={args.cl}')
    if np.isfinite(limit_mu):
        plt.axvline(limit_mu, color='green', linestyle='--', linewidth=2, label=f'Limit: μ={limit_mu:.3g}')
    plt.xlabel('Signal strength μ')
    plt.ylabel('CLs')
    plt.title(f'{args.dim.upper()} Empirical Toy CLs Scan (Combine Style)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim([0, 1.1])
    plt.tight_layout()
    plot_path = output_dir / f'card_{args.dim}_empirical_cls_scan.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info('Saved CLs scan plot: %s', plot_path)


def main():
    parser = argparse.ArgumentParser(description='Card-only Combine-style CLs limits')
    parser.add_argument('--datacard', required=True)
    parser.add_argument('--dim', choices=['1d', '2d'], default='2d')
    parser.add_argument('--method', choices=['asymptotic', 'empirical'], default='asymptotic')
    parser.add_argument('--toys', type=int, default=1000)
    parser.add_argument('--mu-min', type=float, default=0.0)
    parser.add_argument('--mu-max', type=float, default=80.0)
    parser.add_argument('--mu-points', type=int, default=33)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cl', type=float, default=0.90)
    parser.add_argument('--n-muons', type=float, default=5.588e15)
    parser.add_argument('--capture-rate', type=float, default=0.609)
    parser.add_argument('--signal-efficiency', type=float, default=0.1535)
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--freeze-nuisances', action='store_true')
    args = parser.parse_args()

    if args.method == 'empirical':
        run_empirical_toy_cls(args)
        return

    initialize_worker(args.datacard, args.dim, args.freeze_nuisances, args.cl, None, None, None)
    _CONTEXT['sampler'] = _CONTEXT['model'].create_sampler()
    logger.info('Running %d card-driven background toys with %d worker(s)', args.toys, args.jobs)

    tasks = [(index, args.seed + index if args.seed is not None else None) for index in range(args.toys)]
    if args.jobs > 1:
        context = mp.get_context('spawn')
        with context.Pool(
            args.jobs,
            initializer=initialize_worker,
            initargs=(args.datacard, args.dim, args.freeze_nuisances, args.cl, None, None, None),
        ) as pool:
            limits = list(pool.starmap(limit_from_toy, tasks))
    else:
        limits = [limit_from_toy(index, seed) for index, seed in tasks]

    limits = np.asarray(limits, dtype=float)
    limits = limits[np.isfinite(limits)]
    if not len(limits):
        raise RuntimeError('No card-driven toy limits converged')

    median, low, high = np.percentile(limits, [50.0, 15.87, 84.13])
    signal_rate = _CONTEXT['nominals']['ref_signal_rate']
    event_limits = limits * signal_rate
    events = median * signal_rate
    effective_muons = args.n_muons * args.capture_rate
    rmue_limits = event_limits / (effective_muons * args.signal_efficiency)
    rmue_median, rmue_low, rmue_high = np.percentile(rmue_limits, [50.0, 15.87, 84.13])
    logger.info('Converged toys: %d/%d', len(limits), args.toys)
    logger.info('Reference CE yield: %.8g events', signal_rate)
    logger.info('Median expected limit: %.4f events', events)
    logger.info('68%% band: [%.4f, %.4f] events', low * signal_rate, high * signal_rate)
    logger.info('R_mu_e normalization: N_muons=%.8g, capture_rate=%.8g, effective_muons=%.8g, signal_efficiency=%.8g', args.n_muons, args.capture_rate, effective_muons, args.signal_efficiency)
    logger.info('Median expected limit: R_mu_e = %.6e', rmue_median)
    logger.info('68%% band: [%.6e, %.6e] R_mu_e', rmue_low, rmue_high)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f'card_{args.dim}_cls_limits.npz',
        limits=limits,
        event_limits=event_limits,
        rmue_limits=rmue_limits,
        median=events,
        low=low * signal_rate,
        high=high * signal_rate,
        rmue_median=rmue_median,
        rmue_low=rmue_low,
        rmue_high=rmue_high,
        n_muons=args.n_muons,
        capture_rate=args.capture_rate,
        effective_muons=effective_muons,
        signal_efficiency=args.signal_efficiency,
        confidence_level=args.cl,
    )

    plt.figure(figsize=(9, 6))
    plt.hist(event_limits, bins=35, color='mediumpurple', alpha=0.6, edgecolor='indigo')
    plt.axvline(events, color='black', linestyle='--', linewidth=2, label=f'Median: {events:.2f} events')
    plt.axvspan(low * signal_rate, high * signal_rate, color='green', alpha=0.2, label='68% band')
    plt.xlabel('Expected upper limit on CE signal yield')
    plt.ylabel('Toys / bin')
    plt.title(f'{args.dim.upper()} card-driven asymptotic CLs limits')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / f'card_{args.dim}_cls_limits.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info('Saved limit distribution plot: %s', plot_path)


if __name__ == '__main__':
    main()