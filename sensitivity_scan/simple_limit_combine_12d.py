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
import pandas as pd
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
    
    yield_keys = {str(key).lower(): key for key in yields_dict}
    if 'rpcext' in yield_keys:
        yield_keys['rpc_ext'] = yield_keys['rpcext']
    if 'rpcint' in yield_keys:
        yield_keys['rpc_int'] = yield_keys['rpcint']
    has_split_rpc = 'rpc_ext' in yield_keys or 'rpc_int' in yield_keys
    if 'RPC' in yields_dict and not has_split_rpc:
        n_rpc_nom = yields_dict['RPC']
    else:
        n_rpc_nom = yields_dict.get(yield_keys.get('rpc_ext'), 0.0) + yields_dict.get(yield_keys.get('rpc_int'), 0.0)
    
    if 'RMC' in yields_dict:
        n_rmc_nom = yields_dict['RMC']
    else:
        n_rmc_nom = 0.0
        
    logger.info(f"[Datacard {dim.upper()}] Ref Signal Rate: {ref_signal_rate:.4f}")
    logger.info(f"[Datacard {dim.upper()}] Yields -> Cosmic: {n_cosmic_nom:.2f} | DIO: {n_dio_nom:.2f} | RPC: {n_rpc_nom:.2f} | RMC: {n_rmc_nom:.2f}")

    # POI is Signal Strength mu (N_CE = mu * ref_signal_rate)
    poi_mu = zfit.Parameter('mu', 0.0, lower=0.0, upper=100.0)
    
    nominal_yields = {
        'CE': ref_signal_rate,
        'DIO': n_dio_nom,
        'Cosmic': n_cosmic_nom,
        'RPC': n_rpc_nom,
        'rpc_ext': yields_dict.get(yield_keys.get('rpc_ext'), 0.0),
        'rpc_int': yields_dict.get(yield_keys.get('rpc_int'), 0.0),
        'RMC': n_rmc_nom,
    }
    nuisance_parameters = {}
    for syst_name, syst_info in card.systematics.items():
        if syst_info.get('type') == 'lnN':
            nuisance_parameters[syst_name] = zfit.Parameter(
                f'theta_{syst_name}', 0.0, lower=-7.0, upper=7.0
            )

    def process_yield(process_name, upper):
        nominal = nominal_yields[process_name]
        affected = []
        for syst_name, theta in nuisance_parameters.items():
            effects = card.systematics[syst_name].get('effects', {})
            effect = next(
                (value for name, value in effects.items()
                 if str(name).lower() in {
                     process_name.lower(),
                     'rpc' if process_name.lower() in ('rpc_ext', 'rpc_int') else process_name.lower(),
                 }),
                None,
            )
            if effect is not None and float(effect) > 0.0:
                affected.append((theta, np.log(float(effect))))

        if not affected:
            return zfit.Parameter(f'N_{process_name}', nominal, lower=0.0, upper=upper)

        parameters = [theta for theta, _ in affected]
        log_effects = [log_effect for _, log_effect in affected]

        def scaled_value(*values):
            parameter_values = values[0] if len(values) == 1 else values
            value_vector = _parameter_value_vector(parameter_values)
            effect_vector = tf.constant(log_effects, dtype=value_vector.dtype)
            return nominal * tf.exp(tf.reduce_sum(value_vector * effect_vector))

        return zfit.ComposedParameter(
            f'N_{process_name}',
            scaled_value,
            params=parameters,
        )

    yield_dio = process_yield('DIO', 5000.0)
    yield_cosmic = process_yield('Cosmic', 5000.0)
    if has_split_rpc:
        yield_rpc_ext = process_yield('rpc_ext', 1000.0)
        yield_rpc_int = process_yield('rpc_int', 1000.0)
    else:
        yield_rpc = process_yield('RPC', 1000.0)
    yield_rmc = process_yield('RMC', 1000.0)
    
    # Extract Shape Parameters
    ce_shapes = card.get_process_shape_info('CE')['params']
    cosmic_shapes = card.get_process_shape_info('Cosmic')['params']
    process_keys = {str(key).lower(): key for key in card.processes}
    if 'rpcext' in process_keys:
        process_keys['rpc_ext'] = process_keys['rpcext']
    if 'rpcint' in process_keys:
        process_keys['rpc_int'] = process_keys['rpcint']
    rpc_shape_name = process_keys.get('rpc', process_keys.get('rpc_ext', 'RPC'))
    rpc_shapes = card.get_process_shape_info(rpc_shape_name)['params']
    rpc_ext_shapes = card.get_process_shape_info(process_keys.get('rpc_ext', rpc_shape_name)).get('params', rpc_shapes)
    rpc_int_shapes = card.get_process_shape_info(process_keys.get('rpc_int', rpc_shape_name)).get('params', rpc_shapes)
    dio_shapes = card.get_process_shape_info('DIO')['params']
    rmc_shapes = card.get_process_shape_info('RMC')['params'] if 'RMC' in card.processes else {}
    
    # 1D PDFs - Momentum
    cosmic_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[cosmic_shapes['c1'], cosmic_shapes['c2']])
    rpc_ext_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[rpc_ext_shapes['c1'], rpc_ext_shapes['c2']])
    rpc_int_mom_pdf = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=[rpc_int_shapes['c1'], rpc_int_shapes['c2']])
    rpc_mom_pdf = rpc_ext_mom_pdf
    
    dio_a5 = zfit.Parameter('a5_DIO', dio_shapes['a5'], floating=False)
    dio_a6 = zfit.Parameter('a6_DIO', dio_shapes['a6'], floating=False)
    dio_a7 = zfit.Parameter('a7_DIO', dio_shapes['a7'], floating=False)
    dio_a8 = zfit.Parameter('a8_DIO', dio_shapes['a8'], floating=False)
    dio_mom_pdf = poly58(obs=obs_mom, a5=dio_a5, a6=dio_a6, a7=dio_a7, a8=dio_a8)
    
    signal_mom_pdf = zfit.pdf.DoubleCB(
        mu=ce_shapes['mu'],
        sigma=ce_shapes['sigma'],
        alphal=ce_shapes['alphaL'],
        alphar=ce_shapes['alphaR'],
        nl=ce_shapes['nL'],
        nr=ce_shapes['nR'],
        obs=obs_mom,
    )
    
    # RMC PDF - GammaPolyHybrid
    rmc_x0 = zfit.Parameter('rmc_x0', rmc_shapes.get('x0', 90.0), floating=False)
    rmc_alpha = zfit.Parameter('rmc_alpha', rmc_shapes.get('alpha', 2.0), floating=False)
    rmc_beta = zfit.Parameter('rmc_beta', rmc_shapes.get('beta', 1.0), floating=False)
    rmc_lam = zfit.Parameter('rmc_lambda', rmc_shapes.get('lam', 0.5), floating=False)
    rmc_mom_pdf = GammaPolyHybrid(obs=obs_mom, x0=rmc_x0, alpha=rmc_alpha, beta=rmc_beta, lam=rmc_lam)
    
    if dim == '1d':
        cosmic_pdf = cosmic_mom_pdf
        if has_split_rpc:
            rpc_ext_pdf = rpc_ext_mom_pdf
            rpc_int_pdf = rpc_int_mom_pdf
        else:
            rpc_pdf = rpc_mom_pdf
        dio_pdf = dio_mom_pdf
        rmc_pdf = rmc_mom_pdf
        signal_pdf = signal_mom_pdf
    else:
        # 1D PDFs - Time
        tau_dio = zfit.Parameter('tau_DIO', dio_shapes.get('tau', -864.0), floating=False)
        cosmic_time_pdf = zfit.pdf.Uniform(obs=obs_time, low=fit_range_time[0], high=fit_range_time[1])
        dio_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_dio, obs=obs_time)
        tau_rpc_ext = zfit.Parameter('tau_RPC_ext', rpc_ext_shapes.get('tau', -25.0), floating=False)
        tau_rpc_int = zfit.Parameter('tau_RPC_int', rpc_int_shapes.get('tau', -25.0), floating=False)
        rpc_ext_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_rpc_ext, obs=obs_time)
        rpc_int_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_rpc_int, obs=obs_time)
        signal_time_pdf = zfit.pdf.Exponential(lambda_=1.0/tau_dio, obs=obs_time) # CE follows DIO timing structure
        
        # 2D Product PDFs
        cosmic_pdf = zfit.pdf.ProductPDF([cosmic_mom_pdf, cosmic_time_pdf])
        if has_split_rpc:
            rpc_ext_pdf = zfit.pdf.ProductPDF([rpc_ext_mom_pdf, rpc_ext_time_pdf])
            rpc_int_pdf = zfit.pdf.ProductPDF([rpc_int_mom_pdf, rpc_int_time_pdf])
        else:
            rpc_pdf = zfit.pdf.ProductPDF([rpc_mom_pdf, rpc_ext_time_pdf])
        dio_pdf = zfit.pdf.ProductPDF([dio_mom_pdf, dio_time_pdf])
        rmc_pdf = zfit.pdf.ProductPDF([rmc_mom_pdf, cosmic_time_pdf])  # RMC uses Cosmic timing
        signal_pdf = zfit.pdf.ProductPDF([signal_mom_pdf, signal_time_pdf])
    
    # Extended PDFs
    signal_nuisances = []
    signal_log_effects = []
    for syst_name, theta in nuisance_parameters.items():
        effects = card.systematics[syst_name].get('effects', {})
        effect = next(
            (value for name, value in effects.items() if str(name).lower() == 'ce'),
            None,
        )
        if effect is not None and float(effect) > 0.0:
            signal_nuisances.append(theta)
            signal_log_effects.append(np.log(float(effect)))

    signal_parameters = [poi_mu] + signal_nuisances

    def signal_yield(*values):
        parameter_values = values[0] if len(values) == 1 else values
        value_vector = _parameter_value_vector(parameter_values)
        if not signal_log_effects:
            return value_vector[0] * ref_signal_rate
        effect_vector = tf.constant(signal_log_effects, dtype=value_vector.dtype)
        return value_vector[0] * ref_signal_rate * tf.exp(
            tf.reduce_sum(value_vector[1:] * effect_vector)
        )

    signal_yield_func = zfit.ComposedParameter(
        'signal_yield',
        signal_yield,
        params=signal_parameters,
    )
    
    dio_ext = dio_pdf.create_extended(yield_dio)
    cosmic_ext = cosmic_pdf.create_extended(yield_cosmic)
    if has_split_rpc:
        rpc_ext_extended = rpc_ext_pdf.create_extended(yield_rpc_ext)
        rpc_int_extended = rpc_int_pdf.create_extended(yield_rpc_int)
    else:
        rpc_extended = rpc_pdf.create_extended(yield_rpc)
    rmc_ext = rmc_pdf.create_extended(yield_rmc)
    signal_ext = signal_pdf.create_extended(signal_yield_func)
    
    rpc_extended_pdfs = [rpc_ext_extended, rpc_int_extended] if has_split_rpc else [rpc_extended]
    combined_pdf = zfit.pdf.SumPDF([dio_ext, cosmic_ext, *rpc_extended_pdfs, rmc_ext, signal_ext])
    
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

def build_gaussian_constraint(param):
    """Build the standard-normal constraint for a Combine lnN nuisance."""
    def gaussian_nll():
        return 0.5 * tf.square(param)

    return zfit.constraint.SimpleConstraint(gaussian_nll, params=[param])


def _parameter_value_vector(values):
    if isinstance(values, (tuple, list)):
        return tf.stack(values)
    return tf.reshape(values, (-1,))


def build_asimov_data(combined_pdf, obs_fit, nominals, dim):
    """Create a deterministic weighted quadrature approximation to Combine's Asimov data."""
    lower_limits, upper_limits = obs_fit.limits
    lower_values = np.asarray(lower_limits).reshape(-1)
    upper_values = np.asarray(upper_limits).reshape(-1)
    mom_low = float(lower_values[0])
    mom_high = float(upper_values[0])
    mom_bins = 240
    mom_step = (mom_high - mom_low) / mom_bins
    mom_values = np.linspace(mom_low, mom_high, mom_bins, endpoint=False) + mom_step / 2.0

    background_total = nominals['dio'] + nominals['cosmic'] + nominals['rpc'] + nominals['rmc']
    if dim == '1d':
        points = mom_values
        cell_volume = mom_step
    else:
        time_low = float(lower_values[1])
        time_high = float(upper_values[1])
        time_bins = 48
        time_step = (time_high - time_low) / time_bins
        time_values = np.linspace(time_low, time_high, time_bins, endpoint=False) + time_step / 2.0
        mom_grid, time_grid = np.meshgrid(mom_values, time_values, indexing='ij')
        points = np.column_stack((mom_grid.reshape(-1), time_grid.reshape(-1)))
        cell_volume = mom_step * time_step

    density = np.asarray(combined_pdf.pdf(points)).reshape(-1)
    weights = background_total * density * cell_volume
    finite = np.isfinite(weights) & (weights > 0.0)
    return points[finite], weights[finite]


def build_weighted_mc_data(component_specs, card, dim, mom_column='momentum', time_column='time'):
    """Build weighted background MC data using ParquetFitBuilder normalization."""
    mom_range = tuple(card.observables['mom'])
    time_range = tuple(card.observables.get('time', (475.0, 1650.0)))
    target_yields = card.get_expected_yields()
    mom_values = []
    time_values = []
    weights = []

    column_options = ['momentum', 'mom', 'p', 'recomom', 'recomom_ttfront']
    time_options = ['time', 'tracktime', 'tracktime_ttfront', 't']

    canonical_names = {
        'CE': 'CE',
        'COSMIC': 'Cosmic',
        'DIO': 'DIO',
        'RPC': 'RPC',
        'RMC': 'RMC',
        'RMC0N_EXT': 'RMC',
        'RMC0N_INT': 'RMC',
    }

    loaded_processes = set()
    for component_name, parquet_path in component_specs.items():
        frame = pd.read_parquet(parquet_path)
        selected_mom_column = mom_column if mom_column in frame.columns else next(
            (column for column in column_options if column in frame.columns), None
        )
        if selected_mom_column is None:
            raise ValueError(f'No momentum column found in {parquet_path}')

        selected_time_column = time_column if time_column in frame.columns else next(
            (column for column in time_options if column in frame.columns), None
        )
        component_mom = np.asarray(frame[selected_mom_column], dtype=np.float64)
        if dim == '2d':
            if selected_time_column is None:
                raise ValueError(f'No time column found in {parquet_path}')
            component_time = np.asarray(frame[selected_time_column], dtype=np.float64)
            in_range = (
                (component_mom >= mom_range[0]) & (component_mom <= mom_range[1]) &
                (component_time >= time_range[0]) & (component_time <= time_range[1])
            )
        else:
            component_time = None
            in_range = (component_mom >= mom_range[0]) & (component_mom <= mom_range[1])

        component_mom = component_mom[in_range]
        if dim == '2d':
            component_time = component_time[in_range]

        process_name = canonical_names.get(component_name.upper(), component_name)
        if process_name == 'CE':
            continue
        target_yield = float(target_yields.get(process_name, 0.0))
        scale_factor = target_yield / len(component_mom) if len(component_mom) and target_yield > 0.0 else 0.0
        if scale_factor == 0.0:
            continue

        mom_values.append(component_mom)
        loaded_processes.add(process_name)
        weights.append(np.full(len(component_mom), scale_factor, dtype=np.float64))
        if dim == '2d':
            time_values.append(component_time)

        logger.info(
            'Weighted MC %s: %d in-window events, scale %.8g, expected %.8g',
            component_name, len(component_mom), scale_factor, target_yield
        )

    if not mom_values:
        raise ValueError('No weighted background MC events were found in the fit range')

    missing_processes = [
        process_name for process_name in ('Cosmic', 'DIO', 'RPC', 'RMC')
        if float(target_yields.get(process_name, 0.0)) > 0.0 and process_name not in loaded_processes
    ]
    if missing_processes:
        raise ValueError(
            'Missing usable MC components for positive target yields: '
            + ', '.join(missing_processes)
        )

    if dim == '1d':
        points = np.concatenate(mom_values)
    else:
        points = np.column_stack((np.concatenate(mom_values), np.concatenate(time_values)))
    return points, np.concatenate(weights)


# ==============================================================================
# MULTIPROCESSING WORKER FUNCTIONS FOR PARALLEL TOYS
# ==============================================================================

_WORKER_CONTEXT = {}

def _worker_init(datacard_path, dim, freeze_nuisances, cl, verbosity, component_specs, asimov, method='asymptotic'):
    """Initializer for multiprocessing pool workers: builds zfit model once per worker process."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    combined_pdf, poi_mu, obs_fit, nominals, card = rebuild_combine_model_from_datacard(datacard_path, dim=dim)
    logger.info('Worker initialized model for %s mode', 'Asimov' if asimov else 'toy')
    target_cls = 1.0 - cl

    if freeze_nuisances:
        constraints = []
    else:
        constraints = [
            build_gaussian_constraint(parameter)
            for parameter in combined_pdf.get_params()
            if parameter.name.startswith('theta_')
        ]

    poi_mu.set_value(0.0)

    initial_parameter_values = {}
    for parameter in combined_pdf.get_params():
        if not hasattr(parameter, 'set_value') or not hasattr(parameter, 'value'):
            continue
        try:
            initial_parameter_values[parameter] = float(parameter.value())
        except (TypeError, ValueError):
            continue

    weighted_mc = None
    if asimov:
        logger.info('Loading and weighting MC components for expected dataset')
        weighted_mc = build_weighted_mc_data(component_specs, card, dim)
        logger.info('Weighted MC data ready: %d points, total weight %.3f', len(weighted_mc[0]), np.sum(weighted_mc[1]))
    sampler = None if asimov else combined_pdf.create_sampler()
    asimov_data = weighted_mc if method == 'toys' else None

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
        'weighted_mc': weighted_mc,
        'dim': dim,
        'verbosity': verbosity,
        'freeze_nuisances': freeze_nuisances
        , 'asimov_data': asimov_data
        , 'method': method
    }

def _worker_run_toy(args_tuple):
    """Executes a single asymptotic CLs toy evaluation inside a worker process."""
    toy_idx, seed, asimov = args_tuple

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
    weighted_mc = ctx['weighted_mc']
    dim = ctx['dim']
    verbosity = ctx['verbosity']
    freeze_nuisances = ctx['freeze_nuisances']

    if asimov:
        raw_sample, sample_weights = weighted_mc
        logger.info('Starting Asimov limit fit')
    else:
        sampler.resample({poi_mu: 0.0})
        raw_sample = sampler.numpy()
        sample_weights = None
        logger.info('Toy %d sampled %d events; starting fit', toy_idx, len(raw_sample))

    if raw_sample.size == 0 or not np.all(np.isfinite(raw_sample)):
        return toy_idx, np.nan, np.nan, 'nan'

    for parameter, value in initial_parameter_values.items():
        try:
            parameter.set_value(value)
        except Exception:
            pass
    poi_mu.set_value(0.0)
    poi_mu.floating = True

    minimizer = zfit.minimize.Minuit()
    data_kwargs = {'array': raw_sample, 'obs': obs_fit}
    if sample_weights is not None:
        data_kwargs['weights'] = sample_weights
    data_zfit = zfit.Data.from_numpy(**data_kwargs)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combined_pdf, data=data_zfit, constraints=constraints)

    poi_mu.lower = 0.0
    poi_mu.upper = 100.0
    poi_mu.floating = True

    try:
        logger.info('Toy %d: running global fit', toy_idx)
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
        logger.info('Toy %d: global fit complete (mu_hat=%.6g, fmin=%.6g)', toy_idx, mu_hat, nll_best)

        try:
            global_fit.hesse()
            sigma_mu = float(global_fit.params[poi_mu]['errors']['hesse']['error'])
        except Exception:
            sigma_mu = np.sqrt(max(1.0, (mu_hat * nominals['ref_signal_rate']) + nominals['cosmic'] + nominals['dio'] + nominals['rpc'])) / nominals['ref_signal_rate']
    except Exception:
        return toy_idx, np.nan, np.nan, 'fit_failure'

    profile_evaluations = 0

    def compute_asymptotic_cls(mu_test):
        nonlocal profile_evaluations
        if mu_test <= mu_hat:
            return 1.0

        poi_mu.set_value(mu_test)
        poi_mu.floating = False

        try:
            profile_evaluations += 1
            if profile_evaluations == 1 or profile_evaluations % 5 == 0:
                logger.info('Toy %d: conditional profile fit %d at mu=%.6g', toy_idx, profile_evaluations, mu_test)
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
            p_sb = stats.norm.sf(sqrt_q_mu)
            one_minus_pb = stats.norm.cdf(sqrt_q_mu_A - sqrt_q_mu)
        else:
            p_sb = stats.norm.sf((q_mu + sqrt_q_mu_A**2) / (2.0 * sqrt_q_mu_A))
            one_minus_pb = stats.norm.sf((q_mu - sqrt_q_mu_A**2) / (2.0 * sqrt_q_mu_A))

        if one_minus_pb <= 1e-9:
            return 1.0

        return p_sb / one_minus_pb

    def root_func(mu_test):
        return compute_asymptotic_cls(mu_test) - target_cls

    left_bracket = max(0.0, mu_hat)
    right_bracket = max(10.0, mu_hat + 30.0)

    try:
        logger.info('Toy %d: bracketing CLs limit', toy_idx)
        while root_func(right_bracket) > 0:
            right_bracket += 20.0
            if right_bracket > 300.0:
                return toy_idx, np.nan, np.nan, 'root_failure'

        mu_limit = opt.brentq(root_func, left_bracket, right_bracket, xtol=0.05)
        n_ce_limit = mu_limit * nominals['ref_signal_rate']
        logger.info('Toy %d: limit complete (mu=%.6g, N_CE=%.6g)', toy_idx, mu_limit, n_ce_limit)
        return toy_idx, n_ce_limit, mu_limit, 'success'
    except Exception:
        return toy_idx, np.nan, np.nan, 'root_failure'
    finally:
        poi_mu.floating = True


def _worker_run_q_mu(args_tuple):
    """Fit one toy and return its profile-likelihood q_mu statistic."""
    toy_idx, seed, mu_test, signal_mu, use_asimov = args_tuple
    if seed is not None:
        toy_seed = int((seed + toy_idx * 10007 + round(mu_test * 1000)) % (2**31 - 1))
        tf.random.set_seed(toy_seed)
        np.random.seed(toy_seed)

    ctx = _WORKER_CONTEXT
    combined_pdf = ctx['combined_pdf']
    poi_mu = ctx['poi_mu']
    obs_fit = ctx['obs_fit']
    constraints = ctx['constraints']
    sampler = ctx['sampler']
    initial_parameter_values = ctx['initial_parameter_values']

    if use_asimov:
        raw_sample, sample_weights = build_asimov_data(
            combined_pdf, obs_fit, ctx['nominals'], ctx['dim']
        )
    else:
        sampler.resample({poi_mu: signal_mu})
        raw_sample = sampler.numpy()
        sample_weights = None

    if raw_sample.size == 0 or not np.all(np.isfinite(raw_sample)):
        return np.nan

    for parameter, value in initial_parameter_values.items():
        try:
            parameter.set_value(value)
        except Exception:
            pass
    poi_mu.set_value(0.0)
    poi_mu.lower = 0.0
    poi_mu.upper = 100.0
    poi_mu.floating = True

    data_kwargs = {'array': raw_sample, 'obs': obs_fit}
    if sample_weights is not None:
        data_kwargs['weights'] = sample_weights
    data_zfit = zfit.Data.from_numpy(**data_kwargs)
    loss = zfit.loss.ExtendedUnbinnedNLL(model=combined_pdf, data=data_zfit, constraints=constraints)
    def reset_parameters():
        for parameter, value in initial_parameter_values.items():
            try:
                parameter.set_value(value)
            except Exception:
                pass
        poi_mu.set_value(0.0)
        poi_mu.lower = 0.0
        poi_mu.upper = 100.0
        poi_mu.floating = True

    minimizer = zfit.minimize.Minuit(tol=1e-3, maxiter=10000, verbosity=0)

    def usable_fit(result):
        return (
            np.isfinite(float(result.fmin)) and
            (result.valid or (getattr(result, 'converged', False) and result.status == 0))
        )

    try:
        global_fit = minimizer.minimize(loss)
        if not usable_fit(global_fit):
            reset_parameters()
            retry_minimizer = zfit.minimize.Minuit(tol=1e-2, maxiter=10000, verbosity=0)
            global_fit = retry_minimizer.minimize(loss)
            if not usable_fit(global_fit):
                logger.warning('q_mu global fit invalid for toy %d at mu=%.6g', toy_idx, mu_test)
                return np.nan
        mu_hat = max(0.0, float(global_fit.params[poi_mu]['value']))
        nll_best = float(global_fit.fmin)
        if mu_test <= mu_hat:
            return 0.0

        poi_mu.set_value(mu_test)
        poi_mu.floating = False
        profile_fit = minimizer.minimize(loss)
        if not usable_fit(profile_fit):
            retry_minimizer = zfit.minimize.Minuit(tol=1e-2, maxiter=10000, verbosity=0)
            profile_fit = retry_minimizer.minimize(loss)
            if not usable_fit(profile_fit):
                logger.warning('q_mu conditional fit invalid for toy %d at mu=%.6g', toy_idx, mu_test)
                return np.nan
        return max(0.0, 2.0 * (float(profile_fit.fmin) - nll_best))
    except Exception as error:
        logger.warning(
            'q_mu fit failed for toy %d at mu=%.6g: %s',
            toy_idx, mu_test, error
        )
        return np.nan
    finally:
        poi_mu.floating = True


def run_empirical_toy_cls(args, component_specs):
    """Scan empirical CLs using background and signal-plus-background toy ensembles."""
    target_cls = 1.0 - args.cl
    _worker_init(
        args.datacard, args.dim, args.freeze_nuisances, args.cl, args.verbosity,
        component_specs, False, method='toys'
    )

    mu_grid = np.linspace(args.mu_min, args.mu_max, args.mu_points)
    logger.info('Toy CLs scan: %d points from mu=%.3f to %.3f, %d toys per ensemble',
                len(mu_grid), args.mu_min, args.mu_max, args.toys)
    cls_values = []
    pool = None
    if args.jobs > 1:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(
            processes=args.jobs,
            initializer=_worker_init,
            initargs=(args.datacard, args.dim, args.freeze_nuisances, args.cl,
                      args.verbosity, component_specs, False, 'toys'),
        )

    try:
        for grid_index, mu_test in enumerate(mu_grid):
            q_obs = _worker_run_q_mu((0, args.seed, float(mu_test), 0.0, True))
            if not np.isfinite(q_obs):
                cls_values.append(np.nan)
                logger.warning('Toy CLs point %d/%d failed while fitting Asimov statistic', grid_index + 1, len(mu_grid))
                continue

            tasks = []
            for ensemble, signal_mu in (('b', 0.0), ('sb', float(mu_test))):
                offset = 0 if ensemble == 'b' else args.toys
                tasks.extend((offset + toy_index, args.seed, float(mu_test), signal_mu, False)
                             for toy_index in range(args.toys))

            if pool is not None:
                q_values = list(pool.imap(_worker_run_q_mu, tasks))
            else:
                q_values = [_worker_run_q_mu(task) for task in tasks]

            q_b = np.asarray(q_values[:args.toys], dtype=float)
            q_sb = np.asarray(q_values[args.toys:], dtype=float)
            q_b = q_b[np.isfinite(q_b)]
            q_sb = q_sb[np.isfinite(q_sb)]
            if not len(q_b) or not len(q_sb):
                cls_values.append(np.nan)
                continue

            p_sb = np.mean(q_sb >= q_obs)
            one_minus_pb = np.mean(q_b >= q_obs)
            cls = p_sb / one_minus_pb if one_minus_pb > 0.0 else 1.0
            cls_values.append(cls)
            logger.info('Toy CLs %d/%d: mu=%.5g, qobs=%.5g, p_sb=%.5g, 1-p_b=%.5g, CLs=%.5g',
                        grid_index + 1, len(mu_grid), mu_test, q_obs, p_sb, one_minus_pb, cls)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    cls_values = np.asarray(cls_values, dtype=float)
    valid = np.isfinite(cls_values)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError('Too few valid empirical CLs scan points.')

    crossing = np.where((cls_values[:-1] > target_cls) & (cls_values[1:] <= target_cls))[0]
    if len(crossing):
        index = crossing[0]
        limit_mu = np.interp(target_cls, cls_values[index:index + 2][::-1], mu_grid[index:index + 2][::-1])
    else:
        limit_mu = np.nan
        logger.warning('CLs scan did not cross %.5g; increase --mu-max or --mu-points', target_cls)

    limit_events = limit_mu * _WORKER_CONTEXT['nominals']['ref_signal_rate'] if np.isfinite(limit_mu) else np.nan
    logger.info('Empirical toy CLs limit: mu=%.6g, N_CE=%.6g events', limit_mu, limit_events)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f'datacard_{args.dim}_toy_cls_scan.npz',
             mu_grid=mu_grid, cls=cls_values, limit_mu=limit_mu,
             limit_events=limit_events, confidence_level=args.cl)


# ==============================================================================
# MAIN EXECUTION ROUTINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='1D/2D Datacard-driven Asymptotic CLs Limit Setter')
    parser.add_argument('--datacard', required=True, help='Path to YAML datacard')
    parser.add_argument('--dim', choices=['1d', '2d'], default='2d', help='Fit dimension: 1d (momentum) or 2d (momentum + time)')
    parser.add_argument('--method', choices=['asymptotic', 'toys'], default='asymptotic', help='Limit method')
    parser.add_argument('--components', nargs='+', default=[], help='Weighted MC components as name=/path/file.parquet')
    parser.add_argument('--asimov', action='store_true', help='Fit one deterministic weighted Asimov dataset')
    parser.add_argument('--toys', type=int, default=1000, help='Number of toy experiments')
    parser.add_argument('--jobs', '-j', type=int, default=1, help='Number of parallel worker processes (default: 1)')
    parser.add_argument('--mu-min', type=float, default=0.0, help='Minimum signal strength for empirical toy scan')
    parser.add_argument('--mu-max', type=float, default=80.0, help='Maximum signal strength for empirical toy scan')
    parser.add_argument('--mu-points', type=int, default=33, help='Number of signal-strength points for empirical toy scan')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cl', type=float, default=0.90, help='Confidence Level (default 0.90)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--freeze-nuisances', action='store_true', help='Freeze nuisance parameters (stat only)')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for debugging failed fits')
    args = parser.parse_args()

    component_specs = {}
    for specification in args.components:
        component_name, separator, parquet_path = specification.partition('=')
        if not separator or not component_name or not parquet_path:
            parser.error(f'Invalid component specification: {specification}')
        component_specs[component_name.lower()] = parquet_path
    if args.asimov and not component_specs:
        parser.error('--components is required with --asimov')

    if args.method == 'toys':
        if args.asimov:
            parser.error('--asimov cannot be combined with --method toys')
        run_empirical_toy_cls(args, component_specs)
        return

    n_ce_limits = []
    mu_limits = []
    n_converged = 0
    n_failed = 0
    n_nan_samples = 0
    n_fit_failures = 0

    n_requested = 1 if args.asimov else args.toys
    n_jobs = 1 if args.asimov else max(1, args.jobs)
    run_label = 'Asimov dataset' if args.asimov else f'{args.toys} background-only toys'
    logger.info(f"Running {run_label} for {args.dim.upper()} from datacard: {args.datacard} using {n_jobs} process(es)...")

    toy_tasks = [
        (i, args.seed + i if args.seed is not None else None, args.asimov)
        for i in range(n_requested)
    ]

    if n_jobs > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_jobs, initializer=_worker_init, initargs=(args.datacard, args.dim, args.freeze_nuisances, args.cl, args.verbosity, component_specs, args.asimov)) as pool:
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

                if (idx + 1) % max(1, n_requested // 10) == 0:
                    logger.info(f"Progress: {idx + 1} / {n_requested} completed... ({n_converged} converged, {n_failed} failed)")
    else:
        _worker_init(args.datacard, args.dim, args.freeze_nuisances, args.cl, args.verbosity, component_specs, args.asimov)
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

            if (idx + 1) % max(1, n_requested // 10) == 0:
                logger.info(f"Progress: {idx + 1} / {n_requested} completed... ({n_converged} converged, {n_failed} failed)")

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
    logger.info(f"Evaluations Requested: {n_requested}")
    logger.info(f"Evaluations Converged: {n_converged} ({100.0*n_converged/n_requested:.1f}%)")
    logger.info(f"Evaluations Failed:    {n_failed} ({100.0*n_failed/n_requested:.1f}%)")
    logger.info(f"  → NaN Samples:      {n_nan_samples}")
    logger.info(f"  → Fit Failures:     {n_fit_failures}")
    logger.info(f"Median Expected Limit (Signal Strength mu): {med_mu:.2f}")
    logger.info(f"Median Expected Limit (Events N_CE):        {med_n_ce:.2f}")
    logger.info(f"68% Band (Events N_CE):                     [{p16_n_ce:.2f}, {p84_n_ce:.2f}]")
    logger.info("="*60)

    if n_converged < n_requested * 0.5:
        logger.error(f"WARNING: Less than 50% convergence rate ({n_converged}/{n_requested})")
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
