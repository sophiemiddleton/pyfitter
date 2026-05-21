# momentum_pdf_builder.py: Concrete PDF Builders for Momentum and Time
# Consolidates momentum and time PDF building with custom PDF definitions and parameters

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import tensorflow as tf
import zfit
from pdf_builder import PDFBuilder
from pyutils.pylogger import Logger

# ============================================================================
# Custom PDF Classes (consolidated from momPDF_module.py)
# ============================================================================

# Physical constants
m_mu = 105.194  # mass of the muon [MeV]

class poly58(zfit.pdf.ZPDF):
    """5th to 8th order polynomial for DIO background (momentum space)."""
    _N_OBS = 1
    _PARAMS = ['a5', 'a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        a5, a6, a7, a8 = self.params['a5'], self.params['a6'], self.params['a7'], self.params['a8']
        m_Al = 25133  # mass of the Aluminum atom [MeV]
        delta = tf.nn.relu(m_mu - x - x**2 / (2 * m_Al))
        return a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8


class DIO_custom_model_2025(zfit.pdf.ZPDF):
    """Custom DIO model with endpoint, beta parameter, and degree shift."""
    _N_OBS = 1
    _PARAMS = ['DIO_endpoint', 'beta', 'degree_shift']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        endpoint = self.params['DIO_endpoint']
        beta = self.params['beta']
        degree_shift = self.params['degree_shift']

        delta_E = (endpoint - x)
        is_valid = delta_E > 0
        safe_delta_E = tf.where(is_valid, delta_E, 1.0)
        log_delta_E_over_mu = tf.math.log(safe_delta_E / m_mu)

        power = 5.0 + degree_shift
        poly_term = beta * tf.square(log_delta_E_over_mu)
        pdf_val = tf.pow(safe_delta_E, power) * tf.exp(poly_term)

        return tf.where(is_valid, pdf_val, 0.0)


# ============================================================================
# Default Parameters (consolidated from momPDF_module.py and timePDF_module.py)
# ============================================================================

# Momentum model defaults
mom_default_model_params = {
    'poly2': {'c1': (0.47, 0.46, 0.48), 'c2': (0.011, 0.0018, 0.0202)},
    'dscb': {'mu': (104, 100, 107), 'sigma': (0.5, 0.0, 2.0), 'alphaL': (0.422, 0, 10),
             'nL': (25.1, 0, 100), 'alphaR': (2.227, 0, 100), 'nR': (5.954, 0, 100)},
    'Gauss': {'mu': (100, 95, 115), 'sigma': (10.0, 1e-3, 1e3)},
    'poly58': {'a5': (8.97879e-17, 1e-17, 1e-16),
               'a6': (1.17169e-17, 1e-18, 1e-16),
               'a7': (-1.06599e-19, -1e-18, -1e-19),
               'a8': (8.14251e-20, 1e-20, 1e-19)},
    'uniform': {}
}

mom_default_norms = {'CE': 0, 'DIO': 55000, 'Cosmic': 5000, 'RPC': 24}
default_N_CE_bounds = (0.0, 1e4)  # CE lower bound allows negative exploration for BG-only fits

# Time model defaults
time_default_model_params = {
    'muexp': {'decay_rate_mu': (-0.001157, -0.0015, -0.001)},
    'piexp': {'decay_rate_pi': (-0.03846, -0.04, -0.01)},
    'cosmicexp': {'decay_rate_cosmic': (-0.037, -0.04, -0.03)},
    'uniform': {}
}

time_default_norms = {'Cosmic': 35, 'RPC': 39, 'Muon': 55600}

# Module-level logger
try:
    logger = Logger(print_prefix='[momentum_pdf_builder] ', verbosity=2)
except Exception:
    logger = None

# Shared parameters cache for MomTimeModel
_shared_time_params = {}


class MomPDFBuilder(PDFBuilder):
    """
    Concrete builder for momentum PDFs.
    
    Supports models: 'dscb', 'Gauss', 'uniform', 'poly2', 'poly5', 'poly58', 'DIO_custom_model_2025'
    Handles both simple and advanced (theo_exp) configurations.
    """
    
    def __init__(self):
        """Initialize with momentum-specific defaults."""
        super().__init__(
            default_model_params=mom_default_model_params,
            default_norms=mom_default_norms
        )
        self.advanced_config = None
        self.use_advanced = False
    
    def build(self,
              obs: zfit.Space,
              params_tot: List[zfit.Parameter],
              process: str,
              model: str,
              pardict: Optional[Dict] = None,
              treat_params: str = 'float',
              fit_range: Tuple[float, float] = None,
              constraints: List = None,
              advanced_config: Dict = None,
              use_advanced: bool = False,
              **kwargs) -> Tuple[zfit.pdf.ZPDF, zfit.Parameter]:
        """
        Build momentum PDF with optional advanced configuration support.
        
        Args:
            advanced_config: Dict with 'pdf_theo', 'treat_params_adv', 'fitpars_in_formatted'
            use_advanced: Whether to use advanced (theo_exp) path
            **kwargs: Passed to parent class
        """
        self.advanced_config = advanced_config
        self.use_advanced = use_advanced and bool(advanced_config)
        
        if self.use_advanced:
            return self._build_advanced_pdf(
                obs, params_tot, process, pardict, fit_range, constraints
            )
        else:
            return super().build(
                obs, params_tot, process, model, pardict,
                treat_params, fit_range, constraints, **kwargs
            )
    
    def _get_normalization_bounds(self, process: str) -> Tuple[float, float]:
        """
        Custom bounds for specific processes.
        CE has special handling to allow negative exploration in background-only fits.
        """
        if process == 'CE':
            return default_N_CE_bounds
        return (0.0, 1e6)
    
    def _build_advanced_pdf(self,
                           obs: zfit.Space,
                           params_tot: List[zfit.Parameter],
                           process: str,
                           pardict: Optional[Dict],
                           fit_range: Tuple[float, float],
                           constraints: List) -> Tuple[zfit.pdf.ZPDF, zfit.Parameter]:
        """
        Build PDF using advanced theo_exp convolution path.
        """
        # Get normalization
        N = self._get_normalization(process, params_tot, pardict)
        params_tot.append(N)
        
        config = self.advanced_config
        adv_model = config.get('pdf_theo')
        adv_treat = config.get('treat_params_adv', 'float')
        
        if adv_model != 'theo_exp':
            raise ValueError(f"Advanced model {adv_model} not supported")
        
        # Extract formatted parameters
        theo_exp_pars = config['fitpars_in_formatted']
        prob, edges = theo_exp_pars['lineshape']
        info = theo_exp_pars['info']
        
        # Map parameters from res/loss components
        zpars = {}
        for comp in ['res', 'loss']:
            comp_obj = theo_exp_pars[comp]
            comp_pars = comp_obj.get_params()
            
            for p, val in comp_pars.items():
                if adv_treat == 'simul':
                    # Create composed parameter referencing the component parameter
                    zpars[p] = zfit.ComposedParameter(
                        f'{p}_{process}',
                        lambda x: 1 * x,
                        params=val
                    )
                else:
                    # Use fixed parameter
                    zpars[p] = val
                    if val not in params_tot:
                        params_tot.append(val)
        
        # Build convolution PDF
        from helper import doConv, make_HistogramPDF
        obs_gen = zfit.Space('x', fit_range)
        obs_res = zfit.Space('x', -10, 10)  # Standard resolution window
        
        true_pdf_slice = (prob, edges)
        pdf_conv = doConv(true_pdf_slice, obs_gen, obs_res, process, info, zpars)
        
        # Wrap and extend
        pdf_conv = zfit.pdf.TruncatedPDF(pdf_conv, limits=obs, obs=obs, extended=N)
        try:
            pdf_conv.set_yield(N)
        except Exception:
            pass
        
        return pdf_conv, N
    
    def _build_pdf(self,
                   obs: zfit.Space,
                   model: str,
                   zpars: Dict[str, zfit.Parameter],
                   process: str,
                   fit_range: Tuple[float, float] = None,
                   **kwargs) -> zfit.pdf.ZPDF:
        """
        Build simple path PDF based on model type.
        """
        N = zpars['N']
        
        if model == 'dscb':
            return zfit.pdf.DoubleCB(
                obs=obs,
                mu=zpars['mu'],
                sigma=zpars['sigma'],
                alphal=zpars['alphaL'],
                nl=zpars['nL'],
                alphar=zpars['alphaR'],
                nr=zpars['nR'],
                extended=N
            )
        
        elif model == 'Gauss':
            return zfit.pdf.Gauss(
                obs=obs,
                mu=zpars['mu'],
                sigma=zpars['sigma'],
                extended=N
            )
        
        elif model == 'uniform':
            return zfit.pdf.Uniform(
                low=fit_range[0],
                high=fit_range[1],
                obs=obs,
                extended=N
            )
        
        elif model == 'poly58':
            return poly58(
                obs=obs,
                a5=zpars['a5'],
                a6=zpars['a6'],
                a7=zpars['a7'],
                a8=zpars['a8'],
                extended=N
            )
        
        elif model == 'DIO_custom_model_2025':
            return DIO_custom_model_2025(
                obs=obs,
                DIO_endpoint=zpars.get('endpoint', 104.97),
                beta=zpars.get('beta', -0.002),
                degree_shift=zpars.get('degree_shift', 0),
                extended=N
            )
        
        elif model in ('poly2', 'poly5'):
            coeffs = self._extract_polynomial_coeffs(model, zpars, process)
            return zfit.pdf.Chebyshev(
                obs=obs,
                coeffs=coeffs,
                extended=N
            )
        
        else:
            raise ValueError(f"Model {model} not recognized in MomPDFBuilder")
    
    def _extract_polynomial_coeffs(self,
                                  model: str,
                                  zpars: Dict[str, zfit.Parameter],
                                  process: str,
                                  params_tot: List[zfit.Parameter] = None) -> List[zfit.Parameter]:
        """
        Extract polynomial coefficients, creating parameters if needed.
        """
        order = int(model[-1])  # Extract order from 'poly2', 'poly5'
        coeffs = []
        
        for i in range(1, order + 1):
            coeff_name = f'c{i}'
            if coeff_name in zpars:
                coeffs.append(zpars[coeff_name])
            else:
                # Create default fixed parameter
                p = zfit.Parameter(
                    f'{coeff_name}_{process}',
                    0.0,
                    floating=False
                )
                coeffs.append(p)
                if params_tot is not None:
                    try:
                        params_tot.append(p)
                    except Exception:
                        pass
        
        return coeffs


class TimePDFBuilder(PDFBuilder):
    """
    Concrete builder for time PDFs.
    
    Supports models: 'muexp', 'piexp', 'cosmicexp', 'uniform'
    """
    
    def __init__(self):
        """Initialize with time-specific defaults."""
        super().__init__(
            default_model_params=time_default_model_params,
            default_norms=time_default_norms
        )
    
    def _build_pdf(self,
                   obs: zfit.Space,
                   model: str,
                   zpars: Dict[str, zfit.Parameter],
                   process: str,
                   fit_range: Tuple[float, float] = None,
                   **kwargs) -> zfit.pdf.ZPDF:
        """
        Build time PDF based on model type.
        """
        N = zpars['N']
        
        if model == 'muexp':
            return zfit.pdf.Exponential(
                zpars['decay_rate_mu'],
                obs=obs,
                extended=N
            )
        
        elif model == 'piexp':
            return zfit.pdf.Exponential(
                zpars['decay_rate_pi'],
                obs=obs,
                extended=N
            )
        
        elif model == 'cosmicexp':
            return zfit.pdf.Exponential(
                zpars['decay_rate_cosmic'],
                obs=obs,
                extended=N
            )
        
        elif model == 'uniform':
            return zfit.pdf.Uniform(
                low=fit_range[0],
                high=fit_range[1],
                obs=obs,
                extended=N
            )
        
        else:
            raise ValueError(f"Model {model} not recognized in TimePDFBuilder")


class MomTimePDFBuilder:
    """
    Builder for combined 2D momentum×time PDFs.
    
    Delegates momentum and time parts to specialized builders and combines them.
    """
    
    def __init__(self):
        """Initialize with momentum and time builders."""
        self.mom_builder = MomPDFBuilder()
        self.time_builder = TimePDFBuilder()
    
    def build(self,
              obs_mom: zfit.Space,
              obs_time: zfit.Space,
              mom_params_tot: List[zfit.Parameter],
              time_params_tot: List[zfit.Parameter],
              process: str,
              mom_model: str,
              time_model: str,
              pardict: Optional[Dict] = None,
              treat_params: str = 'float',
              fit_range: Tuple[float, float] = None,
              constraints: List = None,
              advanced_config: Dict = None,
              use_advanced: bool = False,
              **kwargs) -> Tuple[zfit.pdf.ZPDF, zfit.Parameter, zfit.pdf.ZPDF, zfit.pdf.ZPDF]:
        """
        Build combined 2D momentum×time PDF.
        
        Returns:
            Tuple of (pdf_2d, N, mom_pdf, time_pdf)
        """
        if constraints is None:
            constraints = []
        
        # Build momentum part (supports advanced configurations)
        mom_pdf, N = self.mom_builder.build(
            obs_mom,
            mom_params_tot,
            process,
            mom_model,
            pardict,
            treat_params,
            fit_range,
            constraints,
            advanced_config=advanced_config,
            use_advanced=use_advanced,
            **kwargs
        )
        
        # Ensure shared/fixed decay parameters exist
        if 'decay_shared_CE_DIO' not in _shared_time_params:
            _shared_time_params['decay_shared_CE_DIO'] = zfit.Parameter(
                'decay_shared_CE_DIO',
                -1.0 / 864.0,
                floating=False
            )
        if 'decay_rpc' not in _shared_time_params:
            _shared_time_params['decay_rpc'] = zfit.Parameter(
                'decay_rpc',
                -1.0 / 26.0,
                floating=False
            )
        
        # Select time PDF based on process
        time_pdf = self._get_time_pdf(
            obs_time,
            time_params_tot,
            process,
            time_model,
            pardict,
            fit_range
        )
        
        # Append shared decay parameters to time_params_tot if needed
        self._add_shared_params(process, time_params_tot)
        
        # Combine into 2D PDF
        obs_2d = obs_mom * obs_time
        pdf_2d = zfit.pdf.ProductPDF([mom_pdf, time_pdf], obs=obs_2d)
        
        try:
            pdf_2d.set_yield(N)
        except Exception:
            try:
                pdf_2d = zfit.pdf.TruncatedPDF(
                    pdf_2d,
                    limits=obs_2d,
                    obs=obs_2d,
                    extended=N
                )
            except Exception:
                pass
        
        return pdf_2d, N, mom_pdf, time_pdf
    
    def _get_time_pdf(self,
                     obs_time: zfit.Space,
                     time_params_tot: List[zfit.Parameter],
                     process: str,
                     time_model: str,
                     pardict: Optional[Dict],
                     fit_range: Tuple[float, float]) -> zfit.pdf.ZPDF:
        """
        Get appropriate time PDF for the process.
        
        Uses fixed decay rates for physics-motivated processes,
        delegates to TimePDFBuilder for others.
        """
        if process in ('DIO', 'CE'):
            lam = _shared_time_params['decay_shared_CE_DIO']
            return zfit.pdf.Exponential(lam, obs=obs_time)
        
        elif process == 'RPC':
            lam = _shared_time_params['decay_rpc']
            return zfit.pdf.Exponential(lam, obs=obs_time)
        
        elif process == 'Cosmic':
            return zfit.pdf.Uniform(low=400.0, high=1695.0, obs=obs_time)
        
        else:
            # Fall back to builder
            try:
                pdf, _ = self.time_builder.build(
                    obs_time,
                    time_params_tot,
                    process,
                    time_model,
                    pardict,
                    'float',
                    fit_range,
                    None
                )
                return pdf
            except Exception:
                # Last resort: uniform
                return zfit.pdf.Uniform(
                    low=fit_range[0],
                    high=fit_range[1],
                    obs=obs_time
                )
    
    def _add_shared_params(self,
                          process: str,
                          time_params_tot: List[zfit.Parameter]) -> None:
        """
        Safely add shared decay parameters to time params list.
        """
        try:
            if process in ('DIO', 'CE'):
                if _shared_time_params['decay_shared_CE_DIO'] not in time_params_tot:
                    time_params_tot.append(_shared_time_params['decay_shared_CE_DIO'])
            elif process == 'RPC':
                if _shared_time_params['decay_rpc'] not in time_params_tot:
                    time_params_tot.append(_shared_time_params['decay_rpc'])
        except Exception:
            pass
