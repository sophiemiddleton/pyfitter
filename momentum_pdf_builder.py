# momentum_pdf_builder.py: Concrete PDF Builders for Momentum and Time
# Consolidates momentum and time PDF building with custom PDF definitions and parameters

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import tensorflow as tf
import zfit
from pdf_builder import PDFBuilder
from pyutils.pylogger import Logger
from custom_models import poly58, DIO_custom_model_2025
from pyutils.pylogger import Logger
from pathlib import Path
from config import GLOBAL_VERBOSITY
from model.physics_components import mom_components, time_components

# Module-level logger
module_logger = Logger(print_prefix='[process] ', verbosity=GLOBAL_VERBOSITY)

# ============================================================================
# Default Parameters (consolidated from momPDF_module.py and timePDF_module.py)
# ============================================================================

# Momentum model defaults #NOTE model specific parameters including norms are now defined within the component dicts in physics_components.py
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
logger = Logger(print_prefix='[momentum_pdf_builder] ', verbosity=2)


# Shared parameters cache for MomTimeModel
_shared_time_params = {}


class MomPDFBuilder(PDFBuilder):
    """
    Concrete builder for momentum PDFs.
    
    Supports models: 'dscb', 'Gauss', 'uniform', 'poly2', 'poly5', 'poly58', 'DIO_custom_model_2025'
    Handles both simple and advanced (theo_exp) configurations.
    """
    
    def __init__(self):
        """Initialize with momentum-specific defaults and physics components."""
        super().__init__(
            default_model_params=mom_default_model_params,
            default_norms=mom_default_norms,
            physics_components=mom_components  # Use physics components for norm lookup
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
        Build PDF using advanced convolution paths.
        Supports:
          - 'theo_exp': Theoretical lineshape convolved with parametric resolution
          - 'dscb_conv': Theory PDF convolved with calibrated DSCB resolution
        """
        # Get normalization
        N = self._get_normalization(process, params_tot, pardict)
        params_tot.append(N)
        
        # Extract advanced configuration from component dict
        config = self.advanced_config
        adv_pars = config.get('advanced_pars', {})
        
        if not adv_pars:
            raise ValueError(f"Advanced configuration missing for process {process}")
        
        adv_model = adv_pars.get('pdf_theo')
        adv_treat = adv_pars.get('treat_params_adv', 'float')
        
        # Route to appropriate builder
        if adv_model == 'theo_exp':
            return self._build_theo_exp_pdf(
                obs, params_tot, process, adv_pars, adv_treat, N
            )
        elif adv_model == 'dscb_conv':
            return self._build_dscb_convolution_pdf(
                obs, params_tot, process, adv_pars, adv_treat, N, fit_range, constraints
            )
        else:
            raise ValueError(f"Advanced model {adv_model} not supported. "
                           f"Choose from: 'theo_exp', 'dscb_conv'")
    
    def _build_theo_exp_pdf(self,
                           obs: zfit.Space,
                           params_tot: List[zfit.Parameter],
                           process: str,
                           adv_pars: Dict,
                           adv_treat: str,
                           N: zfit.Parameter) -> Tuple[zfit.pdf.ZPDF, zfit.Parameter]:
        """
        Build PDF using theo_exp (theoretical lineshape + parametric resolution) path.
        """
        # Extract formatted parameters
        theo_exp_pars = adv_pars['fitpars_in_formatted']
        prob, edges = theo_exp_pars['lineshape']
        info = theo_exp_pars['info']
        
        # Map parameters from res component (doConv uses resolution for convolution)
        # Rename parameters from 'mu0_res' -> 'mu0_{process}' (e.g., 'mu0_CE')
        zpars = {}
        res_obj = theo_exp_pars['res']
        res_pars = res_obj.get_params()
        
        for p, val in res_pars.items():
            if p == 'info':  # Skip the info dict
                continue
            
            # Extract parameter base name by removing component type suffix
            if p.endswith('_res'):
                param_base = p[:-4]  # Remove '_res'
                new_name = f'{param_base}_{process}'  # Rename to process name (e.g., 'mu0_CE')
                
                if adv_treat == 'simul':
                    zpars[new_name] = zfit.ComposedParameter(
                        new_name,
                        lambda x: 1 * x,
                        params=val
                    )
                else:
                    zpars[new_name] = val
                    if val not in params_tot:
                        params_tot.append(val)
        
        # Build convolution PDF
        from helper import doConv, make_HistogramPDF
        obs_gen = obs  # Use the momentum observable directly (already properly configured)
        obs_res = zfit.Space('x', -5, 5)  # Narrower resolution window (width=10 < observable width=15)
        
        true_pdf_slice = (prob, edges)
        pdf_conv = doConv(true_pdf_slice, obs_gen, obs_res, process, info, zpars)
        
        # Wrap and extend
        pdf_conv = zfit.pdf.TruncatedPDF(pdf_conv, limits=obs, obs=obs, extended=N)
        try:
            pdf_conv.set_yield(N)
        except Exception:
            pass
        
        return pdf_conv, N
    
    def _build_dscb_convolution_pdf(self,
                                   obs: zfit.Space,
                                   params_tot: List[zfit.Parameter],
                                   process: str,
                                   adv_pars: Dict,
                                   adv_treat: str,
                                   N: zfit.Parameter,
                                   fit_range: Tuple[float, float],
                                   constraints: List) -> Tuple[zfit.pdf.ZPDF, zfit.Parameter]:
        """
        Build PDF using DSCB convolution path.
        Convolves a theory PDF with calibrated DSCB resolution parameters.
        
        Requires in adv_pars['fitpars_in_formatted']:
          - 'theory_pdf': zfit.pdf.ZPDF - the theoretical lineshape
          - 'calib_vals': dict with dscb calibration values
          - 'calib_errs': dict with dscb calibration uncertainties
        """
        module_logger.log(f"[DSCB_CONV] Starting for {process}: obs={obs.obs}, fit_range={fit_range}", 'info')
        theo_pars = adv_pars.get('fitpars_in_formatted', {})
        
        # Extract required components
        theory_pdf = theo_pars.get('theory_pdf')
        calib_vals = theo_pars.get('calib_vals')
        calib_errs = theo_pars.get('calib_errs')
        
        if theory_pdf is None:
            raise ValueError(f"DSCB convolution requires 'theory_pdf' in fitpars_in_formatted")
        
        if calib_vals is None or calib_errs is None:
            calib_vals, calib_errs = self._load_default_calibration()
        
        # Create DSCB resolution parameters with calibration constraints
        mu_param = zfit.Parameter(
            f'res_mu_{process}',
            calib_vals['dscb']['mu'],
            calib_vals['dscb']['mu'] - 5 * calib_errs['dscb']['mu'],
            calib_vals['dscb']['mu'] + 5 * calib_errs['dscb']['mu']
        )
        
        sigma_param = zfit.Parameter(
            f'res_sigma_{process}',
            calib_vals['dscb']['sigma'],
            calib_vals['dscb']['sigma'] - 5 * calib_errs['dscb']['sigma'],
            calib_vals['dscb']['sigma'] + 5 * calib_errs['dscb']['sigma']
        )
        
        # Shape parameters usually fixed during physics fit
        alphal_param = zfit.Parameter(
            f'res_alphal_{process}',
            calib_vals['dscb']['alphaL'],
            floating=False
        )
        alphar_param = zfit.Parameter(
            f'res_alphar_{process}',
            calib_vals['dscb']['alphaR'],
            floating=False
        )
        nl_param = zfit.Parameter(
            f'res_nl_{process}',
            calib_vals['dscb']['nL'],
            floating=False
        )
        nr_param = zfit.Parameter(
            f'res_nr_{process}',
            calib_vals['dscb']['nR'],
            floating=False
        )
        
        # Add resolution parameters to fit parameters
        if adv_treat == 'float':
            params_tot.extend([mu_param, sigma_param])
        
        # Create Gaussian constraints for calibration-based uncertainty
        constraint_mu = zfit.constraint.GaussianConstraint(
            params=mu_param,
            observation=calib_vals['dscb']['mu'],
            uncertainty=calib_errs['dscb']['mu']
        )
        constraint_sigma = zfit.constraint.GaussianConstraint(
            params=sigma_param,
            observation=calib_vals['dscb']['sigma'],
            uncertainty=calib_errs['dscb']['sigma']
        )
        
        # Add constraints if provided
        if constraints is not None:
            constraints.extend([constraint_mu, constraint_sigma])
        
        # For FFTConvPDFV1, both theory_pdf and res_pdf must use the SAME observable.
        # The input obs might be 1D (mom) or 2D (mom, time), so extract just the momentum observable
        obs_name = obs.obs[0] if isinstance(obs.obs, tuple) else obs.obs
        
        # Extract fit range globally (from process.py args or defaults)
        if fit_range is None:
            lo, hi = 95, 115  # Standard physics range (matches AnaProcessor default)
            module_logger.log(f"[DSCB_CONV] fit_range is None, using defaults: [{lo}, {hi}]", 'info')
        else:
            lo, hi = fit_range
            module_logger.log(f"[DSCB_CONV] Using passed fit_range: [{lo}, {hi}]", 'info')
        
        module_logger.log(f"[DSCB_CONV] DIO convolution: fit_range=[{lo}, {hi}], obs_name={obs_name}", 'info')
        
        # IMPORTANT: Handle theory_pdf FIRST to determine observable range from histogram edges
        # This is critical for FFTConvPDFV1 which needs data to cover the full convolution range
        if isinstance(theory_pdf, tuple):
            # theory_pdf is histogram data (prob, edges) - use histogram's actual edge range
            from helper import make_HistogramPDF
            prob, edges = theory_pdf
            
            # Use FIT RANGE for observable, NOT histogram range
            # This allows evaluation across [95, 115], and the histogram naturally pads with 0 outside its data
            edge_min, edge_max = float(edges[0]), float(edges[-1])
            obs_1d = zfit.Space(obs_name, limits=(lo, hi))
            
            module_logger.log(f"[DSCB_CONV] Theory histogram edge range: [{edge_min:.2f}, {edge_max:.2f}]", 'info')
            module_logger.log(f"[DSCB_CONV] Observable set to fit_range: [{lo}, {hi}]", 'info')
            
            # Histogram will return 0 outside its data range - this is expected behavior
            theory_pdf_conv = make_HistogramPDF(prob, edges)(obs=obs_1d)
        else:
            # theory_pdf is a zfit.pdf.ZPDF - use fit range for observable
            obs_1d = zfit.Space(obs_name, limits=(lo, hi))
            
            # Use theory PDF as-is or re-obs if needed
            if hasattr(theory_pdf, 'obs'):
                if theory_pdf.obs.obs != (obs_name,):
                    try:
                        theory_pdf_conv = theory_pdf.with_obs(obs_1d)
                    except Exception:
                        module_logger.log(f"Could not re-obs theory_pdf to '{obs_name}', using as-is", 'warning')
                        theory_pdf_conv = theory_pdf
                else:
                    theory_pdf_conv = theory_pdf
            else:
                theory_pdf_conv = theory_pdf
        
        # Create resolution PDF space using same observable as theory_pdf
        res_range = (-7, 7)
        lo_res, hi_res = res_range
        obs_res = zfit.Space(obs_name, limits=res_range)
        
        module_logger.log(f"[DSCB_CONV] Observable: {obs_name}, resolution range: {res_range}", 'info')
        module_logger.log(f"[DSCB_CONV] Calibration: mu={mu_param.numpy():.6f}, sigma={sigma_param.numpy():.6f}", 'info')
        
        # Build DSCB resolution PDF
        res_pdf = zfit.pdf.DoubleCB(
            mu=mu_param,
            sigma=sigma_param,
            alphal=alphal_param,
            alphar=alphal_param,
            nl=nl_param,
            nr=nr_param,
            obs=obs_res
        )
        
        # Setup convolution spaces
        # Use fit range [lo, hi] for observable
        # FFTConvPDFV1 will work on this range, with histogram padding 0 outside its data
        obs_conv = obs_1d  # This is already set to fit_range (lo, hi)
        obs_full = obs_1d  # Normalization on same range
        
        module_logger.log(f"[DSCB_CONV] FFTConvPDFV1 observable: [{lo}, {hi}]", 'info')
        
        # Build convolution using the observable where we have actual data
        pdf_conv = zfit.pdf.FFTConvPDFV1(
            func=theory_pdf_conv,
            kernel=res_pdf,
            n=1024,
            obs=obs_conv,
            norm=obs_full
        )
        
        module_logger.log(f"[DSCB_CONV] FFTConvPDFV1 created successfully", 'info')
        
        # Set yield on convolved PDF
        try:
            pdf_conv.set_yield(N)
        except Exception:
            pass
        
        return pdf_conv, N
    
    def _load_default_calibration(self) -> Tuple[Dict, Dict]:
        """
        Load default calibration values and uncertainties from copyfiles.
        Returns: (calib_vals, calib_errs)
        """
        import json
        import os
        
        calib_path = os.path.join(
            os.path.dirname(__file__),
            'copyfiles',
            'calibration.json'
        )
        calib_err_path = os.path.join(
            os.path.dirname(__file__),
            'copyfiles',
            'calibration_errors.json'
        )
        
        try:
            with open(calib_path, 'r') as f:
                calib_vals = json.load(f)
            with open(calib_err_path, 'r') as f:
                calib_errs = json.load(f)
            module_logger.log(f"Loaded default calibration from {calib_path}")
            return calib_vals, calib_errs
        except FileNotFoundError as e:
            module_logger.log(f"Could not load calibration files: {e}",'error')
            # Return minimal defaults
            return {
                'dscb': {
                    'mu': -0.560698,
                    'sigma': 0.268712,
                    'alphaL': 0.46302,
                    'alphaR': 2.67316,
                    'nL': 4.37511,
                    'nR': 2.75572
                }
            }, {
                'dscb': {
                    'mu': 0.0086,
                    'sigma': 0.0057,
                    'alphaL': 0.02,
                    'alphaR': 0.14,
                    'nL': 0.33,
                    'nR': 0.63
                }
            }
    
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
        """Initialize with time-specific defaults and physics components."""
        super().__init__(
            default_model_params=time_default_model_params,
            default_norms=time_default_norms,
            physics_components=time_components  # Use physics components for norm lookup
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
