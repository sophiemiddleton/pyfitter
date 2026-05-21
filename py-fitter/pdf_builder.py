# pdf_builder.py: Generic PDF Builder Interface and Utilities

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional, Any
import zfit
from pyutils.pylogger import Logger

# Module-level logger
try:
    logger = Logger(print_prefix='[pdf_builder] ', verbosity=2)
except Exception:
    logger = None


class PDFBuilder(ABC):
    """
    Abstract base class for building zfit PDF models with standardized parameter handling.
    
    This class encapsulates the common workflow:
    1. Extract/create normalization parameter
    2. Extract and process model parameters
    3. Handle parameter reuse from existing parameter lists
    4. Build the PDF object
    5. Return (PDF, normalization)
    """
    
    def __init__(self, 
                 default_model_params: Dict[str, Dict] = None,
                 default_norms: Dict[str, float] = None):
        """
        Initialize the builder with default parameters.
        
        Args:
            default_model_params: Dict mapping model names to parameter defaults
                                 Format: {'model_name': {'param': (value, lo, hi), ...}}
            default_norms: Dict mapping process names to default normalization values
                          Format: {'process_name': value}
        """
        self.default_model_params = default_model_params or {}
        self.default_norms = default_norms or {}
    
    def build(self,
              obs: zfit.Space,
              params_tot: List[zfit.Parameter],
              process: str,
              model: str,
              pardict: Optional[Dict] = None,
              treat_params: str = 'float',
              fit_range: Tuple[float, float] = None,
              constraints: List = None,
              **kwargs) -> Tuple[zfit.pdf.ZPDF, zfit.Parameter]:
        """
        Build a PDF for the given process and model.
        
        Args:
            obs: zfit.Space observable
            params_tot: List to accumulate all zfit.Parameter objects
            process: Physics process name (e.g., 'CE', 'DIO', 'Cosmic')
            model: PDF model type (e.g., 'dscb', 'Gauss', 'uniform')
            pardict: Parameter override dictionary
            treat_params: How to treat parameters - 'float', 'fix', or 'constrain'
            fit_range: (low, high) tuple for fit range
            constraints: List to accumulate constraints (for 'constrain' mode)
            **kwargs: Additional builder-specific arguments
            
        Returns:
            Tuple of (built_pdf, normalization_parameter)
        """
        if constraints is None:
            constraints = []
        
        # Step 1: Extract or create normalization parameter
        N = self._get_normalization(process, params_tot, pardict)
        params_tot.append(N)
        
        # Step 2: Extract and process model parameters
        params = self._extract_model_params(model, pardict, process)
        
        # Step 3: Create zfit parameters with proper reuse
        zpars = self._create_zfit_params(
            params, process, params_tot, treat_params, constraints
        )
        zpars['N'] = N
        
        # Step 4: Build the PDF (delegated to subclass)
        pdf = self._build_pdf(obs, model, zpars, process, fit_range, **kwargs)
        
        return pdf, N
    
    def _get_normalization(self,
                          process: str,
                          params_tot: List[zfit.Parameter],
                          pardict: Optional[Dict] = None) -> zfit.Parameter:
        """
        Extract or create the normalization (yield) parameter.
        
        Searches params_tot for existing N_<process> parameter before creating new one.
        Respects special bounds for specific processes (e.g., CE).
        """
        pname = f'N_{process}'
        
        # Check if N already exists in params_tot
        existing_N = self._find_param_by_name(pname, params_tot)
        if existing_N is not None:
            return existing_N
        
        # Check for N in pardict
        if isinstance(pardict, dict) and 'N' in pardict:
            return zfit.Parameter(
                pname,
                pardict['N'][0],
                pardict['N'][1],
                pardict['N'][2]
            )
        
        # Use process-specific defaults with custom bounds if defined
        default_value = self.default_norms.get(process, 10.0)
        bounds = self._get_normalization_bounds(process)
        
        return zfit.Parameter(pname, default_value, bounds[0], bounds[1])
    
    def _get_normalization_bounds(self, process: str) -> Tuple[float, float]:
        """
        Get custom normalization bounds for a process.
        Can be overridden in subclasses for process-specific logic.
        """
        return (0.0, 1e6)
    
    def _extract_model_params(self,
                             model: str,
                             pardict: Optional[Dict] = None,
                             process: str = None) -> Dict[str, Tuple]:
        """
        Extract model parameters, merging defaults with overrides.
        
        Priority: pardict > defaults from default_model_params
        Handles both parameterized and non-parameterized models.
        """
        # Start with defaults for this model
        params = self.default_model_params.get(model, {}).copy()
        
        if pardict is None:
            return params
        
        # Override with values from pardict
        for par, val in pardict.items():
            if par == 'N':
                continue  # N is handled separately
            
            # Exact match
            if par in params:
                params[par] = val
            # Try with process suffix stripped
            elif process and par.endswith(f'_{process}'):
                base = par[:-len(f'_{process}')]
                if base in params:
                    params[base] = val
                    continue
            # Store as-is if new parameter
            else:
                params[par] = val
        
        return params
    
    def _create_zfit_params(self,
                           params: Dict[str, Tuple],
                           process: str,
                           params_tot: List[zfit.Parameter],
                           treat_params: str = 'float',
                           constraints: List = None) -> Dict[str, zfit.Parameter]:
        """
        Create zfit.Parameter objects with proper reuse from params_tot.
        
        Handles three treatment modes:
        - 'float': Parameters float freely within bounds
        - 'fix': Parameters fixed to initial value
        - 'constrain': Parameters constrained with Gaussian constraints
        """
        if constraints is None:
            constraints = []
        
        zpars = {}
        
        for p, val in params.items():
            p_name = f'{p}_{process}'
            
            # Check if parameter already exists
            existing = self._find_param_by_name(p_name, params_tot)
            if existing is not None:
                zpars[p] = existing
                continue
            
            # Create new parameter based on treatment mode
            if treat_params == 'constrain':
                zpars[p] = self._create_constrained_param(
                    p_name, val, constraints
                )
            elif treat_params == 'fix':
                zpars[p] = zfit.Parameter(
                    p_name,
                    val[0],
                    floating=False
                )
            else:  # 'float' (default)
                zpars[p] = zfit.Parameter(
                    p_name,
                    val[0],
                    val[1],
                    val[2]
                )
            
            # Safely append to params_tot
            try:
                params_tot.append(zpars[p])
            except Exception:
                pass
        
        return zpars
    
    def _create_constrained_param(self,
                                 p_name: str,
                                 val: Tuple,
                                 constraints: List) -> zfit.Parameter:
        """
        Create a constrained parameter with Gaussian constraint.
        
        val should be (observed_value, uncertainty_lo, uncertainty_hi)
        Creates symmetric bounds and adds Gaussian constraint.
        """
        obs = float(val[0])
        unc_lo = float(val[1]) if len(val) > 1 else 0.0
        unc_hi = float(val[2]) if len(val) > 2 else unc_lo if unc_lo != 0.0 else 1.0
        
        # Create bounds symmetric around observed value
        lower = obs - 5.0 * abs(unc_lo)
        upper = obs + 5.0 * abs(unc_hi)
        
        # Guard: ensure lower < obs < upper
        if lower >= obs:
            lower = obs - abs(unc_lo) if abs(unc_lo) > 0 else obs - 1e-3
        if upper <= obs:
            upper = obs + abs(unc_hi) if abs(unc_hi) > 0 else obs + 1e-3
        
        param = zfit.Parameter(p_name, obs, lower, upper, step_size=0.0001)
        
        # Add Gaussian constraint
        unc = max(abs(unc_lo), abs(unc_hi))
        constraints.append(
            zfit.constraint.GaussianConstraint(param, observation=obs, uncertainty=unc)
        )
        
        return param
    
    @staticmethod
    def _find_param_by_name(name: str,
                           params_tot: List[zfit.Parameter]) -> Optional[zfit.Parameter]:
        """
        Find a parameter by name in the parameters list.
        """
        for p in params_tot:
            try:
                if getattr(p, 'name', None) == name:
                    return p
            except Exception:
                continue
        return None
    
    @abstractmethod
    def _build_pdf(self,
                   obs: zfit.Space,
                   model: str,
                   zpars: Dict[str, zfit.Parameter],
                   process: str,
                   fit_range: Tuple[float, float] = None,
                   **kwargs) -> zfit.pdf.ZPDF:
        """
        Build and return the PDF object.
        Must be implemented by subclasses for specific model types.
        
        Args:
            obs: zfit.Space observable
            model: PDF model type
            zpars: Dict of parameter name -> zfit.Parameter
            process: Physics process name
            fit_range: Optional (low, high) tuple
            **kwargs: Additional model-specific arguments
            
        Returns:
            zfit.pdf.ZPDF object
        """
        pass


def validate_pdf_config(model: str,
                        default_model_params: Dict[str, Dict],
                        required_params: Dict[str, List[str]]) -> bool:
    """
    Validate that a model has all required parameters defined.
    
    Args:
        model: Model name
        default_model_params: Available model parameter defaults
        required_params: Dict mapping model names to list of required parameter names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if model not in required_params:
        raise ValueError(f"Model {model} not recognized")
    
    if model not in default_model_params:
        raise ValueError(f"No defaults defined for model {model}")
    
    required = set(required_params[model])
    available = set(default_model_params[model].keys())
    
    missing = required - available
    if missing:
        raise ValueError(
            f"Model {model} missing parameters: {missing}"
        )
    
    return True
