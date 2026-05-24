"""
Systematic Uncertainty Propagation Module

Provides methods to apply systematic variations across the full pipeline:
  - Constraint-based: Add Gaussian constraints to fit parameters
  - Refit-based: Run the fit with data modifications (e.g., momentum scale shift)
  - Template-based: Swap PDF shapes (e.g., alternative DIO PDFs)
  - Reweight-based: Apply event weights to modify yields
  - Toy MC: Generate pseudo-experiments with varied nuisance parameters

Each method is tailored to its use case and tracks:
  - Input variation details
  - Output metrics (shifted parameters, changed uncertainties)
  - Correlations with other systematics
"""

import numpy as np
import json
import os
from typing import Dict, Tuple, Any, Optional, List, Callable
from pathlib import Path
from pyutils.pylogger import Logger
from sysunc_components import (
    sysunc_components, 
    validate_sysunc_spec,
    get_constraints_only,
    get_implemented_systematics
)


class UncertaintyPropagator:
    """
    Main class for propagating systematic uncertainties through the analysis.
    
    Supports:
      - Building constraint lists for fits
      - Applying data transformations (momentum shifts, reweighting)
      - Running systematic variation studies
      - Collecting and summarizing results
    """
    
    def __init__(self, output_dir: str = 'uncertainties/outputs', logger: Optional[Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or Logger(print_prefix='[UncertaintyPropagator] ', verbosity=1)
        self.results = {}  # Store results by systematic name
        
    # ========================================================================
    # CONSTRAINT-BASED PROPAGATION
    # ========================================================================
    
    def build_constraints_dict(self, systematics: Optional[List[str]] = None,
                             nominal_yields: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        """
        Build a constraints.json-compatible dict for specified systematics.
        
        Parameters
        ----------
        systematics : list of str, optional
            System names to include. If None, uses all 'constraint' method systematics.
        nominal_yields : dict, optional
            Nominal yields for each parameter (used to scale fractional uncertainties).
            Keys: 'N_DIO', 'N_RPC', 'N_Cosmic', etc.
            If None, fractional uncertainties are skipped.
            
        Returns
        -------
        dict
            List of constraint specs, ready to save to JSON.
        """
        if systematics is None:
            to_include = get_constraints_only()
        else:
            to_include = {k: sysunc_components[k] for k in systematics if k in sysunc_components}
        
        # Group by parameter name and combine uncertainties in quadrature
        param_contributions = {}  # param_name -> list of (sigma_plus, sigma_minus, notes)
        
        for name, spec in to_include.items():
            if spec.get('method') != 'constraint':
                continue
            
            # Map systematic name to parameter name
            param_name = self._map_systematic_to_parameter(name, spec)
            
            # Extract value bounds
            plus_var, minus_var = spec['value']
            
            # Scale fractional uncertainties by nominal yield if available
            if spec.get('type') == 'frac' and nominal_yields is not None:
                if param_name in nominal_yields:
                    nominal = nominal_yields[param_name]
                    plus_var = plus_var * nominal
                    minus_var = minus_var * nominal
                    self.logger.log(f"{name}: Scaled fractional uncertainty by nominal {param_name}={nominal:.1f}", 'info')
                else:
                    self.logger.log(f"{name}: No nominal yield for {param_name}, skipping", 'warn')
                    continue
            
            if param_name not in param_contributions:
                param_contributions[param_name] = []
            
            param_contributions[param_name].append({
                'plus': plus_var,
                'minus': minus_var,
                'sys_name': name,
                'notes': spec.get('notes', '')
            })
        
        # Build final constraints by combining contributions in quadrature
        constraints = []
        for param_name, contributions in param_contributions.items():
            # Combine multiple systematics affecting same parameter in quadrature
            sigma_plus_sq = sum(c['plus']**2 for c in contributions)
            sigma_minus_sq = sum(c['minus']**2 for c in contributions)
            
            sigma_plus = np.sqrt(sigma_plus_sq)
            sigma_minus = np.sqrt(sigma_minus_sq)
            sigma_bar = (sigma_plus + sigma_minus) / 2
            
            # Set mean based on parameter type
            # For yield parameters (N_*), mean should be the nominal yield value
            # For shape parameters, mean should be 0
            if param_name in (nominal_yields or {}):
                mean = nominal_yields[param_name]
            else:
                mean_shift = (sigma_plus - sigma_minus) / 2
                mean = mean_shift
            
            # Combine notes from all contributors
            all_notes = '; '.join(f"{c['sys_name']}: {c['notes']}" for c in contributions)
            
            constraints.append({
                'name': param_name,
                'pname': param_name,
                'prior': {
                    'dist': 'gauss',
                    'mean': mean,
                    'sigma': sigma_bar
                },
                'contributions': [c['sys_name'] for c in contributions],
                'notes': all_notes
            })
        
        return constraints
    
    def save_constraints_json(self, systematics: Optional[List[str]] = None,
                             nominal_yields: Optional[Dict[str, float]] = None,
                             outfile: Optional[str] = None) -> Path:
        """
        Generate and save constraints.json for loading into fitter.
        
        Parameters
        ----------
        systematics : list, optional
        nominal_yields : dict, optional
            Nominal yields for scaling fractional uncertainties.
        outfile : str, optional
            Output path. Default: uncertainties/outputs/constraints_combined.json
            
        Returns
        -------
        Path
            Path to saved JSON file.
        """
        if outfile is None:
            outfile = self.output_dir / 'constraints_combined.json'
        else:
            outfile = Path(outfile)
        
        constraints = self.build_constraints_dict(systematics, nominal_yields=nominal_yields)
        
        with open(outfile, 'w') as f:
            json.dump(constraints, f, indent=2)
        
        self.logger.log(f"Saved {len(constraints)} constraints to {outfile}", 'info')
        return outfile
    
    # ========================================================================
    # DATA TRANSFORMATION-BASED PROPAGATION
    # ========================================================================
    
    def apply_momentum_shift(self, mom_array: np.ndarray, systematic_name: str, 
                            direction: str = 'plus') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply a momentum shift systematic variation.
        
        Parameters
        ----------
        mom_array : np.ndarray
            1D momentum array (MeV/c)
        systematic_name : str
            Name of systematic (must be type='shift')
        direction : str
            'plus' or 'minus' (default: 'plus')
            
        Returns
        -------
        shifted_array, metadata : tuple
            - shifted_array: Modified momentum array
            - metadata: Dict with applied variation details
        """
        spec = sysunc_components.get(systematic_name)
        if spec is None:
            raise ValueError(f"Unknown systematic: {systematic_name}")
        if spec['type'] != 'shift':
            raise ValueError(f"{systematic_name} is type '{spec['type']}', not 'shift'")
        
        plus_val, minus_val = spec['value']
        shift = plus_val if direction == 'plus' else -minus_val
        
        shifted = mom_array + shift
        metadata = {
            'systematic': systematic_name,
            'type': 'momentum_shift',
            'direction': direction,
            'shift_mev': shift,
            'original_mean': float(np.mean(mom_array)),
            'shifted_mean': float(np.mean(shifted))
        }
        
        self.logger.log(f"Applied {systematic_name} ({direction}): shifted by {shift:.3f} MeV", 'info')
        return shifted, metadata
    
    def apply_momentum_smearing(self, mom_array: np.ndarray, systematic_name: str,
                               direction: str = 'plus') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply momentum resolution smearing (Gaussian convolution).
        
        Parameters
        ----------
        mom_array : np.ndarray
        systematic_name : str
            Name of systematic (must be type='shape' with component 'mom')
        direction : str
            'plus' or 'minus'
            
        Returns
        -------
        smeared_array, metadata
        """
        spec = sysunc_components.get(systematic_name)
        if spec is None:
            raise ValueError(f"Unknown systematic: {systematic_name}")
        if spec['type'] != 'shape':
            raise ValueError(f"{systematic_name} is type '{spec['type']}', not 'shape'")
        
        plus_val, minus_val = spec['value']
        sigma = plus_val if direction == 'plus' else minus_val
        
        # Add Gaussian smearing
        smeared = mom_array + np.random.normal(0, sigma, size=mom_array.shape)
        
        metadata = {
            'systematic': systematic_name,
            'type': 'momentum_smearing',
            'direction': direction,
            'sigma_mev': float(sigma),
            'original_std': float(np.std(mom_array)),
            'smeared_std': float(np.std(smeared))
        }
        
        self.logger.log(f"Applied {systematic_name} ({direction}): smeared with σ={sigma:.3f} MeV", 'info')
        return smeared, metadata
    
    def apply_yield_variation(self, yields_dict: Dict[str, float], systematic_name: str,
                             direction: str = 'plus') -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Apply a fractional yield variation (normalization uncertainty).
        
        Parameters
        ----------
        yields_dict : dict
            Map {process: yield_value}
        systematic_name : str
            Name of systematic (must be type='frac')
        direction : str
            'plus' or 'minus'
            
        Returns
        -------
        varied_yields, metadata
        """
        spec = sysunc_components.get(systematic_name)
        if spec is None:
            raise ValueError(f"Unknown systematic: {systematic_name}")
        if spec['type'] != 'frac':
            raise ValueError(f"{systematic_name} is type '{spec['type']}', not 'frac'")
        
        # Get the process this systematic affects
        process = spec['process']
        
        plus_frac, minus_frac = spec['value']
        frac = plus_frac if direction == 'plus' else -minus_frac
        
        varied_yields = yields_dict.copy()
        
        if process == 'all':
            # Scale all processes
            for key in varied_yields:
                varied_yields[key] *= (1 + frac)
        else:
            # Scale specific process
            if process in varied_yields:
                varied_yields[process] *= (1 + frac)
        
        metadata = {
            'systematic': systematic_name,
            'type': 'yield_variation',
            'direction': direction,
            'fraction': float(frac),
            'process': process,
            'original_yields': {k: float(v) for k, v in yields_dict.items()},
            'varied_yields': {k: float(v) for k, v in varied_yields.items()}
        }
        
        self.logger.log(f"Applied {systematic_name} ({direction}): varied {process} by {frac*100:.1f}%", 'info')
        return varied_yields, metadata
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _map_systematic_to_parameter(self, sys_name: str, spec: Dict) -> str:
        """Map systematic name to fit parameter name convention.
        
        First checks spec's 'fit_param' field, then falls back to heuristic mapping.
        """
        # First try: explicit fit_param field in spec
        if 'fit_param' in spec and spec['fit_param'] is not None:
            return spec['fit_param']
        
        # Fallback: heuristic mapping based on systematic name
        process = spec.get('process')
        param_map = {
            'Abs_Mom_Scale': 'mom_scale',
            'Mom_Resolution': 'mom_res',
            'DIO_Theory': 'N_DIO',
            'RPC_Rate': 'N_RPC',
            'Pion_Rate': 'N_RPC',
            'CRV_Efficiency': 'N_Cosmic',
            'Cosmic_Generator': 'N_Cosmic',
            'InternalConv_Rate': 'N_RPC',
            'OOT_RPC': 'N_RPC',
            'CRV_Aging': 'N_Cosmic',
            'c1_Cosmic': 'c1_Cosmic',
            'c2_Cosmic': 'c2_Cosmic',
        }
        return param_map.get(sys_name, sys_name)
    
    def summarize_systematics(self, include_status: Optional[str] = None) -> str:
        """
        Generate a readable summary of all systematics.
        
        Parameters
        ----------
        include_status : str, optional
            Filter by status ('implemented', 'planned', 'on-hold'). If None, show all.
            
        Returns
        -------
        str
            Formatted summary table.
        """
        to_show = sysunc_components.copy()
        if include_status:
            to_show = {k: v for k, v in to_show.items() if v.get('status') == include_status}
        
        lines = [
            "SYSTEMATIC UNCERTAINTIES SUMMARY",
            "=" * 120,
            f"{'Name':<25} {'Type':<10} {'Process':<12} {'Value':<15} {'Status':<12} {'Method':<15}",
            "-" * 120,
        ]
        
        for name, spec in sorted(to_show.items()):
            val_str = f"[+{spec['value'][0]}, -{spec['value'][1]}]"
            lines.append(
                f"{name:<25} {spec['type']:<10} {spec['process']:<12} {val_str:<15} "
                f"{spec['status']:<12} {spec['method']:<15}"
            )
        
        lines.append("=" * 120)
        return "\n".join(lines)
    
    def validate_all_systematics(self) -> Tuple[int, List[str]]:
        """
        Validate all systematic specifications.
        
        Returns
        -------
        n_valid, errors : tuple
            - n_valid: number of valid systematics
            - errors: list of error messages for invalid ones
        """
        errors = []
        n_valid = 0
        
        for name, spec in sysunc_components.items():
            try:
                validate_sysunc_spec(spec)
                n_valid += 1
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
        
        if errors:
            self.logger.log(f"Validation: {n_valid} valid, {len(errors)} invalid", 'warning')
            for err in errors:
                self.logger.log(f"  {err}", 'warning')
        else:
            self.logger.log(f"All {n_valid} systematics validated successfully", 'success')
        
        return n_valid, errors


# ============================================================================
# Convenience functions
# ============================================================================

def print_uncertainty_inventory():
    """Pretty-print the full uncertainty inventory."""
    prop = UncertaintyPropagator()
    print(prop.summarize_systematics())


def print_implemented_only():
    """Show only implemented systematics."""
    prop = UncertaintyPropagator()
    print(prop.summarize_systematics(include_status='implemented'))


def generate_all_constraints(output_dir: str = 'uncertainties/outputs'):
    """Generate constraints.json from all 'constraint' method systematics."""
    prop = UncertaintyPropagator(output_dir=output_dir)
    return prop.save_constraints_json()
