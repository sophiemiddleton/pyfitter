#!/usr/bin/env python3
"""
Phase 2 (Alternative): Profile Likelihood Systematic Uncertainties

Measure systematic impacts using proper profile likelihood method:
  1. For each systematic, fix parameter to nominal ± 1σ via treat_params='fix'
  2. Refit N_CE and all other free parameters (with background constraints)
  3. Measure Δ N_CE from likelihood
  
This is publication-ready. Combines best of both worlds:
  - Fixed parameters (profile likelihood) for systematic variation
  - Constraints on other parameters (background yields) for stability
  
Usage
-----
# Run single systematic (both +/- directions) using 2D fit
python run_profile_systematics.py --systematic DIO_Theory --data data.npz --fittype 2D

# Run all implemented systematics
python run_profile_systematics.py --all-implemented --data data.npz

# Run all shift/frac/shape systematics
python run_profile_systematics.py --all-momentum --data data.npz

# Collect and summarize results
python run_profile_systematics.py --impact-summary
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple logger fallback
class SimpleLogger:
    def __init__(self, verbosity=1):
        self.verbosity = verbosity
    
    def log(self, msg, level='info'):
        prefix = f"[{level.upper():7}]"
        if level == 'success':
            print(f"✓ {msg}")
        elif level == 'error':
            print(f"✗ {msg}", file=sys.stderr)
        elif level == 'warning':
            print(f"⚠ {msg}")
        elif self.verbosity > 0:
            print(f"{prefix} {msg}")

logger = SimpleLogger(verbosity=1)

from uncertainties.model.sysunc_components import sysunc_components, get_implemented_systematics, get_systematics_by_component
import copy
from model import physics_components
from uncertainties.waterfall_plotter import WaterfallPlotter


# Mapping of fit parameter names to (component_name, component_dict) for dynamic treat_params modification
PARAM_TO_COMPONENT = {
    # Yield parameters
    'N_CE': ('CE', 'mom'),
    'N_DIO': ('DIO', 'mom'),
    'N_RPC': ('RPC', 'mom'),
    'N_Cosmic': ('Cosmic', 'mom'),
    
    # Momentum shape parameters
    'c1_RPC': ('RPC', 'mom'),
    'c2_RPC': ('RPC', 'mom'),
    'c1_Cosmic': ('Cosmic', 'mom'),
    'c2_Cosmic': ('Cosmic', 'mom'),
    
    # Time parameters
    'decay_rate_mu': ('Muon', 'time'),
    'decay_rate_pi': ('RPC', 'time'),
}


class ProfileLikelihoodRunner:
    """
    Phase 2 (Profile Likelihood): Measure systematic impacts using proper profile likelihood.
    
    For each systematic:
      1. Extract parameter value and uncertainty from spec
      2. Fix parameter to nominal ± 1σ via treat_params='fix' in physics_components
      3. Refit N_CE (with constraints on other yields/parameters)
      4. Measure Δ N_CE
      
    Hybrid approach: Fixed parameters for systematic but constrained backgrounds for stability.
    """
    
    def __init__(self, data_file: str, output_dir: str = 'uncertainties/outputs', 
                 fit_type: str = '2D', verbose: int = 1):
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fit_type = fit_type
        self.logger = SimpleLogger(verbosity=verbose)
        self.baseline_data = None
        self.baseline_result = None
        self.results = {}
        
    def load_baseline_data(self) -> Dict[str, np.ndarray]:
        """Load baseline data from NPZ file."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        try:
            data = np.load(self.data_file, allow_pickle=True)
            self.baseline_data = {
                'mom_mag': data['mom_mag'],
                'track_cat': data.get('track_cat', np.array([])),
            }
            
            # Load time data if using 2D fit
            if self.fit_type == '2D' and 'time' in data:
                self.baseline_data['time'] = data['time']
                self.logger.log(f"Loaded time data for 2D fit", 'info')
            elif self.fit_type == '2D' and 'time' not in data:
                raise ValueError(f"2D fit requested but NPZ file {self.data_file} does not contain 'time' data.")
            
            self.logger.log(f"Loaded baseline data from {self.data_file}", 'info')
            self.logger.log(f"  Momentum points: {len(self.baseline_data['mom_mag'])}", 'info')
            if self.fit_type == '2D' and 'time' in self.baseline_data:
                self.logger.log(f"  Time points: {len(self.baseline_data['time'])}", 'info')
            return self.baseline_data
        except Exception as e:
            self.logger.log(f"Failed to load {self.data_file}: {e}", 'error')
            raise
    
    def fit_baseline(self, fit_range: Tuple[float, float] = (95, 115), 
                     fit_range_time: Tuple[float, float] = (700, 1700)) -> Dict[str, Any]:
        """
        Run baseline fit WITH CONSTRAINTS (so background yields are constrained properly).
        All parameters free except those with treat_params='fix' in physics_components.
        
        Returns dict with:
          - signal_yield (N_CE value)
          - uncertainty (stat)
          - result (full zfit result)
          - params_dict (dict of all fitted parameters for reference)
        """
        if self.baseline_data is None:
            raise RuntimeError("Call load_baseline_data() first")
        
        self.logger.log(f"Running baseline {self.fit_type} fit (WITH constraints)...", 'info')
        
        try:
            from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
            
            if self.fit_type == '2D':
                result, poi, loss, _, _ = Unbinned_2d_fit_mom_time(
                    self.baseline_data['mom_mag'],
                    self.baseline_data.get('time'),
                    4,
                    fit_range_mom=fit_range,
                    fit_range_time=fit_range_time,
                    constraints_dir='uncertainties/outputs',  # USE constraints
                    verbose=0,
                    plot_results=False
                )
            else:  # 1d
                result, poi, loss, _, _, _ = Unbinned_fit_mom(
                    self.baseline_data['mom_mag'],
                    4,
                    fit_range[0],
                    fit_range[1],
                    constraints_dir='uncertainties/outputs',  # USE constraints
                    verbose=0,
                    plot_results=False
                )
            
            # Get uncertainties
            poi_uncertainty = 0.0
            try:
                errors = result.errors()
                if poi in errors:
                    poi_uncertainty = float(errors[poi])
            except Exception:
                pass
            
            # Collect all fitted parameters for reference
            params_dict = {}
            for param in loss.get_params():
                params_dict[param.name] = float(param.value())
            
            self.baseline_result = {
                'poi_value': float(poi.value()),
                'poi_uncertainty': poi_uncertainty,
                'result': result,
                'params': params_dict,
                'loss': loss,
            }
            
            self.logger.log(f"Baseline N_CE: {self.baseline_result['poi_value']:.1f} ± {self.baseline_result['poi_uncertainty']:.2f}", 'success')
            return self.baseline_result
            
        except Exception as e:
            self.logger.log(f"Baseline fit failed: {e}", 'error')
            raise
    
    def run_profile_scan(self, systematic_name: str, direction: str = 'plus',
                        fit_range: Tuple[float, float] = (95, 115),
                        fit_range_time: Tuple[float, float] = (700, 1700)) -> Optional[Dict[str, Any]]:
        """
        Run profile likelihood scan for one systematic direction:
          1. Get parameter name and varied value from spec
          2. Fix parameter to that value using treat_params='fix' in physics_components
          3. Refit N_CE (with constraints on other yields/parameters)
          4. Measure Δ N_CE
        
        Returns dict with impact metrics, or None if failed.
        """
        if self.baseline_result is None:
            raise RuntimeError("Call fit_baseline() first")
        
        self.logger.log(f"Profile scan: {systematic_name} ({direction})...", 'info')
        
        try:
            from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
            
            spec = sysunc_components.get(systematic_name)
            if spec is None:
                raise ValueError(f"Unknown systematic: {systematic_name}")
            
            # Determine what parameter to fix and what value to use
            param_name, param_varied_value = self._get_parameter_and_value(systematic_name, direction)
            
            self.logger.log(f"  Fixing {param_name} = {param_varied_value:.4f}", 'info')
            
            # Modify physics_components to set parameter fixed with correct initial value
            original_state = self._set_parameter_fixed_with_value(param_name, param_varied_value)
            
            try:
                # Run fit with parameter fixed via treat_params='fix' and initial value set
                # STILL USE CONSTRAINTS on other parameters
                if self.fit_type == '2D':
                    result, poi, loss, _, _ = Unbinned_2d_fit_mom_time(
                        self.baseline_data['mom_mag'],
                        self.baseline_data.get('time'),
                        4,
                        fit_range_mom=fit_range,
                        fit_range_time=fit_range_time,
                        constraints_dir='uncertainties/outputs',  # KEEP constraints on other params
                        verbose=0,
                        plot_results=False
                    )
                else:  # 1d
                    result, poi, loss, _, _, _ = Unbinned_fit_mom(
                        self.baseline_data['mom_mag'],
                        4,
                        fit_range[0],
                        fit_range[1],
                        constraints_dir='uncertainties/outputs',  # KEEP constraints on other params
                        verbose=0,
                        plot_results=False
                    )
            finally:
                # Restore original treat_params and pardict
                self._restore_parameter_fixed_with_value(param_name, original_state)
            
            # Get value and uncertainties
            poi_value = float(poi.value())
            poi_uncertainty = 0.0
            
            try:
                errors = result.errors()
                if poi in errors:
                    poi_uncertainty = float(errors[poi])
            except Exception:
                pass
            
            # Compute impact metrics
            shift = poi_value - self.baseline_result['poi_value']
            shift_sigma = shift / self.baseline_result['poi_uncertainty'] if self.baseline_result['poi_uncertainty'] > 0 else 0
            
            results = {
                'systematic': systematic_name,
                'type': spec.get('type'),
                'direction': direction,
                'baseline_poi': self.baseline_result['poi_value'],
                'varied_poi': poi_value,
                'poi_shift': shift,
                'poi_shift_sigma': shift_sigma,
                'baseline_unc': self.baseline_result['poi_uncertainty'],
                'varied_unc': poi_uncertainty,
                'unc_change': poi_uncertainty - self.baseline_result['poi_uncertainty'],
                'fixed_param': param_name,
                'fixed_value': param_varied_value,
                'baseline_value': self.baseline_result['params'].get(param_name, None),
            }
            
            self.logger.log(
                f"  N_CE: {self.baseline_result['poi_value']:.1f} → {poi_value:.1f} "
                f"(shift: {shift:+.2f}, σ: {shift_sigma:+.2f})",
                'info'
            )
            
            return results
            
        except Exception as e:
            self.logger.log(f"Failed to run {systematic_name} ({direction}): {e}", 'error')
            import traceback
            self.logger.log(traceback.format_exc(), 'error')
            return None
    
    def _get_parameter_and_value(self, systematic_name: str, direction: str) -> Tuple[str, float]:
        """
        Determine what parameter to fix and what value to use.
        
        Returns: (param_name, varied_value)
        """
        spec = sysunc_components.get(systematic_name)
        plus_var, minus_var = spec['value']
        variation = plus_var if direction == 'plus' else -minus_var
        
        sys_type = spec.get('type')
        
        if sys_type == 'frac':
            # Fractional: varies a yield parameter
            param_name = spec.get('fit_param', systematic_name)
            nominal_yield = self.baseline_result['params'].get(param_name, 1.0)
            varied_value = nominal_yield * (1 + variation)
            
        elif sys_type == 'shape':
            # Shape: varies a spectrum parameter (c1_RPC, c1_Cosmic, etc.)
            param_name = spec.get('fit_param', systematic_name)
            param_value = spec.get('param_value', 0.0)
            # For shape, variation is usually a Gaussian smearing sigma
            # We fix the parameter to nominal ± sigma
            varied_value = param_value + variation if direction == 'plus' else param_value - variation
            
        elif sys_type == 'shift':
            # Shift: varies momentum - don't fix a parameter, instead data is shifted
            # For this, we'd need to modify the data, but profile likelihood approach
            # typically doesn't work for data-level shifts
            raise NotImplementedError(f"Shift systematics (e.g., {systematic_name}) need data modification, "
                                     "not parameter fixing. Use run_systematic_variation.py instead.")
        else:
            raise ValueError(f"Unknown systematic type: {sys_type}")
        
        return param_name, varied_value
    
    def _set_parameter_fixed_with_value(self, param_name: str, varied_value: float) -> Optional[Tuple[str, Any]]:
        """
        Temporarily modify physics_components to:
          1. Set treat_params='fix' for the component containing this parameter
          2. Set the parameter's initial value in the pardict to the varied value
        
        Returns tuple of (original_treat_params, original_pardict_spec) for restoration.
        """
        if param_name not in PARAM_TO_COMPONENT:
            raise ValueError(f"Unknown parameter: {param_name}. "
                            f"Available: {list(PARAM_TO_COMPONENT.keys())}")
        
        comp_name, comp_type = PARAM_TO_COMPONENT[param_name]
        
        # Get the component dictionary
        if comp_type == 'mom':
            comp_dict = physics_components.mom_components
        elif comp_type == 'time':
            comp_dict = physics_components.time_components
        else:
            raise ValueError(f"Unknown component type: {comp_type}")
        
        if comp_name not in comp_dict:
            raise ValueError(f"Component {comp_name} not found in {comp_type}_components")
        
        comp_spec = comp_dict[comp_name]
        
        # Save original treat_params
        original_treat_params = comp_spec.get('treat_params', 'float')
        original_pardict_spec = None
        
        # Set treat_params='fix'
        comp_spec['treat_params'] = 'fix'
        self.logger.log(f"Set {comp_name} treat_params='fix'", 'info')
        
        # For parameters, update the pardict with the varied value as initial value
        pars_dict = comp_spec.get('pars', {})
        
        if param_name.startswith('N_'):
            # Yield parameter: modify pardict['N']
            if 'N' in pars_dict:
                original_pardict_spec = pars_dict['N']
                # Keep original bounds but set initial value to varied_value
                if isinstance(original_pardict_spec, tuple) and len(original_pardict_spec) == 3:
                    lower_bound = original_pardict_spec[1]
                    upper_bound = original_pardict_spec[2]
                else:
                    # Fallback if format is different
                    lower_bound, upper_bound = 0, 1e6
                pars_dict['N'] = (varied_value, lower_bound, upper_bound)
                self.logger.log(f"Set {param_name} initial value = {varied_value:.4f}", 'info')
        else:
            # Shape/spectrum parameter (c1, c2, decay_rate, etc.)
            if param_name in pars_dict:
                original_pardict_spec = pars_dict[param_name]
                # Keep original bounds but set initial value to varied_value
                if isinstance(original_pardict_spec, tuple) and len(original_pardict_spec) == 3:
                    lower_bound = original_pardict_spec[1]
                    upper_bound = original_pardict_spec[2]
                else:
                    # Fallback if format is different
                    lower_bound, upper_bound = varied_value - 1, varied_value + 1
                pars_dict[param_name] = (varied_value, lower_bound, upper_bound)
                self.logger.log(f"Set {param_name} initial value = {varied_value:.4f}", 'info')
        
        return (original_treat_params, (param_name, original_pardict_spec))
    
    def _restore_parameter_fixed_with_value(self, param_name: str, original_state: Optional[Tuple[str, Tuple]]):
        """
        Restore the original treat_params and pardict values for a parameter.
        """
        if original_state is None:
            return
        
        original_treat_params, (saved_param_name, original_pardict_spec) = original_state
        
        if param_name not in PARAM_TO_COMPONENT:
            return
        
        comp_name, comp_type = PARAM_TO_COMPONENT[param_name]
        
        # Get the component dictionary
        if comp_type == 'mom':
            comp_dict = physics_components.mom_components
        elif comp_type == 'time':
            comp_dict = physics_components.time_components
        else:
            return
        
        if comp_name not in comp_dict:
            return
        
        comp_spec = comp_dict[comp_name]
        
        # Restore treat_params
        comp_spec['treat_params'] = original_treat_params
        self.logger.log(f"Restored {comp_name} treat_params='{original_treat_params}'", 'info')
        
        # Restore pardict if it was modified
        if original_pardict_spec is not None:
            pars_dict = comp_spec.get('pars', {})
            if param_name.startswith('N_') and 'N' in pars_dict:
                pars_dict['N'] = original_pardict_spec
            elif param_name in pars_dict:
                pars_dict[param_name] = original_pardict_spec
            self.logger.log(f"Restored {param_name} pardict value", 'info')
    
    def run_multiple_systematics(self, systematics: list, 
                                fit_range: Tuple[float, float] = (95, 115),
                                fit_range_time: Tuple[float, float] = (700, 1700)):
        """Run profile scans for multiple systematics."""
        self.logger.log(f"Running {len(systematics)} systematics (both directions)...", 'info')
        
        for sys_name in systematics:
            if sys_name not in sysunc_components:
                self.logger.log(f"Skipping unknown systematic: {sys_name}", 'warning')
                continue
            
            for direction in ['plus', 'minus']:
                try:
                    result = self.run_profile_scan(sys_name, direction, fit_range, fit_range_time)
                    if result:
                        self.results[f"{sys_name}_{direction}"] = result
                except Exception as e:
                    self.logger.log(f"Error: {e}", 'error')
    
    def save_results(self) -> Path:
        """Save all profile scan results to JSON."""
        results_file = self.output_dir / 'profile_systematic_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.logger.log(f"Saved results to {results_file}", 'info')
        return results_file
    
    def produce_impact_summary(self) -> str:
        """
        Produce a summary table of impacts.
        
        Returns formatted string with impact metrics.
        """
        if not self.results:
            return "No results to summarize"
        
        # Group by systematic (average +/- directions)
        impacts = defaultdict(list)
        for key, res in self.results.items():
            if res is None:
                continue
            sys_name = res['systematic']
            impacts[sys_name].append(abs(res['poi_shift']))
        
        # Sort by impact
        sorted_impacts = sorted(impacts.items(), key=lambda x: np.mean(x[1]), reverse=True)
        
        lines = [
            "\n" + "=" * 70,
            "PROFILE LIKELIHOOD SYSTEMATIC IMPACT SUMMARY",
            "=" * 70,
            f"Baseline N_CE: {self.baseline_result['poi_value']:.1f} ± {self.baseline_result['poi_uncertainty']:.2f}",
            "",
            f"{'Systematic':<30} {'Impact (avg)':<15} {'Relative':<10}",
            "-" * 70,
        ]
        
        total_impact_sq = 0
        for sys_name, shifts in sorted_impacts:
            avg_shift = np.mean(shifts)
            rel_impact = avg_shift / self.baseline_result['poi_uncertainty'] if self.baseline_result['poi_uncertainty'] > 0 else 0
            lines.append(f"{sys_name:<30} {avg_shift:>6.2f} events    {rel_impact:>6.2f} σ")
            total_impact_sq += avg_shift ** 2
        
        total_impact = np.sqrt(total_impact_sq)
        lines.extend([
            "=" * 70,
            f"{'Total (quad sum)':<30} {total_impact:>6.2f} events",
            "=" * 70,
            f"\nFinal uncertainty on N_CE:",
            f"  Stat:       ±{self.baseline_result['poi_uncertainty']:.2f} events",
            f"  Syst:       ±{total_impact:.2f} events",
            f"  Total:      ±{np.sqrt(self.baseline_result['poi_uncertainty']**2 + total_impact**2):.2f} events",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def generate_plots(self) -> Dict[str, Path]:
        """
        Generate publication-quality plots (waterfall, etc.).
        
        Returns dict of {plot_type: output_path}
        """
        if not self.results:
            self.logger.log("No results to plot", 'warning')
            return {}
        
        plotter = WaterfallPlotter(output_dir=str(self.output_dir))
        plots = {}
        
        try:
            # Generate waterfall plot
            stat_unc = self.baseline_result['poi_uncertainty']
            output_file = plotter.plot_waterfall(
                self.results,
                stat_uncertainty=stat_unc,
                title=f'Profile Likelihood Systematic Impacts (Baseline N_CE = {self.baseline_result["poi_value"]:.1f})',
                output_file='profile_waterfall.png'
            )
            plots['waterfall'] = output_file
            self.logger.log(f"Waterfall plot saved: {output_file}", 'success')
        except Exception as e:
            self.logger.log(f"Failed to generate waterfall plot: {e}", 'error')
        
        return plots


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Profile likelihood systematic uncertainties',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--data', type=str, default='MDS3c_mom_mag.npz',
                       help='Data file (default: MDS3c_mom_mag.npz)')
    parser.add_argument('--fittype', type=str, default='2D', choices=['mom1D', '2D'],
                       help='Fit type (default: 2D)')
    parser.add_argument('--systematic', type=str,
                       help='Single systematic to scan')
    parser.add_argument('--direction', type=str, choices=['plus', 'minus'],
                       help='Only run one direction (default: both)')
    parser.add_argument('--all-implemented', action='store_true',
                       help='Run all implemented systematics')
    parser.add_argument('--all-momentum', action='store_true',
                       help='Run all momentum-affecting systematics')
    parser.add_argument('--fit-range-mom', type=float, nargs=2, default=[95, 115],
                       help='Momentum range (default: 95 115)')
    parser.add_argument('--fit-range-time', type=float, nargs=2, default=[700, 1700],
                       help='Time range (default: 700 1700)')
    parser.add_argument('--impact-summary', action='store_true',
                       help='Show impact summary and exit')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                       help='Verbosity')
    
    args = parser.parse_args()
    
    # Handle summary mode
    if args.impact_summary:
        results_file = Path('uncertainties/outputs/profile_systematic_results.json')
        if not results_file.exists():
            print("No results file found. Run systematics first.")
            sys.exit(1)
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Reconstruct runner just to use summary method
        runner = ProfileLikelihoodRunner(args.data, fit_type=args.fittype, verbose=args.verbose)
        runner.results = results
        # Load baseline info (hacky, but works)
        baseline_file = Path('uncertainties/outputs/baseline_result.json')
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline_info = json.load(f)
                runner.baseline_result = baseline_info
        
        print(runner.produce_impact_summary())
        
        # Generate plots
        logger.log("Generating publication plots...", 'info')
        plots = runner.generate_plots()
        for plot_type, path in plots.items():
            logger.log(f"  {plot_type}: {path}", 'success')
        return
    
    # Main workflow
    runner = ProfileLikelihoodRunner(args.data, fit_type=args.fittype, verbose=args.verbose)
    
    try:
        runner.load_baseline_data()
        runner.fit_baseline(tuple(args.fit_range_mom), tuple(args.fit_range_time))
        
        # Save baseline result
        with open(runner.output_dir / 'baseline_result.json', 'w') as f:
            json.dump({
                'poi_value': runner.baseline_result['poi_value'],
                'poi_uncertainty': runner.baseline_result['poi_uncertainty'],
                'params': runner.baseline_result['params']
            }, f, indent=2)
        
        # Determine which systematics to run
        to_run = []
        if args.systematic:
            to_run = [args.systematic]
        elif args.all_implemented:
            to_run = list(get_implemented_systematics().keys())
        elif args.all_momentum:
            to_run = list(get_systematics_by_component('mom').keys())
        else:
            print("Specify --systematic, --all-implemented, or --all-momentum")
            sys.exit(1)
        
        # Run scans
        if args.direction:
            # Single direction
            for sys_name in to_run:
                result = runner.run_profile_scan(sys_name, args.direction, 
                                                tuple(args.fit_range_mom), 
                                                tuple(args.fit_range_time))
                if result:
                    runner.results[f"{sys_name}_{args.direction}"] = result
        else:
            # Both directions
            runner.run_multiple_systematics(to_run, tuple(args.fit_range_mom), 
                                           tuple(args.fit_range_time))
        
        # Save and summarize
        runner.save_results()
        print(runner.produce_impact_summary())
        
        # Generate plots
        logger.log("Generating publication plots...", 'info')
        plots = runner.generate_plots()
        for plot_type, path in plots.items():
            logger.log(f"  {plot_type}: {path}", 'success')
        
    except Exception as e:
        logger.log(f"Fatal error: {e}", 'error')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
