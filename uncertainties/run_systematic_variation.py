#!/usr/bin/env python3
"""
Phase 2: Run Systematic Uncertainty Variations

Run the full pipeline with systematic variations to measure impacts.

Usage
-----
# Run a single systematic (both +/- directions) using 2D fit
python run_systematic_variation.py --systematic Abs_Mom_Scale --data data.npz

# Run using 1D momentum fit instead
python run_systematic_variation.py --systematic Abs_Mom_Scale --data data.npz --fittype mom1D

# Run all momentum-affecting systematics
python run_systematic_variation.py --all-momentum --data data.npz

# Run single direction only
python run_systematic_variation.py --systematic DIO_Theory --direction plus --data data.npz

# Collect results from previous runs
python run_systematic_variation.py --collect

# Show impacts as table
python run_systematic_variation.py --impact-summary
"""

import argparse
import json
import sys
import os
import numpy as np
import pickle as pkl
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

from sysunc_components import sysunc_components, get_implemented_systematics, get_systematics_by_component
from uncertainty_propagation import UncertaintyPropagator


class SystematicVariationRunner:
    """
    Phase 2: Run systematic variations and measure impacts on signal yield (N_CE).
    """
    
    def __init__(self, data_file: str, output_dir: str = 'uncertainties/outputs', 
                 fit_type: str = '2d', verbose: int = 1):
        """
        Parameters
        ----------
        data_file : str
            Path to NPZ file with baseline data (mom_mag, track_cat, time, etc.)
        output_dir : str
            Directory to save variation results
        fit_type : str
            '1d' or '2d' (default: '2d')
        verbose : int
            Verbosity level
        """
        self.data_file = Path(data_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fit_type = fit_type
        self.propagator = UncertaintyPropagator(output_dir=str(self.output_dir))
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
            if self.fit_type == '2d' and 'time' in data:
                self.baseline_data['time'] = data['time']
                self.logger.log(f"Loaded time data for 2D fit", 'info')
            
            self.logger.log(f"Loaded baseline data from {self.data_file}", 'info')
            self.logger.log(f"  Momentum points: {len(self.baseline_data['mom_mag'])}", 'info')
            if self.fit_type == '2D' and 'time' in self.baseline_data:
                self.logger.log(f"  Time points: {len(self.baseline_data['time'])}", 'info')
            return self.baseline_data
        except Exception as e:
            self.logger.log(f"Failed to load {self.data_file}: {e}", 'error')
            raise
    
    def fit_baseline(self, fit_range: Tuple[float, float] = (95, 115)) -> Dict[str, Any]:
        """
        Run baseline fit (Phase 1: with all constraints).
        
        Returns dict with:
          - signal_yield (N_CE value)
          - uncertainty (stat + syst)
          - result (full zfit result)
        """
        if self.baseline_data is None:
            raise RuntimeError("Call load_baseline_data() first")
        
        self.logger.log(f"Running baseline {self.fit_type} fit (Phase 1 with constraints)...", 'info')
        
        try:
            from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
            
            if self.fit_type == '2D':
                result, poi, loss, _, _ = Unbinned_2d_fit_mom_time(
                    self.baseline_data['mom_mag'],
                    self.baseline_data.get('time'),
                    4,
                    fit_range_mom=fit_range,
                    fit_range_time=(700, 1700),
                    constraints_dir='uncertainties/outputs',
                    verbose=0,
                    plot_results=False
                )
            else:  # 1d
                result, poi, loss, _, _, _ = Unbinned_fit_mom(
                    self.baseline_data['mom_mag'],
                    4,
                    fit_range[0],
                    fit_range[1],
                    constraints_dir='uncertainties/outputs',
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
            
            self.baseline_result = {
                'poi_value': float(poi.value()),
                'poi_uncertainty': poi_uncertainty,
                'result': result,
            }
            
            self.logger.log(f"Baseline N_CE: {self.baseline_result['poi_value']:.1f} ± {self.baseline_result['poi_uncertainty']:.2f}", 'success')
            return self.baseline_result
            
        except Exception as e:
            self.logger.log(f"Baseline fit failed: {e}", 'error')
            raise
    
    def apply_variation(self, systematic_name: str, direction: str = 'plus') -> Tuple[Dict, Dict]:
        """
        Apply a single systematic variation and return modified data.
        
        Returns:
          - varied_data: modified data dict
          - metadata: applied variation details
        """
        if self.baseline_data is None:
            raise RuntimeError("Call load_baseline_data() first")
        
        spec = sysunc_components.get(systematic_name)
        if spec is None:
            raise ValueError(f"Unknown systematic: {systematic_name}")
        
        varied_data = self.baseline_data.copy()
        metadata = {}
        
        # Apply variation based on type
        if spec['type'] == 'shift':
            # Momentum shift
            varied_data['mom_mag'], metadata = self.propagator.apply_momentum_shift(
                self.baseline_data['mom_mag'], systematic_name, direction
            )
        elif spec['type'] == 'shape':
            # Momentum smearing
            varied_data['mom_mag'], metadata = self.propagator.apply_momentum_smearing(
                self.baseline_data['mom_mag'], systematic_name, direction
            )
        elif spec['type'] == 'frac':
            # For fractional systematics, we'd need to vary yields
            # For now, store metadata indicating this should be fixed in fit
            metadata = {
                'systematic': systematic_name,
                'type': 'frac',
                'direction': direction,
                'note': 'Fraction type - will be handled via parameter fixing in Phase 2.1'
            }
        else:
            raise ValueError(f"Unsupported variation type: {spec['type']}")
        
        return varied_data, metadata
    
    def run_fit_variation(self, systematic_name: str, direction: str = 'plus',
                         fit_range: Tuple[float, float] = (95, 115)) -> Optional[Dict[str, Any]]:
        """
        Run a single systematic variation: apply variation, fit, collect metrics.
        
        Returns dict with impact metrics, or None if failed.
        """
        if self.baseline_result is None:
            raise RuntimeError("Call fit_baseline() first")
        
        self.logger.log(f"Varying {systematic_name} ({direction})...", 'info')
        
        try:
            from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
            
            # Apply variation
            varied_data, var_metadata = self.apply_variation(systematic_name, direction)
            
            # Save varied data
            var_npz = self.output_dir / f"{systematic_name}__sys-var_{direction}.npz"
            np.savez_compressed(
                var_npz,
                mom_mag=varied_data['mom_mag'],
                track_cat=varied_data.get('track_cat', np.array([])),
                time=varied_data.get('time', np.array([])),
                metadata=json.dumps(var_metadata)
            )
            
            # Run fit on varied data
            if self.fit_type == '2D':
                result, poi, loss, _, _ = Unbinned_2d_fit_mom_time(
                    varied_data['mom_mag'],
                    varied_data.get('time'),
                    4,
                    fit_range_mom=fit_range,
                    fit_range_time=(700, 1700),
                    constraints_dir='uncertainties/outputs',
                    verbose=0,
                    plot_results=False
                )
            else:  # 1d
                result, poi, loss, _, _, _ = Unbinned_fit_mom(
                    varied_data['mom_mag'],
                    4,
                    fit_range[0],
                    fit_range[1],
                    constraints_dir='uncertainties/outputs',
                    verbose=0,
                    plot_results=False
                )
            
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
                'type': sysunc_components[systematic_name]['type'],
                'direction': direction,
                'baseline_poi': self.baseline_result['poi_value'],
                'varied_poi': poi_value,
                'poi_shift': shift,
                'poi_shift_sigma': shift_sigma,
                'baseline_unc': self.baseline_result['poi_uncertainty'],
                'varied_unc': poi_uncertainty,
                'unc_change': poi_uncertainty - self.baseline_result['poi_uncertainty'],
                'varied_data_npz': str(var_npz),
            }
            
            self.logger.log(
                f"  N_CE: {self.baseline_result['poi_value']:.1f} → {poi_value:.1f} "
                f"(shift: {shift:+.2f}, σ: {shift_sigma:+.2f})",
                'info'
            )
            
            return results
            
        except Exception as e:
            self.logger.log(f"Failed to run {systematic_name} ({direction}): {e}", 'error')
            return None
    
    def run_multiple_variations(self, systematics: list, 
                               fit_range: Tuple[float, float] = (95, 115)):
        """Run variations for multiple systematics."""
        self.logger.log(f"Running {len(systematics)} systematics (both directions)...", 'info')
        
        for sys_name in systematics:
            if sys_name not in sysunc_components:
                self.logger.log(f"Skipping unknown systematic: {sys_name}", 'warning')
                continue
            
            for direction in ['plus', 'minus']:
                try:
                    result = self.run_fit_variation(sys_name, direction, fit_range)
                    if result:
                        self.results[f"{sys_name}_{direction}"] = result
                except Exception as e:
                    self.logger.log(f"Error: {e}", 'error')
    
    def save_results(self) -> Path:
        """Save all variation results to JSON."""
        results_file = self.output_dir / 'systematic_variation_results.json'
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
            "SYSTEMATIC IMPACT SUMMARY",
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


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Run systematic uncertainty variations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--data', type=str, required=False,
                       help='Path to data NPZ file (required for most operations)')
    parser.add_argument('--fittype', type=str, default='2D', choices=['mom1D', '2D'],
                       help='Fit type: mom1D (momentum only) or 2D (momentum+time, default: 2D)')
    parser.add_argument('--systematic', type=str,
                       help='Single systematic to vary (runs both ± directions)')
    parser.add_argument('--direction', type=str, default='plus', choices=['plus', 'minus'],
                       help='Variation direction (if --systematic specified)')
    parser.add_argument('--all-momentum', action='store_true',
                       help='Run all momentum-affecting systematics')
    parser.add_argument('--all-implemented', action='store_true',
                       help='Run all implemented systematics')
    parser.add_argument('--fit-range', type=float, nargs=2, default=[95, 115],
                       help='Fit range (low high)')
    parser.add_argument('--outdir', type=str, default='uncertainties/outputs',
                       help='Output directory')
    parser.add_argument('--collect', action='store_true',
                       help='Collect results from outputs directory')
    parser.add_argument('--impact-summary', action='store_true',
                       help='Show impact summary from results')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                       help='Verbosity level')
    
    args = parser.parse_args()
    
    # --- Handle result collection / summary (no data needed) ---
    if args.collect or args.impact_summary:
        results_file = Path(args.outdir) / 'systematic_variation_results.json'
        if not results_file.exists():
            print(f"✗ Results file not found: {results_file}")
            sys.exit(1)
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Create dummy runner just for summary
        runner = SystematicVariationRunner('dummy.npz', output_dir=args.outdir, verbose=0)
        runner.results = results
        
        # Extract baseline from first result (approximate)
        if results:
            first = list(results.values())[0]
            runner.baseline_result = {
                'poi_value': first['baseline_poi'],
                'poi_uncertainty': first['baseline_unc'],
            }
        
        print(runner.produce_impact_summary())
        return
    
    # --- Run variations (data required) ---
    if not args.data:
        print("✗ --data is required for running variations")
        print("  Use --collect or --impact-summary to view existing results")
        sys.exit(1)
    
    try:
        runner = SystematicVariationRunner(args.data, output_dir=args.outdir, 
                                           fit_type=args.fittype, verbose=args.verbose)
        runner.load_baseline_data()
        runner.fit_baseline(tuple(args.fit_range))
        
        # Determine which systematics to run
        if args.systematic:
            systematics = [args.systematic]
        elif args.all_momentum:
            systematics = list(get_systematics_by_component('mom').keys())
        elif args.all_implemented:
            systematics = list(get_implemented_systematics().keys())
        else:
            parser.print_help()
            return
        
        runner.run_multiple_variations(systematics, tuple(args.fit_range))
        runner.save_results()
        print(runner.produce_impact_summary())
        
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
