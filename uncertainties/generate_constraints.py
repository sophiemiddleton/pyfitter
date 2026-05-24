#!/usr/bin/env python3
"""
Phase 1: Generate Constraints from Systematic Inventory

Simple, one-time tool to create constraints.json for use in fits.

Usage
-----
# Generate constraints from MDS3c data (default 2D fit, 95-115 MeV/c, 700-1700 ns)
python generate_constraints.py

# Use different data file
python generate_constraints.py --data my_data.npz

# Use 1D momentum fit instead of 2D
python generate_constraints.py --fittype mom1D

# 2D fit with custom momentum range (95-110 MeV/c)
python generate_constraints.py --fittype 2D --fit-range-mom 95 110 --data MDS3c_mom_mag.npz

# Custom momentum range with 1D fit
python generate_constraints.py --fittype mom1D --fit-range-mom 90 120

# Specify both momentum and time ranges (for 2D fit)
python generate_constraints.py --fittype 2D --fit-range-mom 95 115 --fit-range-time 600 1800

# Use explicit yields instead of fitting (no fit required)
python generate_constraints.py --nominal-yields '{"N_CE": 0.1, "N_DIO": 4500, "N_RPC": 1, "N_Cosmic": 100}'

# Validate specs
python generate_constraints.py --validate

# Show inventory
python generate_constraints.py --inventory
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from uncertainty_propagation import UncertaintyPropagator
from sysunc_components import get_implemented_systematics, get_constraints_only


def main():
    parser = argparse.ArgumentParser(
        description='Generate constraints.json from systematic inventory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--output', type=str, default='uncertainties/outputs/constraints.json',
                       help='Output file path (default: uncertainties/outputs/constraints.json)')
    parser.add_argument('--data', type=str, default='MDS3c_mom_mag.npz',
                       help='Data file for baseline fit (default: MDS3c_mom_mag.npz)')
    parser.add_argument('--fittype', type=str, default='2D', choices=['mom1D', '2D'],
                       help='Fit type: mom1D or 2D (default: 2D)')
    parser.add_argument('--fit-range-mom', type=float, nargs=2, default=[95, 115],
                       help='Momentum fit range (low high), default: 95 115')
    parser.add_argument('--fit-range-time', type=float, nargs=2, default=[700, 1700],
                       help='Time fit range (low high), default: 700 1700')
    parser.add_argument('--nominal-yields', type=str,
                       help='JSON string with nominal yields (overrides fit), e.g. \'{"N_CE": 18.5, "N_DIO": 6400}\'')
    parser.add_argument('--validate', action='store_true',
                       help='Validate all systematic specs and exit')
    parser.add_argument('--inventory', action='store_true',
                       help='Print systematic inventory and exit')
    parser.add_argument('--systematics', type=str, nargs='+',
                       help='Only generate constraints for these systematics (space-separated)')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                       help='Verbosity level')
    
    args = parser.parse_args()
    
    propagator = UncertaintyPropagator(output_dir=str(Path(args.output).parent), logger=None)
    
    # --- Handle special modes ---
    if args.validate:
        n_valid, errors = propagator.validate_all_systematics()
        if errors:
            print(f"\n❌ {len(errors)} validation errors found:")
            for err in errors:
                print(f"   {err}")
            sys.exit(1)
        else:
            print(f"\n✓ All {n_valid} systematics validated successfully")
            sys.exit(0)
    
    if args.inventory:
        print("\n" + propagator.summarize_systematics())
        sys.exit(0)
    
    # --- Generate constraints ---
    try:
        nominal_yields = None
        
        # Check if user provided yields (override)
        if args.nominal_yields:
            try:
                nominal_yields = json.loads(args.nominal_yields)
                print(f"\nUsing user-provided nominal yields:")
                for name, value in sorted(nominal_yields.items()):
                    print(f"  {name} = {value:.1f}")
            except Exception as e:
                print(f"  Error parsing --nominal-yields: {e}")
                sys.exit(1)
        else:
            # Run fit to get baseline yields
            import numpy as np
            from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
            
            data_file = args.data
            if not os.path.exists(data_file):
                print(f"✗ Data file not found: {data_file}")
                sys.exit(1)
            
            fit_type = args.fittype
            fit_range_mom = tuple(args.fit_range_mom)
            fit_range_time = tuple(args.fit_range_time)
            
            print(f"\nRunning baseline {fit_type} fit to extract nominal yields...")
            print(f"  Data: {data_file}")
            print(f"  Momentum range: {fit_range_mom}")
            if fit_type == '2D':
                print(f"  Time range: {fit_range_time}")
            
            data = np.load(data_file, allow_pickle=True)
            
            try:
                if fit_type == '2D':
                    result, poi, loss, _, _ = Unbinned_2d_fit_mom_time(
                        data['mom_mag'], data['time'], 4,
                        fit_range_mom=fit_range_mom,
                        fit_range_time=fit_range_time,
                        constraints_dir=None,
                        verbose=0,
                        plot_results=False
                    )
                else:  # mom1D
                    result, poi, loss, _, _, _ = Unbinned_fit_mom(
                        data['mom_mag'], 4, 
                        fit_range_mom[0], fit_range_mom[1],
                        constraints_dir=None,
                        verbose=0,
                        plot_results=False
                    )
                
                # Extract nominal yields from fit
                nominal_yields = {}
                for param in loss.get_params():
                    if param.name.startswith('N_'):
                        nominal_yields[param.name] = float(param.value())
                
                print(f"  Fit valid: {result.valid}")
                print(f"  Baseline yields:")
                for name, value in sorted(nominal_yields.items()):
                    print(f"    {name} = {value:.1f}")
                
            except Exception as e:
                print(f"  ✗ Fit failed: {e}")
                yields_example = '{"N_CE": 18.5, "N_DIO": 6400, "N_RPC": 1, "N_Cosmic": 400}'
                print(f"  Provide explicit yields with: --nominal-yields '{yields_example}'")
                sys.exit(1)
        
        outfile = propagator.save_constraints_json(
            systematics=args.systematics,
            nominal_yields=nominal_yields,
            outfile=args.output
        )
        print(f"\n✓ Constraints generated: {outfile}")
        
        # Show summary
        from sysunc_components import sysunc_components
        n_constraints = len([s for s in (args.systematics or get_constraints_only().keys()) 
                            if s in sysunc_components])
        print(f"  Total constraints: {n_constraints}")
        print(f"\n  Use in your fit:")
        print(f"    Unbinned_fit_mom(..., constraints_dir='{Path(args.output).parent}')")
        print(f"    # or for 2D fit:")
        print(f"    Unbinned_2d_fit_mom_time(..., constraints_dir='{Path(args.output).parent}')")
        
    except Exception as e:
        print(f"\n❌ Failed to generate constraints: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
