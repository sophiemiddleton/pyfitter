#!/usr/bin/env python3
"""
Example: Running fits on scaled MC component samples.

This script demonstrates how to use the ScaledFitBuilder to:
1. Load separate MC component files
2. Apply cuts to each component
3. Scale components by expected yields
4. Execute fits using the existing infrastructure

The approach enables fitting to "highest stats" sample sets where each component
is loaded independently and scaled appropriately, rather than using a single
combined dataset.
"""

import argparse
from pathlib import Path
from scaled_fit_builder import ScaledFitBuilder


def example_basic_1d_fit():
    """Basic example: 1D momentum fit on scaled components."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic 1D Momentum Fit")
    print("="*70)
    
    # Initialize builder
    builder = ScaledFitBuilder(verbosity=2, jobs=4)
    
    # Define component file lists (you'll need actual file paths)
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'rpc_ext': 'file_lists/ExtRPC_MDC2025an_nomix.txt',
        'rpc_int': 'file_lists/IntRPC_MDC2025an_nomix.txt',
        'rmc_ext': 'file_lists/rmc_ext.txt',  # If available
        'rmc_int': 'file_lists/rmc_int.txt',  # If available
        'ce': 'file_lists/signal_ce.txt',     # If available
    }
    
    # Load and scale components using physics default yields
    builder.load_and_scale_components(
        component_files,
        sign='minus',
        location='disk',
        normalize_to_max=False  # Use default physics yields
    )
    
    # Optional: Load real data for comparison
    # builder.load_real_data('file_lists/data.txt', sign='minus', location='local')
    
    # Run 1D fit
    result = builder.fit_mom_1d(
        variable='recomom_ttfront',
        fit_range=(95, 110),
        plot=True,
        minos=False,
        constraints_dir='uncertainties/outputs'
    )
    
    print("\n✓ 1D fit completed successfully")
    return result


def example_custom_yields_1d():
    """Example: 1D fit with custom component yields."""
    print("\n" + "="*70)
    print("EXAMPLE 2: 1D Fit with Custom Yields")
    print("="*70)
    
    builder = ScaledFitBuilder(verbosity=1, jobs=4)
    
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'ce': 'file_lists/signal_ce.txt',
    }
    
    builder.load_and_scale_components(component_files, sign='minus', location='disk')
    
    # Override default yields with custom values
    custom_yields = {
        'dio': 6000,      # Override default 5870
        'cosmic': 450,    # Override default 500.5
        'ce': 70,         # Override default 65
    }
    builder.set_component_yields(custom_yields)
    
    # The scaling will be re-applied with new yields
    # Note: You'd need to reload components to apply new yields
    # This is better done before load_and_scale_components call:
    
    builder = ScaledFitBuilder(verbosity=1, jobs=4)
    builder.set_component_yields(custom_yields)
    builder.load_and_scale_components(component_files, sign='minus', location='disk')
    
    result = builder.fit_mom_1d(
        variable='recomom_ttfront',
        fit_range=(95, 110),
        plot=True,
    )
    
    print("✓ 1D fit with custom yields completed")
    return result


def example_2d_fit():
    """Example: 2D momentum + time fit on scaled components."""
    print("\n" + "="*70)
    print("EXAMPLE 3: 2D Momentum+Time Fit")
    print("="*70)
    
    builder = ScaledFitBuilder(verbosity=2, jobs=4)
    
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'rpc_ext': 'file_lists/ExtRPC_MDC2025an_nomix.txt',
        'rpc_int': 'file_lists/IntRPC_MDC2025an_nomix.txt',
        'ce': 'file_lists/signal_ce.txt',
    }
    
    # Load and scale
    builder.load_and_scale_components(component_files, sign='minus', location='disk')
    
    # Run 2D fit
    result = builder.fit_mom_time_2d(
        mom_variable='recomom_ttfront',
        time_variable='trkfit.trksegpars_lh.t0',
        fit_range_mom=(95, 110),
        fit_range_time=(475, 1650),
        plot=True,
        constraints_dir='uncertainties/outputs'
    )
    
    print("✓ 2D fit completed successfully")
    return result


def example_normalize_to_max():
    """Example: Normalize all components to match max sample size."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Scale to Maximum Component Size")
    print("="*70)
    
    # This approach scales all components to match the largest one,
    # useful for sensitivity studies or testing
    
    builder = ScaledFitBuilder(verbosity=1, jobs=4)
    
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'ce': 'file_lists/signal_ce.txt',
    }
    
    # Scale all to the maximum
    builder.load_and_scale_components(
        component_files,
        sign='minus',
        location='disk',
        normalize_to_max=True  # Key difference!
    )
    
    result = builder.fit_mom_1d(
        variable='recomom_ttfront',
        fit_range=(95, 110),
        plot=True,
    )
    
    print("✓ Fit with normalized scaling completed")
    return result


def example_mc_truth_variable():
    """Example: Fit MC truth momentum instead of reconstructed."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Fit MC True Momentum")
    print("="*70)
    
    builder = ScaledFitBuilder(verbosity=1, jobs=4)
    
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'ce': 'file_lists/signal_ce.txt',
    }
    
    builder.load_and_scale_components(component_files, sign='minus', location='disk')
    
    # Use MC truth momentum variable instead
    result = builder.fit_mom_1d(
        variable='recomom_mc_ttfront',  # MC true momentum
        fit_range=(95, 110),
        plot=True,
    )
    
    print("✓ MC truth momentum fit completed")
    return result


def example_different_variable():
    """Example: Fit a different reconstructed variable."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Fit Different Variable (Not Momentum)")
    print("="*70)
    
    builder = ScaledFitBuilder(verbosity=1, jobs=4)
    
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'ce': 'file_lists/signal_ce.txt',
    }
    
    builder.load_and_scale_components(component_files, sign='minus', location='disk')
    
    # You can fit any variable accessible in the data structure
    # Examples:
    # - 'trkfit.trksegpars_lh.p': Total momentum (LH hypothesis)
    # - 'trk.pt': Transverse momentum
    # - 'trkfit.trksegpars_lh.t0': Time-of-arrival
    # - 'trk.nactive': Number of active tracker planes
    
    result = builder.fit_mom_1d(
        variable='trkfit.trksegpars_lh.p',  # Different variable
        fit_range=(95, 110),
        plot=True,
    )
    
    print("✓ Alternative variable fit completed")
    return result


def example_with_data_overlay():
    """Example: 1D fit with real data overlaid as points."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Fit with Real Data Overlay")
    print("="*70)
    
    # This example shows how to overlay real MDS data as unscaled points
    # on top of the scaled MC components and fit result
    
    builder = ScaledFitBuilder(verbosity=2, jobs=4)
    
    # Define MC component files
    component_files = {
        'dio': 'file_lists/MDS3c_1e-13.txt',
        'cosmic': 'file_lists/CM_Cosmic.txt',
        'rpc_ext': 'file_lists/ExtRPC_MDC2025an_nomix.txt',
        'rpc_int': 'file_lists/IntRPC_MDC2025an_nomix.txt',
        'ce': 'file_lists/signal_ce.txt',
    }
    
    # Load and scale MC components using physics default yields
    builder.load_and_scale_components(
        component_files,
        sign='minus',
        location='disk',
        normalize_to_max=False
    )
    
    # Load real data - will be overlaid as black points with error bars (unscaled)
    data_loaded = builder.load_real_data(
        'file_lists/MDS3c.txt',  # Path to real data file list
        sign='minus',
        location='disk'
    )
    
    if not data_loaded:
        print("Warning: Could not load real data, proceeding with MC only")
    
    # Run 1D fit - real data will automatically be overlaid on the plot
    result = builder.fit_mom_1d(
        variable='recomom_ttfront',
        fit_range=(95, 110),
        plot=True,
        minos=False,
        constraints_dir='uncertainties/outputs'
    )
    
    print("\n✓ Fit with data overlay completed")
    print("  - Stacked bars: Scaled MC components")
    print("  - Black points: Real MDS data (unscaled)")
    print("  - Black solid line: Total fit")
    print("  - Dashed lines: Individual component fits")
    return result


def main():
    """Run available examples."""
    parser = argparse.ArgumentParser(
        description="Run examples of scaled sample fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples of available demo functions:
  python fit_scaled_examples.py --example basic
  python fit_scaled_examples.py --example custom_yields
  python fit_scaled_examples.py --example 2d
  python fit_scaled_examples.py --example normalize_to_max
  python fit_scaled_examples.py --example mc_truth
  python fit_scaled_examples.py --example different_var
  python fit_scaled_examples.py --example data_overlay
        """
    )
    
    parser.add_argument('--example', type=str, 
                       choices=['basic', 'custom_yields', '2d', 'normalize_to_max', 
                               'mc_truth', 'different_var', 'data_overlay', 'all'],
                       default='basic',
                       help='Which example to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would run without executing')
    
    args = parser.parse_args()
    
    examples = {
        'basic': example_basic_1d_fit,
        'custom_yields': example_custom_yields_1d,
        '2d': example_2d_fit,
        'normalize_to_max': example_normalize_to_max,
        'mc_truth': example_mc_truth_variable,
        'different_var': example_different_variable,
        'data_overlay': example_with_data_overlay,
    }
    
    if args.dry_run:
        print("DRY RUN MODE - No actual processing\n")
    
    if args.example == 'all':
        for name, example_func in examples.items():
            try:
                print(f"\n{'='*70}")
                print(f"Running: {name}")
                print('='*70)
                if not args.dry_run:
                    example_func()
                else:
                    print(f"  [SKIPPED in dry-run mode]")
            except Exception as e:
                print(f"✗ Error in example '{name}': {e}")
                import traceback
                traceback.print_exc()
    else:
        example_func = examples[args.example]
        try:
            if not args.dry_run:
                example_func()
            else:
                print(f"Would run: {example_func.__doc__}")
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
