"""
Utility script to generate data cards from existing configs/snapshots.

Usage:
    python generate_datacard.py --snapshot fit_snapshot.npz --output card.yaml
    python generate_datacard.py --from-config config.dict --output card.yaml
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import logging

# Add parent directory to path for imports
sys_path_parent = Path(__file__).parent.parent
if str(sys_path_parent) not in sys.path:
    sys.path.insert(0, str(sys_path_parent))

from datacard import DataCard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def datacard_from_snapshot(snapshot_path: str, name: str = None, 
                          output_path: str = None) -> DataCard:
    """
    Generate data card from existing NPZ snapshot.
    
    Requires systematics to be stored in snapshot (from parquet_fit_builder --card input).
    This enforces the Combine-style workflow: User provides input card with systematics
    to parquet_fit_builder, which preserves them in the snapshot.
    
    Args:
        snapshot_path: Path to .npz snapshot file (from parquet_fit_builder with --card)
        name: Card name (defaults to stem of snapshot file)
        output_path: If provided, save card to this file
        
    Returns:
        DataCard object
        
    Raises:
        ValueError: If required yields, CE DSCB shape parameters, or systematics are missing
    """
    logger.info(f"Loading snapshot from {snapshot_path}")
    snapshot = np.load(snapshot_path, allow_pickle=True)
    
    if name is None:
        name = Path(snapshot_path).stem + "_card"
    
    card = DataCard(name)
    
    # Extract fit ranges
    if 'fit_ranges' in snapshot:
        fit_ranges = snapshot['fit_ranges']
        card.set_observables({
            'mom': (float(fit_ranges[0]), float(fit_ranges[1])),
            'time': (float(fit_ranges[2]), float(fit_ranges[3]))
        })
        logger.info(f"Observable ranges: mom={card.observables['mom']}, "
                   f"time={card.observables['time']}")
    
    # Extract process yields
    processes_found = []
    yields_dict = {}  # Track yields for POI range setting
    yield_keys = {
        'val_N_CE': 'CE',
        'val_N_Cosmic': 'Cosmic',
        'val_N_RPC': 'RPC',
        'val_N_DIO': 'DIO',
        'val_N_RMC': 'RMC',
    }
    
    for file_key, proc_name in yield_keys.items():
        if file_key not in snapshot.files:
            raise ValueError(f"Required yield key '{file_key}' ({proc_name}) not found in snapshot {snapshot_path}. "
                           f"Available keys: {list(snapshot.files)}")
        
        yield_val = float(snapshot[file_key])
        
        # Ensure yields are non-negative (physical requirement)
        if yield_val < 0:
            logger.warning(f"{proc_name} yield is negative ({yield_val:.4f}), using absolute value")
            yield_val = abs(yield_val)
        
        # Track for later use
        yields_dict[proc_name] = yield_val
        
        # Determine shape type and params
        shape_type = 'fixed' if proc_name == 'CE' else 'free'
        shape_params = {}
        
        if proc_name == 'DIO':
            required_keys = ['val_a5_DIO', 'val_a6_DIO', 'val_a7_DIO', 'val_a8_DIO']
            missing = [k for k in required_keys if k not in snapshot.files]
            if missing:
                raise ValueError(f"DIO shape parameters missing from snapshot: {missing}")
            # Decay rate is optional - may not be in snapshot
            decay_mu = float(snapshot['val_decay_rate_mu']) if 'val_decay_rate_mu' in snapshot.files else None
            shape_params = {
                'model': 'poly58',
                'a5': float(snapshot['val_a5_DIO']),
                'a6': float(snapshot['val_a6_DIO']),
                'a7': float(snapshot['val_a7_DIO']),
                'a8': float(snapshot['val_a8_DIO']),
            }
            if decay_mu is not None:
                shape_params['decay_mu'] = decay_mu
        elif proc_name == 'Cosmic':
            required_keys = ['val_c1_Cosmic', 'val_c2_Cosmic']
            missing = [k for k in required_keys if k not in snapshot.files]
            if missing:
                raise ValueError(f"Cosmic shape parameters missing from snapshot: {missing}")
            shape_params = {
                'model': 'chebyshev',
                'c1': float(snapshot['val_c1_Cosmic']),
                'c2': float(snapshot['val_c2_Cosmic']),
            }
        elif proc_name == 'RPC':
            required_keys = ['val_c1_RPC', 'val_c2_RPC']
            missing = [k for k in required_keys if k not in snapshot.files]
            if missing:
                raise ValueError(f"RPC shape parameters missing from snapshot: {missing}")
            # Decay rate is optional - may not be in snapshot
            decay_pi = float(snapshot['val_decay_rate_pi']) if 'val_decay_rate_pi' in snapshot.files else None
            shape_params = {
                'model': 'chebyshev',
                'c1': float(snapshot['val_c1_RPC']),
                'c2': float(snapshot['val_c2_RPC']),
            }
            if decay_pi is not None:
                shape_params['decay_pi'] = decay_pi
        elif proc_name == 'CE':
            # Signal shape parameters: DSCB (Double-sided CrystalBall)
            # These are REQUIRED - they're physics calibration constants used downstream
            required_keys = ['val_mu_CE', 'val_sigma_CE', 'val_alphaL_CE', 'val_nL_CE',
                           'val_alphaR_CE', 'val_nR_CE']
            missing = [k for k in required_keys if k not in snapshot.files]
            
            if missing:
                available_ce_keys = [k for k in snapshot.files if 'CE' in k or 'mu_' in k.lower()]
                raise ValueError(
                    f"CE DSCB shape parameters REQUIRED but missing from snapshot: {missing}\n"
                    f"These parameters must be in the snapshot to generate a usable datacard.\n"
                    f"Available keys with 'CE' or 'mu': {available_ce_keys}\n"
                    f"Add these CE calibration parameters to your fit snapshot and try again.")
            
            decay_mu = float(snapshot['val_decay_rate_mu']) if 'val_decay_rate_mu' in snapshot.files else None
            shape_params = {
                'model': 'dscb',  # Double-sided crystal ball
                'mu': float(snapshot['val_mu_CE']),
                'sigma': float(snapshot['val_sigma_CE']),
                'alphaL': float(snapshot['val_alphaL_CE']),
                'nL': float(snapshot['val_nL_CE']),
                'alphaR': float(snapshot['val_alphaR_CE']),
                'nR': float(snapshot['val_nR_CE']),
            }
            if decay_mu is not None:
                shape_params['decay_mu'] = decay_mu

        else:
            shape_params = {}
        
        card.add_process(proc_name, yield_val, 
                        shape_type=shape_type, 
                        shape_params=shape_params)
        processes_found.append(f"{proc_name}={yield_val:.4f}")
    
    logger.info(f"Extracted processes: {', '.join(processes_found)}")
    
    # Set POI range based on fitted CE yield
    # Get CE yield (should be in processes_found by now)
    ce_yield = yields_dict.get('CE')
    if ce_yield is None:
        raise ValueError("CE yield not found in snapshot")
    
    # POI range optional - only set if available in snapshot
    if 'poi_upper' in snapshot.files:
        # POI range available in snapshot
        poi_upper = float(snapshot['poi_upper'])
        card.set_poi('N_CE', (0.0, poi_upper))
        logger.info(f"Set POI N_CE range to (0.0, {poi_upper:.4f}) from snapshot")
    else:
        # No POI range in snapshot - skip it, downstream tools will handle
        logger.info("POI range not in snapshot. Must be configured via downstream analysis framework.")
    
    # Add systematics from snapshot (required - part of input card passed to parquet_fit_builder)
    if 'systematics' in snapshot.files:
        systematics_dict = dict(snapshot['systematics'].item() if snapshot['systematics'].size == 1 else snapshot['systematics'])
        for syst_name, syst_values in systematics_dict.items():
            syst_type = syst_values.get('type', 'lnN')
            syst_effects = syst_values.get('effects', {})
            card.add_systematic(syst_name, syst_type, syst_effects)
        logger.info(f"✓ Loaded {len(systematics_dict)} systematics from snapshot")
    else:
        raise ValueError(
            "ERROR: Systematics not found in snapshot!\n"
            "Workflow requires: User → input card with systematics → parquet_fit_builder --card → snapshot\n"
            "The snapshot must contain systematics from the input card for reproducibility.\n"
            "Run parquet_fit_builder with --card input_card.yaml to generate valid snapshot."
        )
    
    card.metadata = {
        'source': 'snapshot',
        'snapshot_path': str(snapshot_path),
    }
    
    # Save if requested
    if output_path:
        if output_path.endswith('.json'):
            card.to_json(output_path)
        else:
            card.to_yaml(output_path)
        logger.info(f"Saved data card to {output_path}")
    
    return card


def datacard_from_yields_dict(yields: dict, name: str = 'generated_card',
                             obs_ranges: dict = None,
                             poi_range: tuple = None,
                             output_path: str = None) -> DataCard:
    """
    Generate data card from a simple yields dictionary.
    
    Args:
        yields: {process_name: yield_value}
        name: Card name
        obs_ranges: {'mom': (min, max), 'time': (min, max)} - REQUIRED
        poi_range: (min, max) for N_CE parameter - optional
        output_path: Save location
        
    Returns:
        DataCard object
    """
    # Require explicit configuration, no hardcoded defaults
    if obs_ranges is None:
        raise ValueError("obs_ranges must be provided explicitly (e.g., {'mom': (97, 115), 'time': (0, 1700)})")
    
    card = DataCard(name)
    card.set_observables(obs_ranges)
    
    # Add processes
    for proc_name, yield_val in yields.items():
        shape_type = 'fixed' if proc_name == 'CE' else 'free'
        card.add_process(proc_name, yield_val, shape_type=shape_type)
    
    # Set POI if provided
    if 'CE' in yields and poi_range is not None:
        card.set_poi('N_CE', poi_range)
    
    # Save if requested
    if output_path:
        if output_path.endswith('.json'):
            card.to_json(output_path)
        else:
            card.to_yaml(output_path)
        logger.info(f"Saved data card to {output_path}")
    
    return card


def make_systematic_variations(card: DataCard, output_dir: str = '.'):
    """
    Create individual data cards with systematic variations.
    Useful for impact studies.
    
    Args:
        card: Base data card
        output_dir: Where to save variation cards
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    base_yields = card.get_expected_yields()
    
    # Save nominal
    nominal_path = output_dir / f"{card.name}_nominal.yaml"
    card.to_yaml(str(nominal_path))
    logger.info(f"Saved nominal: {nominal_path}")
    
    # Create variations for each systematic
    for syst_name in card.systematics:
        for variation_level in [0, 1, 2]:  # down, nominal, up
            var_name = {0: 'down', 1: 'nominal', 2: 'up'}[variation_level]
            
            # Apply systematic
            varied_yields = card.apply_systematic_variation(syst_name, variation_level)
            
            # Create new card with varied yields
            var_card = DataCard(f"{card.name}_{syst_name}_{var_name}")
            var_card.observables = card.observables.copy()
            var_card.poi = card.poi
            var_card.poi_range = card.poi_range
            
            for proc, yield_val in varied_yields.items():
                shape_info = card.get_process_shape_info(proc)
                var_card.add_process(proc, yield_val,
                                    shape_type=shape_info['type'],
                                    shape_params=shape_info['params'])
            
            var_path = output_dir / f"{var_card.name}.yaml"
            var_card.to_yaml(str(var_path))
            logger.info(f"  Saved {var_name}: {var_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate data cards from existing configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From snapshot (recommended - all required parameters in file)
  python generate_datacard.py --snapshot fit.npz --output card.yaml
  
  # From snapshot with explicit POI range
  python generate_datacard.py --snapshot fit.npz --poi-range "0,50" --output card.yaml
  
  # From custom yields (requires explicit observable ranges)
  python generate_datacard.py --yields "DIO:1430,Cosmic:330,RPC:9.4,CE:0.3" \\
    --mom-range "97,115" --time-range "0,1700" --output card.yaml
  
  # From custom yields with POI range
  python generate_datacard.py --yields "DIO:1430,CE:0.3" \\
    --mom-range "97,115" --time-range "0,1700" --poi-range "0,40" --output card.yaml
  
  # Create systematic variations from snapshot
  python generate_datacard.py --snapshot fit.npz --systematics --output-dir vars/
        """)
    
    parser.add_argument('--snapshot', default=None,
                       help='NPZ snapshot file to extract from')
    parser.add_argument('--yields', default=None,
                       help='Comma-separated yields: "DIO:1430,Cosmic:330"')
    parser.add_argument('--name', default=None,
                       help='Data card name')
    parser.add_argument('--output', '-o', default=None,
                       help='Output file (YAML or JSON)')
    parser.add_argument('--mom-range', type=str, default=None,
                       help='Momentum range as "min,max"')
    parser.add_argument('--time-range', type=str, default=None,
                       help='Time range as "min,max"')
    parser.add_argument('--poi-range', type=str, default=None,
                       help='POI range as "min,max" (optional, can be configured later)')
    parser.add_argument('--systematics', action='store_true',
                       help='Generate systematic variation cards')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for multiple cards')
    
    args = parser.parse_args()
    
    if args.snapshot:
        card = datacard_from_snapshot(args.snapshot, name=args.name, 
                                     output_path=args.output)
        
        if args.systematics:
            make_systematic_variations(card, output_dir=args.output_dir)
    
    elif args.yields:
        # Parse yields string: "DIO:1430,Cosmic:330,CE:0.3"
        yields_dict = {}
        for item in args.yields.split(','):
            proc, value = item.split(':')
            yields_dict[proc.strip()] = float(value.strip())
        
        # Parse observable ranges (required)
        if not args.mom_range or not args.time_range:
            parser.error("When using --yields, must also provide --mom-range and --time-range")
        
        try:
            mom_min, mom_max = map(float, args.mom_range.split(','))
            time_min, time_max = map(float, args.time_range.split(','))
        except ValueError:
            parser.error("Ranges must be in format 'min,max' with numeric values")
        
        # Parse optional POI range
        poi_range = None
        if args.poi_range:
            try:
                poi_min, poi_max = map(float, args.poi_range.split(','))
                poi_range = (poi_min, poi_max)
            except ValueError:
                parser.error("POI range must be in format 'min,max' with numeric values")
        
        obs_ranges = {
            'mom': (mom_min, mom_max),
            'time': (time_min, time_max)
        }
        
        card = datacard_from_yields_dict(yields_dict, name=args.name or 'custom_card',
                                        obs_ranges=obs_ranges, poi_range=poi_range,
                                        output_path=args.output)
    
    else:
        parser.print_help()
        return
    
    if card:
        logger.info(f"\n✓ Created data card: {card.name}")
        logger.info(f"  Processes: {list(card.processes.keys())}")
        logger.info(f"  POI: {card.poi} {card.poi_range}")


if __name__ == "__main__":
    main()
