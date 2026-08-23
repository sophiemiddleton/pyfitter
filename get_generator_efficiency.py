#!/usr/bin/env python
"""
Script to calculate generator efficiency for Mu2e datasets
Efficiency = Triggered Events / Generated Events
"""

import subprocess
import argparse
import sys

def get_dataset_stats(dataset_name):
    """
    Get generated and triggered event counts from SAM dataset
    
    Args:
        dataset_name: Full SAM dataset name
    
    Returns:
        tuple: (generated_count, triggered_count) or (None, None) on error
    """
    
    try:
        # Get generated events
        cmd_gen = f'samDatasetsSummary.sh "{dataset_name}" | awk \'/Generated/ {{print $2}}\''
        result_gen = subprocess.run(cmd_gen, shell=True, capture_output=True, text=True, timeout=30)
        generated = int(result_gen.stdout.strip())
        
        # Get triggered events
        cmd_trig = f'samDatasetsSummary.sh "{dataset_name}" | awk \'/Triggered/ {{print $2}}\''
        result_trig = subprocess.run(cmd_trig, shell=True, capture_output=True, text=True, timeout=30)
        triggered = int(result_trig.stdout.strip())
        
        return generated, triggered
    
    except subprocess.TimeoutExpired:
        print("Error: Command timed out", file=sys.stderr)
        return None, None
    except ValueError:
        print(f"Error: Could not parse event counts", file=sys.stderr)
        return None, None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None, None

def calculate_efficiency(generated, triggered):
    """Calculate generator efficiency"""
    if generated == 0:
        return 0.0
    return (triggered / generated) * 100.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate generator efficiency from SAM dataset summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python get_generator_efficiency.py "mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.art"
  python get_generator_efficiency.py "mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.art" --json
        """
    )
    parser.add_argument("dataset", help="Full SAM dataset name")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    print(f"Querying dataset: {args.dataset}")
    generated, triggered = get_dataset_stats(args.dataset)
    
    if generated is None or triggered is None:
        print("Failed to retrieve dataset statistics", file=sys.stderr)
        sys.exit(1)
    
    efficiency = calculate_efficiency(generated, triggered)
    
    if args.json:
        import json
        output = {
            "dataset": args.dataset,
            "generated": generated,
            "triggered": triggered,
            "efficiency_percent": efficiency
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Dataset: {args.dataset}")
        print(f"{'='*60}")
        print(f"Generated Events:  {generated:>12,}")
        print(f"Triggered Events:  {triggered:>12,}")
        print(f"-{'-'*58}")
        print(f"Generator Efficiency: {efficiency:>9.5g}%")
        print(f"{'='*60}\n")
    
    sys.exit(0)
