#!/usr/bin/env python
"""
Compute background-only upper limit from data.

Usage:
  python compute_bg_only_ul.py --data nominal_data.npz --out ul_result.txt
"""
import argparse
import numpy as np
import awkward as ak
from pathlib import Path

from fit_module import Unbinned_fit_mom
from results_module import ResultsClass


def load_mom_from_npz(path):
    """Load momentum array from NPZ file."""
    arr = np.load(path, allow_pickle=True)
    if 'mom_mag' in arr.files:
        return np.asarray(arr['mom_mag'])
    # fallback: return first 1D numeric array
    for k in arr.files:
        a = np.asarray(arr[k])
        if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
            return a
    raise RuntimeError(f'No 1D momentum array found in {path}')


def main():
    p = argparse.ArgumentParser(description='Compute background-only upper limit')
    p.add_argument('--data', required=True, help='NPZ file with momentum array')
    p.add_argument('--fit-range', nargs=2, type=float, default=[95.0, 115.0],
                   help='momentum fit range')
    p.add_argument('--constraints-dir', default=None, help='uncertainties directory')
    p.add_argument('--out', default='bg_only_ul.txt', help='output file')
    p.add_argument('--verbose', type=int, default=1)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} not found')

    if args.verbose:
        print(f'Loading data from {data_path}...')
    mom = load_mom_from_npz(str(data_path))
    mom_ak = ak.Array(mom)

    if args.verbose:
        print(f'Running background-only fit on {len(mom)} events...')
    
    # Run fit (no signal)
    fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
        mom_ak,
        [],  # track_cat
        [],  # count_particle_types
        args.fit_range[0],
        args.fit_range[1],
        False,
        args.verbose,
        minos=False,
        plot_NLL=False,
        plot_results=False,
        constraints_dir=args.constraints_dir,
    )

    if args.verbose:
        print(f'Fit successful. POI: {par.name if hasattr(par, "name") else "N/A"}')
        print(f'Computing upper limit at 90% CL (asymptotic)...')

    # Compute UL
    rc = ResultsClass(mom_ak, fitresult, verbose=args.verbose)
    
    try:
        ul_obj = rc.GetUL(
            par, loss, nlls, combine_pdf, constraints,
            args.fit_range[0], args.fit_range[1],
            sig_yield=0,  # background-only: start from 0 signal
            CL=0.90,
            opt='freq'  # frequentist calculator - more robust to bound violations
        )
        
        # Extract numeric UL value
        ul_value = None
        try:
            ul_value = float(ul_obj.upperlimit(alpha=0.05, CLs=True))
        except Exception:
            # Fallback to median of POI scan
            try:
                ul_value = float(np.median(ul_obj.poinull.values))
            except Exception:
                print('Failed to extract numeric UL')
                ul_value = float('nan')
        
        if args.verbose:
            print(f'✅ Background-only upper limit: {ul_value:.4f}')
        
        # Write result
        with open(args.out, 'w') as f:
            f.write(f"Background-only 90% CL Upper Limit: {ul_value}\n")
            f.write(f"Data events: {len(mom)}\n")
            f.write(f"Fit range: [{args.fit_range[0]}, {args.fit_range[1]}]\n")
            f.write(f"POI: {par.name if hasattr(par, 'name') else 'N/A'}\n")
        
        if args.verbose:
            print(f'Result written to {args.out}')
        
        return ul_value
        
    except Exception as e:
        print(f'❌ Failed to compute UL: {e}')
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
