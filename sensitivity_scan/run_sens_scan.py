"""
Usage example:
  python run_sens_scan_v2.py \\
    --data nominal_data.npz \\
    --fit-range 95 115 \\
    --mu-min 0 --mu-max 30 --n-mu 7 \\
    --ntoys 100 --out scan_v2.csv --plot scan_v2.png
"""
import argparse
import multiprocessing as mp
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path
import awkward as ak

from fit_module import Unbinned_fit_mom


def load_1d_from_npz(path):
    arr = np.load(path, allow_pickle=True)
    if 'mom_mag' in arr.files:
        return np.asarray(arr['mom_mag'])
    for k in arr.files:
        a = np.asarray(arr[k])
        if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
            return a
    raise RuntimeError(f'No 1D numeric array found in {path}')


def main():
    p = argparse.ArgumentParser(description='Improved toy-based sensitivity scan (v2)')
    p.add_argument('--data', required=True, help='NPZ file with nominal (background-only) momentum array')
    p.add_argument('--fit-range', nargs=2, type=float, default=[95.0, 115.0], metavar=('LOW', 'HIGH'))
    p.add_argument('--constraints-dir', default=None, help='Path to constraints JSON directory')
    p.add_argument('--mu-min', type=float, default=0.0, help='Minimum injected N_CE to scan')
    p.add_argument('--mu-max', type=float, default=30.0, help='Maximum injected N_CE to scan')
    p.add_argument('--n-mu', type=int, default=7, help='Number of mu grid points')
    p.add_argument('--ntoys', type=int, default=100, help='Toys per mu point')
    p.add_argument('--out', default='sensitivity_scan_v2.csv', help='Output CSV path')
    p.add_argument('--plot', default='sensitivity_scan_v2.png', help='Output plot path')
    p.add_argument('--results-out', default=None, help='Save full per-toy results as JSON')
    p.add_argument('--plot-toys', type=int, default=0, help='Save plots for first N toys per mu (0=off)')
    p.add_argument('--plot-dir', default='toy_plots_v2', help='Directory for per-toy plots')
    p.add_argument('--n-workers', type=int, default=4, help='Number of parallel workers for toy scan')
    p.add_argument('--verbose', type=int, default=1)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} not found')

    # ------------------------------------------------------------------
    # Load nominal data and run initial background-only fit
    # ------------------------------------------------------------------
    mom = load_1d_from_npz(str(data_path))
    mom_ak = ak.Array(mom)

    if args.verbose:
        print(f'Loaded {len(mom)} events from {data_path}')
        print('Running initial background-only fit on nominal data...')

    # Fix N_CE=0 for the initial fit so DIO+Cosmic correctly absorb all data
    # events. If CE floats on background-only data the DSCB finds a spurious
    # minimum and inflates/deflates the background yields.
    from physics_components import mom_components
    import copy
    # Mutate the dict IN PLACE so fit_module's cached reference (imported via
    # 'from physics_components import mom_components') also sees the change.
    # Remove CE entirely: on background-only data the DSCB floats to a spurious
    # minimum and inflates/deflates DIO+Cosmic yields.
    _ce_backup = mom_components.pop('CE', None)
    try:
        fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
            mom_ak, [], [],
            args.fit_range[0], args.fit_range[1],
            False, args.verbose,
            minos=False, plot_NLL=False, plot_results=False,
            constraints_dir=args.constraints_dir,
        )
    finally:
        if _ce_backup is not None:
            mom_components['CE'] = _ce_backup

    if args.verbose:
        print(f'Initial fit valid: {fitresult.valid}')
        print('Fitted parameters:')
        for name, info in fitresult.params.items():
            print(f'  {name} = {info["value"]:.4f}')

    # ------------------------------------------------------------------
    # Import mom_components (needed for PDF grid setup inside runners_v2)
    # ------------------------------------------------------------------
    mom_components_dict = mom_components

    # ------------------------------------------------------------------
    # Build mu grid
    # ------------------------------------------------------------------
    mu_grid = np.linspace(args.mu_min, args.mu_max, args.n_mu)
    if args.verbose:
        print(f'Scanning mu grid: {mu_grid}')

    # ------------------------------------------------------------------
    # Run parallel toy scan
    # ------------------------------------------------------------------
    from sensitivity_runners_v2 import parallel_toy_scan_v2

    results = parallel_toy_scan_v2(
        mu_grid=mu_grid,
        ntoys=args.ntoys,
        fit_result=fitresult,
        mom_components_dict=mom_components_dict,
        fit_range=tuple(args.fit_range),
        constraints_dir=args.constraints_dir,
        verbose=args.verbose,
        plot_toys=args.plot_toys,
        plot_dir=args.plot_dir,
        n_workers=args.n_workers,
    )

    # ------------------------------------------------------------------
    # Write CSV summary
    # ------------------------------------------------------------------
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mu', 'n_success', 'n_failed', 'median_ul', 'p16', 'p84'])
        for mu in sorted(results.keys()):
            r = results[mu]
            writer.writerow([mu, r['n_success'], r['n_failed'], r['median'], r['p16'], r['p84']])

    if args.verbose:
        print(f'Wrote CSV summary to {args.out}')

    # ------------------------------------------------------------------
    # Optionally save full per-toy results as JSON
    # ------------------------------------------------------------------
    if args.results_out:
        def _safe_float(x):
            try:
                v = float(x)
                return None if np.isnan(v) else v
            except Exception:
                return None

        serial = {}
        for mu in sorted(results.keys()):
            r = results[mu]
            serial[str(mu)] = {
                'n_success': r['n_success'],
                'n_failed': r['n_failed'],
                'median': _safe_float(r['median']),
                'p16': _safe_float(r['p16']),
                'p84': _safe_float(r['p84']),
                'values': [_safe_float(v) for v in r['values']],
            }
        with open(args.results_out, 'w') as jf:
            json.dump(serial, jf, indent=2)
        if args.verbose:
            print(f'Wrote full results to {args.results_out}')

    # ------------------------------------------------------------------
    # Plot median UL vs injected mu
    # ------------------------------------------------------------------
    mus = sorted(results.keys())
    med = [results[m]['median'] for m in mus]
    p16 = [results[m]['p16'] for m in mus]
    p84 = [results[m]['p84'] for m in mus]
    n_ok = [results[m]['n_success'] for m in mus]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mus, med, marker='o', label='Median UL$_{90}$')
    ax.fill_between(mus, p16, p84, alpha=0.35, color='steelblue', label='68% band')
    ax.set_xlabel('Injected signal $N_{CE}$')
    ax.set_ylabel('Upper limit on $N_{CE}$ (90% CL)')
    ax.set_title('Sensitivity scan — frequentist UL (v2)')
    # Annotate number of successful toys per point
    for m, y, n in zip(mus, med, n_ok):
        if np.isfinite(y):
            ax.annotate(f'n={n}', (m, y), textcoords='offset points',
                        xytext=(0, 6), ha='center', fontsize=7, color='grey')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot)
    if args.verbose:
        print(f'Wrote plot to {args.plot}')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
