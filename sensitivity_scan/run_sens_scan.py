"""
Usage example:
  python run_sens_scan.py \
    --data nominal_data_2d.npz \
    --fit-range-mom 97 110 \
    --fit-range-time 475 1650 \
    --mu-min 0 --mu-max 30 --n-mu 7 \
    --ntoys 50 --out scan_2d.csv --plot scan_2d.png
"""
import argparse
import multiprocessing as mp
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyutils.pylogger import Logger
from config import GLOBAL_VERBOSITY
import numpy as np
import matplotlib.pyplot as plt
import csv
import json
from pathlib import Path
import awkward as ak

from fit_module import Unbinned_2d_fit_mom_time

logger = Logger(print_prefix='[run_sens_scan_2d] ', verbosity=GLOBAL_VERBOSITY)


def load_2d_from_npz(path):
    """Loads momentum, time, and category definitions from the npz cache."""
    arr = np.load(path, allow_pickle=True)
    return (
        np.asarray(arr['mom']),
        np.asarray(arr['time']),
        np.asarray(arr['categories']) if 'categories' in arr.files else []
    )


def main():
    p = argparse.ArgumentParser(description='2D Toy-based sensitivity scan (Momentum + Time)')
    p.add_argument('--data', required=True, help='NPZ file with nominal mom, time and categories arrays')
    p.add_argument('--fit-range-mom', nargs=2, type=float, default=[97.0, 110.0])
    p.add_argument('--fit-range-time', nargs=2, type=float, default=[475.0, 1650.0])
    p.add_argument('--constraints-dir', default='uncertainties/outputs')
    p.add_argument('--mu-min', type=float, default=0.0)
    p.add_argument('--mu-max', type=float, default=30.0)
    p.add_argument('--n-mu', type=int, default=7)
    p.add_argument('--ntoys', type=int, default=1)
    p.add_argument('--out', default='sensitivity_scan_2d.csv')
    p.add_argument('--plot', default='sensitivity_scan_2d.png')
    p.add_argument('--results-out', default=None)
    p.add_argument('--plot-toys', type=int, default=5)
    p.add_argument('--plot-dir', default='toy_plots_2d')
    p.add_argument('--n-workers', type=int, default=4)
    p.add_argument('--verbose', type=int, default=1)
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} not found')

    # Load 2D Arrays
    mom, times, categories = load_2d_from_npz(str(data_path))
    mom_ak, times_ak = ak.Array(mom), ak.Array(times)

    if args.verbose:
        logger.log(f'Loaded {len(mom)} 2D events from {data_path}', 'info')
        logger.log('Running baseline background-only 2D fit...', 'info')

    # Remove CE for baseline background initialization phase
    from model.physics_components import mom_components
    _ce_backup = mom_components.pop('CE', None)
    
    try:
        fitresult, par, loss, combine_pdf, _ = Unbinned_2d_fit_mom_time(
            mom_ak, times_ak, categories,
            tuple(args.fit_range_mom), tuple(args.fit_range_time),
            plot_truth=False, verbose=args.verbose,
            plot_NLL=False, plot_results=False
        )
    finally:
        if _ce_backup is not None:
            mom_components['CE'] = _ce_backup

    # Build scanning grid
    mu_grid = np.linspace(args.mu_min, args.mu_max, args.n_mu)

    # Run 2D Parallel Scan Engine
    from sensitivity_runners import parallel_toy_scan_2d
    results = parallel_toy_scan_2d(
        mu_grid=mu_grid,
        ntoys=args.ntoys,
        fit_result=fitresult,
        component_cats=categories,
        fit_range_mom=args.fit_range_mom,
        fit_range_time=args.fit_range_time,
        constraints_dir=args.constraints_dir,
        verbose=args.verbose,
        plot_toys=args.plot_toys,
        plot_dir=args.plot_dir,
        n_workers=args.n_workers
    )

    # Output Results to CSV
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mu', 'n_success', 'n_failed', 'median_ul', 'p16', 'p84'])
        for mu in sorted(results.keys()):
            r = results[mu]
            writer.writerow([mu, r['n_success'], r['n_failed'], r['median'], r['p16'], r['p84']])

    # Plot Sensitivity
    mus = sorted(results.keys())
    med = [results[m]['median'] for m in mus]
    p16 = [results[m]['p16'] for m in mus]
    p84 = [results[m]['p84'] for m in mus]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mus, med, marker='o', color='black', label='Median UL$_{90}$ 2D')
    ax.fill_between(mus, p16, p84, alpha=0.35, color='forestgreen', label='68% Expected Band')
    ax.set_xlabel('Injected 2D signal $N_{CE}$')
    ax.set_ylabel('Upper limit on $N_{CE}$ (90% CL)')
    ax.set_title('2D Sensitivity Scan (Momentum + Time)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()