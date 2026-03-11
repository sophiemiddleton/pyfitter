"""Driver to run a toy-based sensitivity scan using the project's fit code.

Workflow:
  - Load a nominal momentum array (NPZ file containing a 1D numeric array, key e.g. 'mom_mag')
  - Run `Unbinned_fit_mom` on the nominal data to obtain `combine_pdf` and POI parameter
  - Run `toy_scan_from_model` from `sensitivity_runners` to build toy ensembles at several injected signal strengths
  - Save CSV summary and a median-vs-mu plot

Usage example:
  python run_sensitivity_scan.py --data nominal_mom.npz --mu-min 0 --mu-max 50 --n-mu 6 --ntoys 200 --n-per-toy 1000

"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import awkward as ak

from fit_module import Unbinned_fit_mom
from sensitivity_runners import toy_scan_from_model, fit_runner_1d_ul


def load_1d_from_npz(path):
    arr = np.load(path, allow_pickle=True)
    # prefer key 'mom_mag' if present
    if 'mom_mag' in arr.files:
        return np.asarray(arr['mom_mag'])
    # otherwise return first 1D numeric array
    for k in arr.files:
        a = np.asarray(arr[k])
        if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
            return a
    raise RuntimeError(f'No 1D numeric array found in {path}')


def main():
    p = argparse.ArgumentParser(description='Run toy-based sensitivity scan')
    p.add_argument('--data', required=True, help='NPZ file with nominal momentum array')
    p.add_argument('--fit-range', nargs=2, type=float, default=[95.0, 115.0], help='momentum fit range')
    p.add_argument('--constraints-dir', default=None, help='uncertainties package directory')
    p.add_argument('--mu-min', type=float, default=0.0)
    p.add_argument('--mu-max', type=float, default=30.0)
    p.add_argument('--n-mu', type=int, default=7)
    p.add_argument('--background-only', action='store_true', help='Run background-only (no injected signal) scan: override mu grid to [0]')
    p.add_argument('--toy-out', default=None, help='If set, save per-toy numeric metrics to an NPZ file')
    p.add_argument('--results-out', default=None, help='If set, save full results (including errors) to a JSON file')
    p.add_argument('--ntoys', type=int, default=100)
    p.add_argument('--n-per-toy', type=int, default=1000)
    p.add_argument('--out', default='sensitivity_scan.csv')
    p.add_argument('--plot', default='sensitivity_scan.png')
    p.add_argument('--verbose', type=int, default=1)
    p.add_argument('--plot-toys', type=int, default=0, help='Save plots for first N toys per mu (0 = disabled)')
    p.add_argument('--plot-dir', default=None, help='Directory to save per-toy plots when --plot-toys > 0')
    p.add_argument('--save-nominal-out', default=None, help='If set, save the nominal mom (and time if present) arrays to this NPZ file')
    p.add_argument('--compute-sigmas', action='store_true', help='Compute per-toy discovery significance (slower)')
    p.add_argument('--sig-calc-opt', choices=['asym','freq'], default='asym', help='Calculator for significance: asym or freq')
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} not found')

    mom = load_1d_from_npz(str(data_path))
    # convert to awkward for fitter
    mom_ak = ak.Array(mom)

    # run an initial fit to get combine_pdf and POI parameter
    if args.verbose:
        print('Running initial fit on nominal data...')
    fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
        mom_ak,
        [],
        [],
        args.fit_range[0],
        args.fit_range[1],
        False,
        args.verbose,
        minos=False,
        plot_NLL=False,
        plot_results=False,
        constraints_dir=args.constraints_dir,
    )

    # Optionally save the nominal input arrays used for the scan
    if args.save_nominal_out:
        try:
            if args.verbose:
                print(f'Saving nominal data to {args.save_nominal_out} ...')
            # Save momentum array; time not available in this driver
            np.savez(args.save_nominal_out, mom_mag=np.asarray(mom))
            if args.verbose:
                print(f'Wrote nominal arrays to {args.save_nominal_out}')
        except Exception as e:
            print(f'Failed to write nominal NPZ {args.save_nominal_out}: {e}')

    # prepare mu grid (allow quick background-only UL scenario)
    if args.background_only:
        mu_grid = np.array([0.0])
    else:
        mu_grid = np.linspace(args.mu_min, args.mu_max, args.n_mu)

    if args.verbose:
        print('Starting toy scan with mu grid:', mu_grid)

    results = toy_scan_from_model(
        combine_pdf,
        par,
        fit_runner_1d_ul,
        mu_grid,
        ntoys=args.ntoys,
        n_per_toy=args.n_per_toy,
        fit_runner_args=(),
        fit_runner_kwargs={'fit_range': tuple(args.fit_range), 'constraints_dir': args.constraints_dir, 'verbose': args.verbose},
        plot_first_n=args.plot_toys,
        plot_dir=args.plot_dir,
        compute_sigmas=args.compute_sigmas,
        sig_calc_opt=args.sig_calc_opt,
        verbose=args.verbose,
    )

    # write CSV summary
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mu', 'n_success', 'median', 'p16', 'p84'])
        for mu in sorted(results.keys()):
            r = results[mu]
            writer.writerow([mu, len(r['values']), r['median'], r['p16'], r['p84']])

    if args.verbose:
        print(f'Wrote summary to {args.out}')

    # optionally save per-toy values for further UL-only analysis
    if args.toy_out is not None:
        save_dict = {}
        for mu in sorted(results.keys()):
            save_dict[f'mu_{mu}'] = np.asarray(results[mu]['values'], dtype=float)
        np.savez(args.toy_out, **save_dict)
        if args.verbose:
            print(f'Wrote per-toy values to {args.toy_out}')

    # optionally save full results (including error tracebacks) as JSON
    if args.results_out is not None:
        def _conv_float(x):
            try:
                xf = float(x)
                if np.isnan(xf):
                    return None
                return xf
            except Exception:
                return None

        serial = {}
        for mu in sorted(results.keys()):
            r = results[mu]
            serial_mu = {
                'n_success': int(len(r.get('values', []))),
                'median': _conv_float(r.get('median', None)),
                'p16': _conv_float(r.get('p16', None)),
                'p84': _conv_float(r.get('p84', None)),
                'values': [ _conv_float(v) for v in r.get('values', []) ],
                'errors': r.get('errors', []) if isinstance(r.get('errors', []), list) else [r.get('errors')],
            }
            serial[str(mu)] = serial_mu
        with open(args.results_out, 'w') as jf:
            json.dump(serial, jf, indent=2)
        if args.verbose:
            print(f'Wrote full results (with errors) to {args.results_out}')

    # plot median vs mu
    mus = sorted(results.keys())
    med = [results[m]['median'] for m in mus]
    p16 = [results[m]['p16'] for m in mus]
    p84 = [results[m]['p84'] for m in mus]

    fig, ax = plt.subplots()
    ax.plot(mus, med, marker='o', label='median')
    ax.fill_between(mus, p16, p84, color='gray', alpha=0.4, label='1 sigma')
    ax.set_xlabel('Injected signal (mu)')
    ax.set_ylabel('Metric (UL proxy)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plot)
    if args.verbose:
        print(f'Wrote plot to {args.plot}')


if __name__ == '__main__':
    main()
