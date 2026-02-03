#!/usr/bin/env python3
"""
Template: scale_momentum.py

Creates a momentum-scaled systematic variation by running the baseline `process.py`,
finding the produced momentum NPZ, scaling the momentum values, saving a new NPZ
and a diagnostic histogram into `systematics/outputs/`.

Usage:
  python scale_momentum.py --scale 1.01 --file unit_test/test.txt --loc tape
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
OUTPUT_DIR = HERE / 'systematics' / 'outputs'


def find_baseline_npz(basename, here):
    # Prefer files explicitly mentioning 'mom' (e.g. *_mom_mag.npz)
    candidates = list(here.glob(f"{basename}*_mom*.npz"))
    candidates = [c for c in candidates if c.is_file()]
    if candidates:
        return candidates[0]

    # If none found, scan all npz files for a numeric 1D array that looks like momentum
    all_npz = list(here.glob(f"{basename}*.npz")) + list(here.glob(f"*.npz"))
    all_npz = [c for c in all_npz if c.is_file()]

    def score_npz_for_mom(path):
        try:
            data = np.load(path, allow_pickle=True)
        except Exception:
            return None
        best = None
        for k in data.files:
            try:
                a = np.asarray(data[k])
            except Exception:
                continue
            if a.ndim != 1:
                continue
            if not np.issubdtype(a.dtype, np.number):
                # try to coerce object arrays to numeric
                try:
                    a = np.asarray([float(x) for x in a])
                except Exception:
                    continue
            # score by how close the median is to expected momentum range
            med = float(np.median(a)) if a.size > 0 else 0.0
            # prefer med between 50 and 200 MeV/c
            if 50.0 <= med <= 200.0:
                return (path, med)
            # otherwise record as fallback candidate
            if best is None:
                best = (path, med)
        return best

    # find first good candidate with median in physical range
    for p in all_npz:
        res = score_npz_for_mom(p)
        if res is not None and 50.0 <= res[1] <= 200.0:
            return res[0]

    # otherwise return first fallback candidate if any
    for p in all_npz:
        res = score_npz_for_mom(p)
        if res is not None:
            return res[0]

    return None


def load_1d_array_from_npz(path):
    # allow_pickle True because some saved arrays may be object arrays
    data = np.load(path, allow_pickle=True)
    # return first 1D array-like found, coercing object arrays when possible
    for k in data.files:
        arr = data[k]
        try:
            a = np.asarray(arr)
        except Exception:
            continue

        # If it's already numeric 1D, return it
        if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
            return a, k

        # If object dtype, try to coerce to a flat numeric array
        if a.ndim == 1 and a.dtype == object:
            out = []
            for elem in a:
                try:
                    # element may be scalar-like
                    out.append(float(elem))
                    continue
                except Exception:
                    pass
                # element may be an array-like; try to flatten
                try:
                    sub = np.asarray(elem)
                    if sub.size > 0 and np.issubdtype(sub.dtype, np.number):
                        out.extend(sub.flatten().tolist())
                        continue
                except Exception:
                    pass
                # last resort: try parsing string repr
                try:
                    val = float(str(elem))
                    out.append(val)
                    continue
                except Exception:
                    # skip uncoercible entries
                    continue

            if len(out) > 0:
                return np.asarray(out, dtype=float), k

    raise RuntimeError(f"No 1D numeric array found in {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=float, required=True)
    p.add_argument('--file', required=True)
    p.add_argument('--loc', default='tape')
    p.add_argument('--outdir', default=str(OUTPUT_DIR))
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Run baseline process.py to produce baseline NPZs if needed
    print('Running baseline process.py (may be no-op if outputs exist)')
    cmd = [sys.executable, 'process.py', '--file', str(args.file), '--loc', args.loc, '--fittype', '2D']
    proc = subprocess.run(cmd, cwd=HERE)
    if proc.returncode != 0:
        print('process.py returned non-zero; continuing to search for existing outputs')

    basename = os.path.splitext(os.path.basename(args.file))[0]
    baseline = find_baseline_npz(basename, HERE)
    if baseline is None:
        print('Baseline NPZ not found. Ensure process.py produced a *_mom_mag.npz or similar in', HERE)
        sys.exit(2)

    print('Found baseline NPZ:', baseline)
    arr, key = load_1d_array_from_npz(baseline)

    scaled = arr * args.scale

    out_name = f"{basename}__sys-scale_mom-{args.scale:.4f}.npz"
    out_path = outdir / out_name
    np.savez_compressed(out_path, mom_scaled=scaled)

    # diagnostic plot
    fig, ax = plt.subplots()
    ax.hist(arr, bins=80, alpha=0.6, label='baseline')
    ax.hist(scaled, bins=80, alpha=0.6, label=f'scaled x{args.scale}')
    ax.set_xlabel('momentum')
    ax.set_ylabel('counts')
    ax.legend()
    png_name = f"{basename}__sys-scale_mom-{args.scale:.4f}.png"
    fig_path = outdir / png_name
    fig.savefig(fig_path)
    plt.close(fig)

    print('Wrote:', out_path, fig_path)


if __name__ == '__main__':
    main()
