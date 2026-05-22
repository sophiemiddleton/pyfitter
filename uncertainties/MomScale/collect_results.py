#!/usr/bin/env python3
"""
collect_results.py

Scan a directory of systematic NPZ outputs and produce a CSV summary with simple
statistics (n, mean, std, median) for the stored 1D momentum arrays.

Usage:
  python collect_results.py --dir systematics/outputs --out systematics/summary.csv
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def load_1d_array_from_npz(path):
    data = np.load(path)
    for k in data.files:
        arr = data[k]
        a = np.asarray(arr)
        if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
            return a
    return None


def summarize(dirpath, out_csv):
    p = Path(dirpath)
    rows = []
    for npz in sorted(p.glob('*.npz')):
        arr = load_1d_array_from_npz(npz)
        if arr is None:
            continue
        rows.append({'file': npz.name, 'n': int(arr.size), 'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'median': float(np.median(arr))})

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'n', 'mean', 'std', 'median'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Wrote summary to', out_csv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='systematics/outputs')
    p.add_argument('--out', default='systematics/summary.csv')
    args = p.parse_args()
    summarize(args.dir, args.out)


if __name__ == '__main__':
    main()
