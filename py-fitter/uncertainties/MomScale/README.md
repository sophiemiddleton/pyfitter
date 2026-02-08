
# MomScale: momentum-scale systematic helpers

This small package contains utilities to produce and summarize simple
momentum-scale systematic variations for use in fitter tests.

Scripts
-------
- `scale_momentum.py` — Create a momentum-scaled variant from a baseline NPZ.
	- Runs `process.py` (if needed) to ensure baseline NPZs exist, finds a suitable
		1D momentum array inside the produced NPZ, multiplies the values by the
		provided `--scale` factor, and writes a compressed NPZ and a diagnostic PNG
		into the output directory (default: `systematics/outputs`).
	- Key arguments:
		- `--scale` (float, required): multiplicative factor, e.g. `1.01` for +1%.
		- `--file` (str, required): input list / config used by `process.py`.
		- `--loc` (str): location flag forwarded to `process.py` (default: `tape`).
		- `--outdir` (str): destination directory for NPZ/PNG (default: `systematics/outputs`).
	- Output naming: `{basename}__sys-scale_mom-{scale:.4f}.npz` and a PNG with
		the same prefix.

- `collect_results.py` — Scan a directory of NPZ outputs and write a CSV
	summary with simple statistics (count, mean, std, median) for the first
	1D numeric array found in each NPZ.
	- Key arguments:
		- `--dir` (default: `systematics/outputs`) — directory to scan.
		- `--out` (default: `systematics/summary.csv`) — CSV output path.

Example usage
-------------
Run the baseline processing (if needed) and produce a +1% momentum systematic:

```bash
cd py-fitter
python uncertainties/MomScale/scale_momentum.py --scale 1.01 --file unit_test/test.txt --loc tape
```

Collect summaries of produced NPZs:

```bash
python uncertainties/MomScale/collect_results.py --dir systematics/outputs --out systematics/summary.csv
```

Notes
-----
- `scale_momentum.py` attempts to be tolerant about NPZ contents: it will
	search for a 1D numeric array (or coerce object arrays) and treat that as the
	momentum array. If you know the exact key/name, it's easiest to prepare an
	NPZ with that key or edit the script for a strict mapping.
- The produced NPZs are *data variations* (scaled momenta). They are not the
	same format as `templates.npz` used by the fitter's uncertainty loader. To
	convert these into shape templates compatible with the fitter, create a
	`templates.npz` containing nominal/up/down histograms or arrays and place it
	inside an uncertainties package directory used as `constraints_dir`.

````