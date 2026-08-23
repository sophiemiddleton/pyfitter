"""Driver for the full analysis chain.

Stage 1: run the selection over each input dataset file list, producing one
parquet file and one cut-flow CSV per dataset inside a tag directory.
Stage 2: build the input card and run parquet_fit_builder on those parquets.
Stage 3: run the CLs limit on the datacard the fit produced.

    <tag>/cutflows/<dataset>.csv
    <tag>/parquet/<dataset>_postcut.parquet
    <tag>/manifest.yaml
    <tag>/input_card.yaml   # expected yields + systematics for parquet_fit_builder
    <tag>/fit/              # fit snapshot and output card
    <tag>/limits/           # CLs limit results

Datasets, their initial acceptances and the fit type live in the registry
`datasets.yaml` (override with --registry), so nothing has to be typed on the
command line:

    python run_chain.py --tag v1                        # every dataset in the registry
    python run_chain.py --tag v1 --group mix            # one named group
    python run_chain.py --tag v1 --datasets CeM_mix     # selected datasets by name
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from config import GLOBAL_VERBOSITY
from uncertainties_config import STANDARD_SYSTEMATICS
from pyutils.pylogger import Logger

logger = Logger(print_prefix='[run_chain] ', verbosity=GLOBAL_VERBOSITY)

BASE_DIR = Path(__file__).resolve().parent
PROCESS_SCRIPT = BASE_DIR / 'process.py'
FIT_SCRIPT = BASE_DIR / 'sensitivity_scan' / 'parquet_fit_builder.py'
LIMIT_SCRIPT = BASE_DIR / 'sensitivity_scan' / 'simple_limit_combine_12d.py'
DEFAULT_REGISTRY = BASE_DIR / 'datasets.yaml'

# Cosmic acceptance: ratio of run1A time to full sample
# Used to scale cosmic yield from data to full equivalent luminosity
COSMIC_ACCEPTANCE = 0.175999365


def load_registry(path):
    """Load the dataset registry: name -> {filelist, acceptance, process}."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    entries = cfg.get('datasets')
    if not entries:
        raise ValueError(f'No "datasets" section found in {path}')

    registry = {}
    for name, entry in entries.items():
        if 'filelist' not in entry:
            raise ValueError(f'Dataset "{name}" in {path} has no "filelist"')
        filelist = entry['filelist']
        registry[str(name)] = {
            'name': str(name),
            'filelist': filelist if os.path.isabs(filelist) else str(BASE_DIR / filelist),
            'acceptance': float(entry.get('acceptance', 1.0)),
            'expected': float(entry['expected']) if entry.get('expected') is not None else None,
            'process': entry.get('process'),
        }

    return registry, cfg


def load_datasets(args):
    """Resolve which registry datasets to run, in a deterministic order."""
    registry, cfg = load_registry(args.registry)
    groups = cfg.get('groups', {}) or {}

    if args.group:
        if args.group not in groups:
            raise ValueError(f'Group "{args.group}" not found in {args.registry}. Available: {", ".join(sorted(groups))}')
        names = list(groups[args.group])
    elif args.datasets:
        names = list(args.datasets)
    else:
        names = list(registry)

    datasets = []
    for name in names:
        # Allow a bare file-list path as well as a registry key
        if name not in registry and name.endswith('.txt'):
            from process import derive_output_basename
            derived = derive_output_basename(name)
            if derived in registry:
                name = derived
            else:
                logger.log(f'[{derived}] not in registry, using acceptance 1.0', 'warning')
                datasets.append({
                    'name': derived,
                    'filelist': name,
                    'acceptance': 1.0,
                    'expected': None,
                    'process': None,
                })
                continue
        if name not in registry:
            raise ValueError(f'Dataset "{name}" not found in {args.registry}. Available: {", ".join(sorted(registry))}')
        datasets.append(dict(registry[name]))

    return datasets, cfg


def run_selection(dataset, tag_dir, args):
    """Run process.py in --nofit mode for a single dataset."""
    cmd = [
        sys.executable, str(PROCESS_SCRIPT),
        '--file', dataset['filelist'],
        '--nofit',
        '--outdir', str(tag_dir),
        '--loc', args.loc,
        '--jobs', str(args.jobs),
        '--fitrange_low', *[str(v) for v in args.fitrange_low],
        '--fitrange_hi', *[str(v) for v in args.fitrange_hi],
        '--verbose', str(args.verbose),
        '--version', args.version,
    ]

    logger.log(f'[{dataset["name"]}] {" ".join(cmd)}', 'info')
    if args.dry_run:
        return True

    log_path = tag_dir / 'logs' / f'{dataset["name"]}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    if proc.returncode != 0:
        logger.log(f'[{dataset["name"]}] selection failed (exit {proc.returncode}), see {log_path}', 'error')
        return False

    logger.log(f'[{dataset["name"]}] selection done, log: {log_path}', 'info')
    return True


def final_cut_efficiency(cutflow_path):
    """Absolute efficiency of the last cut in a cut-flow CSV, as a fraction."""
    if not os.path.exists(cutflow_path):
        return None
    with open(cutflow_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    last = rows[-1]
    absolute = last.get('absolute_frac') or last.get('Absolute [%]')
    if absolute in (None, ''):
        return None
    return float(absolute) / 100.0


def final_cut_yield(cutflow_path):
    """Return the events passing the final cut in a cut-flow CSV."""
    if not os.path.exists(cutflow_path):
        return None
    with open(cutflow_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    value = rows[-1].get('events_passing') or rows[-1].get('Events Passing')
    return int(float(value)) if value not in (None, '') else None


def canonical_process_name(process):
    """Map registry aliases to the component names understood by the fitter."""
    if process is None:
        return None
    name = str(process).strip().lower()
    if name.startswith('rpc'):
        return 'RPC'
    if name.startswith('rmc'):
        return 'RMC'
    if name in ('cosmic', 'cosmics'):
        return 'Cosmic'
    if name in ('ce', 'cem'):
        return 'CE'
    return str(process)


def scaled_expected_yield(expected, acceptance_factor, process):
    """Scale expected yield by acceptance factor for final fit input.
    
    For MC (CE, DIO, RPC, RMC): acceptance_factor = net_acceptance (acceptance * cut_efficiency)
    For Cosmic: acceptance_factor = acceptance only (already post-cuts)
    """
    if expected is None:
        return None
    if acceptance_factor is None:
        return None
    return float(expected) * float(acceptance_factor)


def write_manifest(tag_dir, datasets, args, statuses):
    """Record dataset provenance, acceptances and output paths for later stages."""
    manifest = {
        'tag': args.tag,
        'registry': str(Path(args.registry).resolve()),
        'fitrange_low': list(args.fitrange_low),
        'fitrange_hi': list(args.fitrange_hi),
        'location': args.loc,
        'datasets': [],
    }
    for dataset in datasets:
        name = dataset['name']
        # Derive cutflow filename from filelist basename (matches process.py logic)
        filelist_basename = os.path.splitext(os.path.basename(dataset['filelist']))[0]
        cutflow_path = tag_dir / 'cutflows' / f'{filelist_basename}.csv'
        efficiency = final_cut_efficiency(cutflow_path)
        expected = dataset.get('expected')
        is_cosmic = str(dataset.get('process', '')).lower() == 'cosmic'
        
        if is_cosmic:
            expected = final_cut_yield(cutflow_path)
            acceptance = COSMIC_ACCEPTANCE  # Use hardcoded cosmic acceptance (run1A / full sample)
            if expected is None and statuses[name]:
                logger.log(f'[{name}] could not read cosmic yield from {cutflow_path}', 'warning')
        else:
            acceptance = dataset['acceptance']
        
        if efficiency is None and statuses[name]:
            logger.log(f'[{name}] could not read final cut efficiency from {cutflow_path}', 'warning')
        
        net_acceptance = acceptance * efficiency if efficiency is not None else None
        
        # For cosmics, scale by acceptance only (expected is already post-cuts)
        # For MC, scale by net_acceptance (acceptance * efficiency)
        yield_scale_factor = acceptance if is_cosmic else net_acceptance
        
        manifest['datasets'].append({
            'name': name,
            'process': canonical_process_name(dataset.get('process')),
            'filelist': os.path.abspath(dataset['filelist']),
            'acceptance': acceptance,
            'expected': expected,
            'cut_efficiency': efficiency,
            'net_acceptance': net_acceptance,
            'scaled_expected_yield': scaled_expected_yield(expected, yield_scale_factor, dataset.get('process')),
            'cutflow': str(cutflow_path),
            'parquet': str(tag_dir / 'parquet' / f'{filelist_basename}_postcut.parquet'),
            'npz': str(tag_dir / f'{filelist_basename}_mom_mag.npz'),
            'status': 'ok' if statuses[name] else 'failed',
        })

    manifest_path = tag_dir / 'manifest.yaml'
    with open(manifest_path, 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    logger.log(f'Wrote manifest to {manifest_path}', 'info')
    return manifest, manifest_path


def refresh_manifest_from_registry(manifest, registry_path):
    """Refresh registry values and cutflow-derived yields without rerunning selection."""
    registry, _ = load_registry(registry_path)
    for entry in manifest.get('datasets', []):
        dataset = registry.get(entry.get('name'))
        if dataset is None:
            logger.log(f'[{entry.get("name")}] not found in registry; preserving manifest values', 'warning')
            continue
        entry['process'] = canonical_process_name(dataset.get('process'))
        is_cosmic = str(entry.get('process', '')).lower() == 'cosmic'
        
        if is_cosmic:
            entry['acceptance'] = COSMIC_ACCEPTANCE  # Use hardcoded cosmic acceptance (run1A / full sample)
            entry['expected'] = final_cut_yield(entry['cutflow'])
        else:
            entry['acceptance'] = dataset['acceptance']
            entry['expected'] = dataset.get('expected')
        
        efficiency = final_cut_efficiency(entry['cutflow'])
        if efficiency is None:
            logger.log(f'[{entry.get("name")}] could not read final cut efficiency from {entry["cutflow"]}', 'warning')
        entry['cut_efficiency'] = efficiency
        entry['net_acceptance'] = entry['acceptance'] * efficiency if efficiency is not None else None
        
        # For cosmics, scale by acceptance only (expected is already post-cuts)
        # For MC, scale by net_acceptance (acceptance * efficiency)
        yield_scale_factor = entry['acceptance'] if is_cosmic else entry['net_acceptance']
        entry['scaled_expected_yield'] = scaled_expected_yield(
            entry['expected'], yield_scale_factor, entry.get('process')
        )
    manifest['registry'] = str(Path(registry_path).resolve())
    return manifest


def write_input_card(tag_dir, manifest, args):
    """Build the parquet_fit_builder input card from the manifest expected yields."""
    processes = {}
    components = {}
    for entry in manifest['datasets']:
        process = canonical_process_name(entry.get('process'))
        if not process:
            logger.log(f'[{entry["name"]}] no process assigned, omitted from input card', 'warning')
            continue
        expected = entry.get('expected')
        is_cosmic = process.lower() == 'cosmic'
        
        if is_cosmic:
            expected = final_cut_yield(entry['cutflow'])
            if expected is None:
                logger.log(f'[{entry["name"]}] could not read cosmic yield from {entry["cutflow"]}', 'warning')
                continue
            # For cosmics, scale by acceptance only (expected is already post-cuts)
            acceptance_factor = entry.get('acceptance')
        else:
            # For MC, scale by net_acceptance (acceptance * efficiency)
            acceptance_factor = entry.get('net_acceptance')
        
        scaled_yield = scaled_expected_yield(expected, acceptance_factor, process)
        if scaled_yield is None:
            logger.log(f'[{entry["name"]}] no expected yield in registry, omitted from input card', 'warning')
            continue
        if entry.get('net_acceptance') is None:
            logger.log(f'[{entry["name"]}] no net acceptance, omitted from input card', 'warning')
            continue
        # Datasets sharing a process (e.g. internal + external RPC) add up
        processes.setdefault(process, 0.0)
        processes[process] += float(scaled_yield)
        components.setdefault(process.lower(), []).append(entry['parquet'])

    if not processes:
        logger.log('No processes with expected yields, skipping input card', 'warning')
        return None, {}

    systematics = {}
    for syst_name, syst in STANDARD_SYSTEMATICS.items():
        effects = {p: v for p, v in syst['processes'].items() if p in processes}
        if effects:
            systematics[syst_name] = {
                'type': syst['type'],
                'description': syst['description'],
                'effects': effects,
            }

    card = {
        'name': f'{args.tag}_input_card',
        'metadata': {
            'experiment': 'Mu2e',
            'signal_process': args.signal_process,
            'description': f'Auto-generated by run_chain.py for tag {args.tag}',
        },
        'observables': {
            'mom': [float(args.fitrange_low[0]), float(args.fitrange_hi[0])],
            'time': [float(args.fitrange_low[1]), float(args.fitrange_hi[1])],
        },
        'processes': {name: {'yield': value} for name, value in processes.items()},
        'systematics': systematics,
        'poi': {
            'name': args.signal_process,
            'initial': processes.get(args.signal_process, 1.0),
            'range': [0.0, 40.0],
        },
    }

    card_path = tag_dir / 'input_card.yaml'
    with open(card_path, 'w') as f:
        yaml.safe_dump(card, f, sort_keys=False)
    logger.log(f'Wrote input card to {card_path}', 'info')

    # The fitter takes one parquet per component, so datasets sharing a process are merged
    resolved = {}
    for token, paths in sorted(components.items()):
        if len(paths) == 1:
            resolved[token] = paths[0]
            continue
        merged = tag_dir / 'parquet' / f'{token}_combined_postcut.parquet'
        frames = [pd.read_parquet(p) for p in paths]
        pd.concat(frames, ignore_index=True).to_parquet(merged, index=False)
        logger.log(f'Merged {len(paths)} parquet files into {merged}', 'info')
        resolved[token] = str(merged)

    return card_path, resolved


def collect_plots(dest_dir, since):
    """Move plots written to the working directory into the tag directory."""
    moved = []
    for png in BASE_DIR.glob('*.png'):
        if png.stat().st_mtime < since:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(png), str(dest_dir / png.name))
        moved.append(png.name)
    if moved:
        logger.log(f'Moved {len(moved)} plot(s) to {dest_dir}: {", ".join(moved)}', 'info')
    return moved


def run_fit(tag_dir, card_path, components, fit_type, args):
    """Run parquet_fit_builder on the tag's parquet files and generated input card."""
    if not components:
        logger.log('No components available, skipping fit', 'error')
        return False

    fit_dir = tag_dir / 'fit'
    fit_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(FIT_SCRIPT),
        '--fit-type', fit_type,
        '--card', str(card_path),
        '--components', *[f'{token}={path}' for token, path in sorted(components.items())],
        '--fit-range-lo', str(args.fitrange_low[0]),
        '--fit-range-hi', str(args.fitrange_hi[0]),
        '--time-range-lo', str(args.fitrange_low[1]),
        '--time-range-hi', str(args.fitrange_hi[1]),
        '--jobs', str(args.jobs),
        '--verbosity', str(args.verbose),
        '--datacard-snapshot', str(fit_dir / f'{Path(args.tag).name}_snapshot.npz'),
        '--output-card', str(fit_dir / 'output_card.yaml'),
    ]

    logger.log(f'[fit] {" ".join(cmd)}', 'info')
    if args.dry_run:
        return True

    log_path = tag_dir / 'logs' / 'fit.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR))

    collect_plots(tag_dir / 'fit' / 'plots', started)

    if proc.returncode != 0:
        logger.log(f'[fit] parquet_fit_builder failed (exit {proc.returncode}), see {log_path}', 'error')
        return False

    logger.log(f'[fit] done, log: {log_path}', 'info')
    return True


def run_limit(tag_dir, fit_type, limit_cfg, args, datacard_path=None):
    """Run the CLs limit on the datacard produced by the fit stage."""
    datacard = Path(datacard_path) if datacard_path is not None else (tag_dir / 'fit' / 'output_card.yaml')
    if not args.dry_run and not datacard.exists():
        logger.log(f'[limit] datacard not found: {datacard}', 'error')
        return False

    limit_dir = tag_dir / 'limits'
    limit_dir.mkdir(parents=True, exist_ok=True)

    # The limit script fits either 1d or 2d; "both" continues with the 2d card
    dim = '1d' if fit_type == '1d' else '2d'

    cmd = [
        sys.executable, str(LIMIT_SCRIPT),
        '--datacard', str(datacard),
        '--dim', dim,
        '--toys', str(args.toys if args.toys is not None else limit_cfg.get('toys', 1000)),
        '--seed', str(limit_cfg.get('seed', 42)),
        '--cl', str(limit_cfg.get('cl', 0.90)),
        '--output-dir', str(limit_dir),
        '--verbosity', str(args.verbose),
    ]
    if args.freeze_nuisances or limit_cfg.get('freeze_nuisances'):
        cmd.append('--freeze-nuisances')

    logger.log(f'[limit] {" ".join(cmd)}', 'info')
    if args.dry_run:
        return True

    log_path = tag_dir / 'logs' / 'limit.log'
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR))

    if proc.returncode != 0:
        logger.log(f'[limit] simple_limit_combine_12d failed (exit {proc.returncode}), see {log_path}', 'error')
        return False

    logger.log(f'[limit] done, results in {limit_dir}, log: {log_path}', 'info')
    return True


def main(args):
    tag_dir = Path(args.tag).resolve()
    for sub in ('cutflows', 'parquet'):
        (tag_dir / sub).mkdir(parents=True, exist_ok=True)
    logger.log(f'Tag directory: {tag_dir}', 'info')

    if args.limit_only:
        if args.manifest_only:
            logger.log('Choose only one mode: --limit-only or --manifest-only', 'error')
            return 1
        _, cfg = load_registry(args.registry)
        fit_type = args.fit_type or str(cfg.get('fit', {}).get('type', '2d')).lower()
        if fit_type not in ('1d', '2d', 'both'):
            raise ValueError(f'Invalid fit type "{fit_type}", expected 1d, 2d or both')
        datacard = Path(args.limit_card).resolve() if args.limit_card else (tag_dir / 'fit' / 'output_card.yaml')
        logger.log(f'Running limit-only mode from datacard: {datacard}', 'info')
        return 0 if run_limit(tag_dir, fit_type, cfg.get('limit', {}) or {}, args, datacard_path=datacard) else 1

    if args.manifest_only:
        manifest_path = tag_dir / 'manifest.yaml'
        if not manifest_path.exists():
            logger.log(f'No manifest to refresh: {manifest_path}', 'error')
            return 1
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        manifest = refresh_manifest_from_registry(manifest, args.registry)
        with open(manifest_path, 'w') as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        logger.log(f'Refreshed manifest from existing cutflows: {manifest_path}', 'info')
        return 0

    if args.skip_selection:
        _, cfg = load_registry(args.registry)
        manifest_path = tag_dir / 'manifest.yaml'
        if not manifest_path.exists():
            logger.log(f'No manifest to resume from: {manifest_path}', 'error')
            return 1
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)
        logger.log(f'Resuming from {manifest_path}', 'info')
        if args.refresh_registry:
            manifest = refresh_manifest_from_registry(manifest, args.registry)
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(manifest, f, sort_keys=False)
            logger.log(f'Refreshed registry values in {manifest_path}', 'info')
    else:
        datasets, cfg = load_datasets(args)

    fit_type = args.fit_type or str(cfg.get('fit', {}).get('type', '2d')).lower()
    if fit_type not in ('1d', '2d', 'both'):
        raise ValueError(f'Invalid fit type "{fit_type}", expected 1d, 2d or both')

    if not args.skip_selection:
        logger.log(f'Processing {len(datasets)} dataset(s) from {args.registry}', 'info')
        for dataset in datasets:
            logger.log(f'  {dataset["name"]}: acceptance={dataset["acceptance"]}, filelist={dataset["filelist"]}', 'info')

        statuses = {}
        for dataset in datasets:
            if not os.path.exists(dataset['filelist']):
                logger.log(f'[{dataset["name"]}] file list not found: {dataset["filelist"]}', 'error')
                statuses[dataset['name']] = False
                continue
            statuses[dataset['name']] = run_selection(dataset, tag_dir, args)

        manifest, _ = write_manifest(tag_dir, datasets, args, statuses)

        failed = [name for name, ok in statuses.items() if not ok]
        if failed:
            logger.log(f'Datasets that failed: {", ".join(failed)}', 'error')
            return 1
        logger.log('All datasets processed successfully', 'info')

    card_path, components = write_input_card(tag_dir, manifest, args)

    if args.skip_fit:
        logger.log('Skipping fit stage (--skip-fit)', 'info')
        return 0
    if card_path is None:
        return 1

    logger.log(f'Running {fit_type} fit', 'info')
    if not run_fit(tag_dir, card_path, components, fit_type, args):
        return 1

    if args.skip_limit:
        logger.log('Skipping limit stage (--skip-limit)', 'info')
        return 0

    logger.log('Running CLs limit', 'info')
    return 0 if run_limit(tag_dir, fit_type, cfg.get('limit', {}) or {}, args) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run the analysis chain over a list of datasets',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--tag', type=str, required=True, help='Output tag directory')
    parser.add_argument('--datasets', type=str, nargs='+', help='Registry dataset names to run (default: all in the registry)')
    parser.add_argument('--group', type=str, default=None, help='Named group from the registry "groups" section')
    parser.add_argument('--registry', type=str, default=str(DEFAULT_REGISTRY), help='YAML dataset registry holding file lists and initial acceptances')
    parser.add_argument('--version', type=str, default='79_v02', help='Analysis version for cut configuration (e.g., "79_v02", "80")')
    parser.add_argument('--signal-process', dest='signal_process', type=str, default='CE', help='Signal process name used as the POI in the generated input card')
    parser.add_argument('--fit-type', dest='fit_type', choices=['1d', '2d', 'both'], default=None, help='Override the fit type set in the registry "fit" section')
    parser.add_argument('--skip-fit', dest='skip_fit', action='store_true', help='Stop after the selection stage, do not run parquet_fit_builder')
    parser.add_argument('--skip-selection', dest='skip_selection', action='store_true', help='Reuse the existing <tag>/manifest.yaml and run only the fit and limit stages')
    parser.add_argument('--refresh-registry', dest='refresh_registry', action='store_true', help='With --skip-selection, reload acceptance, expected and process values from the registry')
    parser.add_argument('--manifest-only', dest='manifest_only', action='store_true', help='Refresh an existing manifest from its cutflows and registry, then stop')
    parser.add_argument('--limit-only', dest='limit_only', action='store_true', help='Run only the limit stage from an existing output card, skip selection and fit')
    parser.add_argument('--limit-card', type=str, default=None, help='Optional datacard path for --limit-only (default: <tag>/fit/output_card.yaml)')
    parser.add_argument('--skip-limit', dest='skip_limit', action='store_true', help='Stop after the fit stage, do not run the CLs limit')
    parser.add_argument('--toys', type=int, default=None, help='Override the number of toys in the registry "limit" section')
    parser.add_argument('--freeze-nuisances', dest='freeze_nuisances', action='store_true', help='Freeze nuisance parameters in the CLs limit stage')
    parser.add_argument('--loc', type=str, default='tape', help='File location: disk, tape or local')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel workers per dataset')
    parser.add_argument('--fitrange_low', type=float, nargs='+', default=[100, 475], help='Minimum ordered mom, time')
    parser.add_argument('--fitrange_hi', type=float, nargs='+', default=[110, 1650], help='Maximum ordered mom, time')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity passed to process.py')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', help='Print the commands without running them')
    args = parser.parse_args()

    sys.exit(main(args))
