# Running the analysis chain

`run_chain.py` orchestrates the selection, parquet fit, and CLs limit stages for a tagged analysis run. The dataset registry is [../datasets.yaml](../datasets.yaml); generated files are written below the tag directory.

## Quick start

```bash
python run_chain.py --tag v80 --group mix --loc tape --jobs 16
```

The registry controls which datasets are available, their process names, initial acceptances, expected yields, fit type, and limit settings.

## Dataset registry

Each dataset entry has this form:

```yaml
datasets:
  CeM_mix:
    filelist: MDC2025au/CeM_mix.txt
    process: CE
    acceptance: 0.4141
    expected: 3.397527477

fit:
  type: 2d

limit:
  toys: 1000
  seed: 42
  cl: 0.90
```

Use all registry datasets, a named group, or selected registry names:

```bash
python run_chain.py --tag v80
python run_chain.py --tag v80 --group mix
python run_chain.py --tag v80 --datasets CeM_mix DIO_mix
```

`--registry path/to/other.yaml` selects another registry. `--fit-type 1d`, `--fit-type 2d`, or `--fit-type both` overrides the registry fit type.

## Stages

### Stage 1: selection

For every selected dataset, the chain invokes `process.py --nofit`. It writes:

```text
<tag>/cutflows/<dataset>.csv
<tag>/parquet/<dataset>_postcut.parquet
<tag>/logs/<dataset>.log
```

The chain then writes `<tag>/manifest.yaml`. It records dataset provenance, process, configured acceptance and expected yield, final cut efficiency, net acceptance, and output paths.

For ordinary datasets:

```text
net_acceptance = acceptance * cut_efficiency
```

For datasets whose process is `Cosmic`, `expected` is taken from the final cut-flow row's `events_passing` value. The registry expected value is ignored for cosmic datasets. That cut-flow yield is then multiplied by `net_acceptance` when the input card is generated.

### Stage 2: parquet fit

The chain generates:

```text
<tag>/input_card.yaml
```

The card contains the observable ranges, systematics, and process yields. For non-cosmic datasets, the process yield is accumulated as:

```text
expected * net_acceptance
```

Cosmic yields are accumulated as the final cut-flow yield multiplied by `net_acceptance`. Datasets sharing a process are merged into one parquet file before fitting.

The fit runs with `archive/parquet_fit_builder.py` and writes:

```text
<tag>/fit/<tag>_snapshot.npz
<tag>/fit/output_card.yaml
<tag>/fit/plots/
<tag>/logs/fit.log
```

The output card is generated from the fit snapshot and contains the fitted values used by the limit stage.

### Stage 3: CLs limit

The chain runs `sensitivity_scan/simple_limit_combine_12d.py` using `<tag>/fit/output_card.yaml`. Results are written to:

```text
<tag>/limits/
<tag>/logs/limit.log
```

The dimension is `1d` for a `1d` fit and `2d` for `2d` or `both`.

## Reusing existing selection outputs

If parquet and cut-flow files already exist, skip Stage 1:

```bash
python run_chain.py --tag v79.2 --skip-selection
```

In this mode the existing `<tag>/manifest.yaml` supplies dataset names, process assignments, and parquet/cut-flow paths. The chain does not reread registry values unless requested.

To update configured `acceptance`, `expected`, and `process` values from the registry while preserving existing parquet and cut-flow files:

```bash
python run_chain.py \
  --tag v79.2 \
  --skip-selection \
  --refresh-registry
```

Cosmic expected yields are reread from the existing cosmic cut-flow rather than from the registry.

To update only the existing manifest, with no selection, parquet merging, fit, or limit work, run:

```bash
python run_chain.py --tag v79.2 --manifest-only
```

This rereads each dataset's final cut-flow row and refreshes `cut_efficiency`,
`net_acceptance`, and `scaled_expected_yield`. It also reloads `acceptance`,
`expected`, and `process` from the registry.

To run only the limit stage from an existing output card, with no selection or fit,
run:

```bash
python run_chain.py --tag v79.2 --limit-only
```

By default this uses `<tag>/fit/output_card.yaml`. To point at another card:

```bash
python run_chain.py --tag v79.2 --limit-only --limit-card /path/to/output_card.yaml
```

## Useful controls

```text
--skip-selection       Reuse the existing manifest and start at Stage 2
--refresh-registry     With --skip-selection, update manifest values from registry
--manifest-only        Refresh an existing manifest from cutflows and registry, then stop
--limit-only           Run only the CLs limit from an existing output card
--limit-card PATH      Datacard path for --limit-only (default: <tag>/fit/output_card.yaml)
--skip-fit             Stop after Stage 1 and input-card generation
--skip-limit           Stop after Stage 2
--dry-run              Print commands without executing subprocess stages
--toys N               Override the registry limit toy count
```

`--skip-selection` requires `<tag>/manifest.yaml`. A failed selection prevents later stages from running. Logs are retained per dataset and for the fit and limit stages.
