# Weighted Fits and Limit Setting

## Overview

This document describes the **integrated workflow** for unbinned maximum likelihood fitting and sensitivity limit setting using the `py-fitter` framework. The analysis pipeline consists of three primary stages:

1. **Data Preparation** — Process ROOT Ntuples through analysis cuts
2. **Weighted Fitting** — Perform unbinned ML fits with component weight injection
3. **Limit Setting** — Compute CLs-based sensitivity limits for signal strength

The framework implements asymptotic frequentist hypothesis testing with proper treatment of systematic uncertainties in the spirit of ROOT Combine, but entirely in Python using `zfit`.

## Pipeline Architecture

```
Raw ROOT Data
     ↓
├─→ Control Region Analysis (data_prep.py)
│   └─→ Pre-cut Parquet Files
│
├─→ Standard Fit Path (fit_module.py)
│   └─→ Unbinned ML Fit
│   └─→ .npz Snapshot
│   └─→ Datacard
│
├─→ Rapid Parquet Path (ParquetFitBuilder)
│   └─→ Skip ROOT processing
│   └─→ Direct parquet → Fit → Datacard
│
└─→ Sensitivity Analysis (simple_limit_combine_12d.py)
    ├─→ Datacard Input
    ├─→ Toy Generation with Systematics
    └─→ CLs Limit Computation
```

## Stage 1: Data Preparation

### 1A. Standard ROOT Processing

Process raw EventNtuples through analysis cuts using `data_prep.py`:

```bash
python data_prep.py \
    --input raw_data.root \
    --output processed_data.parquet \
    --cuts control_region.py \
    --process-type dio
```

This outputs pre-cut parquet files ready for fitting.

### 1B. Control Region Studies

Use control regions to validate background models before performing the signal search:

```bash
python control_region.py \
    --datacard input_card.yaml \
    --region DIO_sideband \
    --output dio_control_plots.pdf
```

See [Control Regions](ControlRegions.md) for detailed methodology.

## Stage 2: Weighted Fitting

### Workflow

The weighted fit builder performs unbinned maximum likelihood fits with automatic component scaling:

```bash
python fit_module.py \
    --fit-type 2d \
    --components \
        dio=file_lists/DIOtail95_MDC2025an_best_nomix.txt \
        cosmic=file_lists/Cosmics_MDC2025an_nomix.txt \
        rpc=file_lists/ExtRPC_MDC2025an_nomix.txt \
    --variable recomom_ttfront \
    --fit-range-lo 97 \
    --fit-range-hi 110 \
    --time-range-lo 475 \
    --time-range-hi 1650 \
    --input-card input_card.yaml \
    --jobs 16 \
    --export-npz Run-1A-bkg.npz
```

#### Key Options

| Option | Description |
| :--- | :--- |
| `--fit-type` | `1d` (momentum only) or `2d` (momentum × time) |
| `--variable` | Observable name: `recomom_ttfront`, `mom`, etc. |
| `--fit-range-lo/hi` | Momentum range for fit (MeV/c) |
| `--time-range-lo/hi` | Time range for 2D fits (ns) |
| `--input-card` | YAML card with expected yields and systematics |
| `--export-npz` | Output file for fit snapshot |
| `--jobs` | Parallel jobs for numerical optimization |

### Weight Injection

The fitter automatically computes and applies weight factors to match expected yields:

$$w_j = \frac{N_{\text{expected},j}}{N_{\text{observed},j}}$$

This ensures final fitted yields match the input card's expected values while allowing PDF shape parameters to float.

### Output: Fit Snapshot (.npz)

The exported `.npz` file contains:
- **Fit parameters** with uncertainties (means, sigmas, shape coefficients)
- **Yields** for each component
- **Correlation matrix** of parameters
- **Fit quality metrics** (NLL, DoF, χ²)
- **Observable ranges** and other metadata

```python
import numpy as np

snapshot = np.load('Run-1A-bkg.npz', allow_pickle=True)
fit_result = snapshot['fit_result'].item()
yields = snapshot['yields_dict'].item()
print(f"DIO: {yields['DIO']:.1f} ± {yields['DIO_err']:.1f}")
```

## Stage 2B: Parquet-Only Fit Path (Alternative)

For rapid prototyping or when data is pre-cut, use `ParquetFitBuilder` to skip ROOT processing:

```bash
python archive/parquet_fit_builder.py \
    --fit-type 2d \
    --components \
        dio=data/dio_precut.parquet \
        cosmic=data/cosmic_precut.parquet \
        rpc=data/rpc_precut.parquet \
    --variable recomom_ttfront \
    --fit-range-lo 97 \
    --fit-range-hi 110 \
    --time-range-lo 475 \
    --time-range-hi 1650 \
    --input-card input_card.yaml \
    --export-npz parquet_fit.npz
```

See [Parquet Fit Builder](ParquetFitBuilder.md) for detailed documentation.

## Stage 3: Datacard Generation

Convert a fit snapshot to a YAML datacard for limit setting:

```bash
python generate_datacard.py \
    --snapshot Run-1A-bkg.npz \
    --output fit_result.card.yaml \
    --include-systematics true
```

The generated datacard contains:
- Observable ranges
- Fitted yields and parameters
- PDF model specifications
- Systematic uncertainty definitions (log-normal constraints)

See [Uncertainty Analysis](UncertaintyAnalysis.md) for details on systematic handling.

## Stage 4: Sensitivity Limit Setting

### Run CLs-based Sensitivity Scan

```bash
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_result.card.yaml \
    --dim 2d \
    --toys 5000 \
    --mu-range 0 10 \
    --mu-points 50 \
    --seed 42 \
    --output sensitivity_2d.json
```

#### Key Options

| Option | Description |
| :--- | :--- |
| `--datacard` | Input YAML datacard from fit stage |
| `--dim` | Dimensionality: `1d` or `2d` |
| `--toys` | Number of toy datasets per signal strength |
| `--mu-range` | Range of signal strength (μ) values to scan |
| `--mu-points` | Number of μ values to sample |
| `--seed` | Random seed for reproducibility |
| `--jobs` | Parallel toy generation processes |

### Interpret Results

The output JSON contains sensitivity curves with expected and observed limits:

```python
import json
import matplotlib.pyplot as plt

with open('sensitivity_2d.json') as f:
    results = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(results['mu_points'], results['expected_limit'], 'b-', linewidth=2)
plt.fill_between(results['mu_points'], 
                  results['sigma_1_down'], results['sigma_1_up'], 
                  alpha=0.3, label='±1σ')
plt.axhline(y=1.0, color='r', linestyle='--', label='Observed = 1')
plt.xlabel('Signal Strength (μ)')
plt.ylabel('90% CL Upper Limit')
plt.legend()
plt.savefig('sensitivity.pdf')
```

## Complete Workflow Example

### 1D Momentum-Only Analysis

```bash
# Fit momentum distribution
python fit_module.py \
    --fit-type 1d \
    --components \
        dio=file_lists/DIOtail95_MDC2025an_best_nomix.txt \
        cosmic=file_lists/Cosmics_MDC2025an_nomix.txt \
        rpc=file_lists/ExtRPC_MDC2025an_nomix.txt \
    --variable recomom_ttfront \
    --fit-range-lo 97 \
    --fit-range-hi 110 \
    --input-card input_card.yaml \
    --jobs 8 \
    --export-npz fit_1d.npz

# Generate datacard
python generate_datacard.py \
    --snapshot fit_1d.npz \
    --output fit_1d.card.yaml

# Compute sensitivity
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_1d.card.yaml \
    --dim 1d \
    --toys 3000 \
    --output sensitivity_1d.json

# Plot results
python -c "
import json, matplotlib.pyplot as plt
with open('sensitivity_1d.json') as f: r=json.load(f)
plt.plot(r['mu_points'], r['expected_limit'], 'b-', lw=2)
plt.axhline(1, color='r', ls='--')
plt.xlabel('μ'); plt.ylabel('90% CL Limit')
plt.savefig('sensitivity_1d.pdf'); print('✓ sensitivity_1d.pdf')
"
```

### 2D Momentum × Time Analysis

```bash
# Fit 2D distribution (requires more data)
python fit_module.py \
    --fit-type 2d \
    --components dio=file_lists/DIOtail95_MDC2025an_best_nomix.txt \
                  cosmic=file_lists/Cosmics_MDC2025an_nomix.txt \
                  rpc=file_lists/ExtRPC_MDC2025an_nomix.txt \
    --variable recomom_ttfront \
    --fit-range-lo 97 --fit-range-hi 110 \
    --time-range-lo 475 --time-range-hi 1650 \
    --input-card input_card.yaml \
    --jobs 16 \
    --export-npz fit_2d.npz

# Fast asymptotic limits (no toys)
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_2d.card.yaml \
    --dim 2d \
    --toys 0 \
    --output sensitivity_2d_asymptotic.json

# Full toy-based limits for final results
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_2d.card.yaml \
    --dim 2d \
    --toys 5000 \
    --jobs 8 \
    --output sensitivity_2d_final.json
```

## Systematic Uncertainties

Systematic uncertainties are treated as log-normal constrained nuisance parameters:

$$\text{Uncertainty Factor} = \kappa$$
$$\text{Yield Scale} = e^{\theta}$$

where $\theta \sim \mathcal{N}(0, [\ln(\kappa)]^2)$.

### Common Systematics

| Source | Uncertainty Factor | Type |
| :--- | :--- | :--- |
| **DIO Shape** | 1.05–1.15 (5–15%) | Shape variation |
| **RPC Cross-section** | 1.10–1.20 (10–20%) | Rate uncertainty |
| **Cosmic Flux** | 1.20–1.30 (20–30%) | Rate + seasonal variation |
| **Energy Scale** | 1.02–1.05 (2–5%) | Per-point correction |
| **Reconstruction Efficiency** | 1.03–1.08 (3–8%) | Calibration |

### Freeze Systematics for Comparisons

To study impact of individual systematics, freeze others:

```bash
# With all systematics
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_2d.card.yaml --dim 2d --toys 3000 \
    --output sensitivity_with_syst.json

# Freeze nuisance parameters (statistical only)
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_2d.card.yaml --dim 2d --toys 3000 \
    --freeze-nuisances true \
    --output sensitivity_stat_only.json
```

## Output Analysis and Interpretation

### Sensitivity Metrics

| Metric | Interpretation |
| :--- | :--- |
| **Expected Limit** | 90% CL limit assuming data matches background prediction |
| **±1σ Band** | Range containing 68% of background-only toy experiments |
| **±2σ Band** | Range containing 95% of background-only toy experiments |
| **Observed Limit** | Actual 90% CL limit from real data |

### Limit Setting Benchmarks

For the Mu2e CE search (typical):
- **1D (momentum only)**: Expected limit μ ≈ 1.0–1.5 at 90% CL
- **2D (momentum × time)**: Expected limit μ ≈ 0.7–1.0 at 90% CL (better sensitivity)
- **Impact of systematics**: ±20–30% change in expected limit

## Best Practices

1. **Always perform control region studies** before fitting the signal region
2. **Use 2D fits** when sufficient statistics available (improved sensitivity)
3. **Validate toy generation** with small toy counts first (quick validation)
4. **Document systematic assumptions** in input cards
5. **Compare 1D vs 2D** to quantify gain from time information
6. **Save intermediate results** (.npz snapshots, datacards, JSON outputs)
7. **Freeze nuisances** to understand systematic contributions

## Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **Fit fails to converge** | Reduce number of free parameters, increase statistics |
| **Yields don't match input card** | Check weight injection is enabled, verify data loading |
| **Asymptotic limit differs from toys** | Increase toy count, check that systematics are valid |
| **Toy generation hangs** | Reduce jobs, use fixed seed, check PDF bounds |
| **Memory issues** | Process components separately, use binned mode |

## Additional Resources

- [Likelihood Definition](LikelihoodDefinition.md) — Extended likelihood formalism
- [Parquet Fit Builder](ParquetFitBuilder.md) — Fast pre-cut data fitting
- [Combine-like Sensitivity Limits](CombineLikeSensitivityLimits.md) — CLs methodology and implementation
- [Uncertainty Analysis](UncertaintyAnalysis.md) — Systematic uncertainty treatment
- [Control Regions](ControlRegions.md) — Validation strategies
- ROOT Combine documentation: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/