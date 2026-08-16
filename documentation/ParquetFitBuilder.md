# Parquet Fit Builder

## Overview

The **ParquetFitBuilder** provides a streamlined workflow for fitting physics distributions stored in pre-processed parquet files. Unlike the standard analysis pipeline that processes raw ROOT Ntuples through analysis cuts, the parquet fit builder:

- **Skips ROOT file processing** and directly loads pre-prepared parquet data (post-analysis-cuts)
- **Performs unbinned maximum likelihood fits** in 1D (momentum only) or 2D (momentum × time) spaces
- **Scales physics components** using weight injection to match expected yields from an input YAML configuration card
- **Exports fit results** as `.npz` snapshots containing PDF parameters and systematic uncertainties
- **Generates output datacards** suitable for limit-setting analyses

This approach is particularly useful when:
- Working with pre-cut datasets that skip the expensive ROOT processing stage
- Rapid prototyping and parameter studies
- Testing model variations without reprocessing entire datasets

## Data Flow

```
Parquet Files (Pre-cut Data)
        ↓
ParquetFitBuilder (1D/2D ML Fit)
        ↓
Weight Injection (Component Scaling)
        ↓
.npz Fit Snapshot (Parameters + Systematics)
        ↓
YAML Datacard Generation
        ↓
Downstream Analysis (Limits, Plots)
```

## Input Data Format

### Parquet Data Files

The parquet files must contain pre-selected events with the following columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `recomom_ttfront` or `mom` | float | Reconstructed momentum (MeV/c) |
| `time` | float | Reconstructed time (ns) |
| `mc_process_code` | int | MC truth process identifier |
| `weight` | float (optional) | Event weight for reweighting |

**Example parquet structure:**
```python
import pandas as pd
df = pd.read_parquet("data.parquet")
print(df.columns)  # Should include 'mom', 'time', 'mc_process_code'
```

### Input Card Format (YAML)

The input configuration card specifies expected yields, model parameters, and analysis setup:

```yaml
observables:
  mom: [95.0, 115.0]      # Momentum fit range (MeV/c)
  time: [475.0, 1650.0]   # Time fit range (ns)

expected_yields:
  CE: 0.57                # Expected signal events
  DIO: 2500.0             # Expected DIO background
  Cosmic: 150.0           # Expected cosmic background
  RPC: 75.0               # Expected RPC background

components:
  CE:
    mom_pdf: "ce_spectrum"
    time_pdf: "exponential"
    tau_lifetime: 864.0   # muonic aluminum lifetime (ns)
  
  DIO:
    mom_pdf: "dio_poly"
    time_pdf: "exponential"
    tau_lifetime: 864.0
  
  Cosmic:
    mom_pdf: "uniform"
    time_pdf: "uniform"
    mom_range: [95.0, 115.0]
    time_range: [640.0, 1650.0]
  
  RPC:
    mom_pdf: "gaussian"
    time_pdf: "exponential"
    mom_mean: 104.5
    mom_sigma: 2.0
    tau_lifetime: 26.0    # free pion lifetime

systematics:
  lnN:
    DIO_shape: 1.10       # 10% log-normal uncertainty
    RPC_rate: 1.15        # 15% rate uncertainty
```

## Core Functionality

### 1D Fit (Momentum Only)

```python
from archive.parquet_fit_builder import ParquetFitBuilder

builder = ParquetFitBuilder(
    parquet_files={
        'dio': 'path/to/dio.parquet',
        'cosmic': 'path/to/cosmic.parquet',
        'rpc': 'path/to/rpc.parquet',
    },
    input_card='input_card.yaml',
    fit_type='1d'
)

fit_result = builder.run_fit()
builder.export_npz('output_fit.npz')
```

### 2D Fit (Momentum × Time)

```python
builder = ParquetFitBuilder(
    parquet_files={...},
    input_card='input_card.yaml',
    fit_type='2d',
    momentum_range=(97, 110),
    time_range=(475, 1650)
)

fit_result = builder.run_fit()
```

## Weight Injection and Component Scaling

The parquet fit builder uses weight injection to match expected yields from the input card:

1. **Load component data** from individual parquet files
2. **Calculate scale factors** to match target yields:
   $$\text{scale}_j = \frac{N_{\text{expected},j}}{N_{\text{observed},j}}$$
3. **Apply weights** to each component in the likelihood
4. **Fit scale parameters** and PDF parameters jointly

This ensures that final component normalizations match physics expectations while allowing shape parameters to float.

## Output: .npz Snapshots

The exported `.npz` file contains:

| Key | Type | Description |
| :--- | :--- | :--- |
| `fit_result` | dict | Complete zfit minimizer result |
| `params_dict` | dict | Final parameter values and uncertainties |
| `yields_dict` | dict | Fitted component yields |
| `correlations` | array | Correlation matrix of parameters |
| `fit_quality` | dict | NLL value, degrees of freedom, p-value |
| `observable_ranges` | dict | Momentum and time fit ranges |

**Accessing results:**
```python
import numpy as np

snapshot = np.load('output_fit.npz', allow_pickle=True)
fit_result = snapshot['fit_result'].item()
params = snapshot['params_dict'].item()
yields = snapshot['yields_dict'].item()

print(f"DIO yield: {yields['DIO']:.2f}")
print(f"CE signal strength: {params['mu'].value:.4f}")
```

## Datacard Generation

Convert a fit snapshot to a YAML datacard for downstream analyses:

```python
from generate_datacard import datacard_from_snapshot

datacard = datacard_from_snapshot(
    fit_snapshot='output_fit.npz',
    output_path='generated_card.yaml',
    include_systematics=True
)
```

This generates a datacard compatible with the Combine-like sensitivity limit setter.

## Usage Example: Complete Workflow

```bash
# Step 1: Run parquet fit (1D momentum)
python archive/parquet_fit_builder.py \
    --fit-type 1d \
    --components \
        dio=file_lists/DIOtail95_MDC2025an_best_nomix.txt \
        cosmic=file_lists/Cosmics_MDC2025an_nomix.txt \
        rpc=file_lists/ExtRPC_MDC2025an_nomix.txt \
    --variable recomom_ttfront \
    --fit-range-lo 97 \
    --fit-range-hi 110 \
    --input-card input_card.yaml \
    --jobs 16 \
    --export-npz parquet_1d_fit.npz

# Step 2: Generate datacard
python generate_datacard.py \
    --snapshot parquet_1d_fit.npz \
    --output fit_card.yaml

# Step 3: Run sensitivity scan
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_card.yaml \
    --dim 1d \
    --toys 2000
```

## Performance Considerations

| Factor | Impact | Recommendation |
| :--- | :--- | :--- |
| **Number of events** | Higher counts improve fit stability | ≥ 1000 events per component recommended |
| **Fit dimensionality** | 2D fits are more computationally expensive | Use 1D for rapid studies, 2D for final results |
| **Number of PDF parameters** | More parameters require better statistics | Regularize with Bayesian penalties if needed |
| **GPU availability** | zfit can utilize GPU for speedup | Use `TF_FORCE_GPU_ALLOW_GROWTH=true` |
| **Parallelization** | Multiple jobs speed up numerical fits | `--jobs N` with N = number of CPU cores |

## Troubleshooting

### Fit fails to converge
- Increase number of events per component
- Simplify PDF model (use fewer parameters)
- Adjust fit range to avoid edge effects

### Yields don't match input card
- Check that weight injection is enabled
- Verify that all expected components have data
- Inspect scale factor values for anomalies

### Memory issues with large datasets
- Reduce dataset size or use binned fits
- Enable memory-mapped parquet reading
- Process components separately and combine results

## See Also

- [Likelihood Definition](LikelihoodDefinition.md) — Mathematical formulation of extended likelihood
- [Combine-like Sensitivity Limits](CombineLikeSensitivityLimits.md) — Using fit output for limit setting
- [Weighted Fits and Limits](WeightedFitsAndLimits.md) — Broader framework documentation
