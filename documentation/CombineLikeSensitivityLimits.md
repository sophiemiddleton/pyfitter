# Combine-like Sensitivity Limits

## Overview

The **Combine-like Sensitivity Limit Setter** (`simple_limit_combine_12d.py`) implements a frequentist statistical framework for computing upper limits on signal production. It mimics the behavior of the ROOT-based Combine tool but operates entirely in Python using `zfit`, making it:

- **Framework-independent**: No dependence on ROOT or RooStats
- **Dimension-flexible**: Supports 1D (momentum only) or 2D (momentum × time) analyses
- **Systematic-aware**: Handles log-normal constrained nuisance parameters matching Combine's `lnN` format
- **Asymptotic**: Uses exact Cowan et al. (2011) frequentist CLs formulae for efficiency

## Statistical Framework

### Extended Likelihood with Systematics

The analysis employs an extended likelihood with log-normal systematic constraints:

$$\mathcal{L}(\mu, \boldsymbol{\theta}) = \text{Poisson}(N_{\text{obs}} | N_{\text{exp}}(\mu, \boldsymbol{\theta})) \cdot \prod_i^{N_{\text{obs}}} \frac{\sum_j w_j(\boldsymbol{\theta}) f_j(\mathbf{x}_i; \boldsymbol{\theta})}{\sum_j w_j(\boldsymbol{\theta})} \cdot \prod_k \text{Constraint}_k(\theta_k)$$

where:
- $\mu$ = signal strength parameter (POI: Parameter of Interest)
- $\boldsymbol{\theta}$ = nuisance parameters (PDF shapes, scale factors)
- $w_j(\boldsymbol{\theta})$ = process weight as function of nuisance parameters
- $f_j(\mathbf{x}_i; \boldsymbol{\theta})$ = PDF for process $j$ at event $i$
- $\text{Constraint}_k$ = log-normal penalty for nuisance $\theta_k$

### Log-Normal Systematics

Systematic uncertainties are modeled using log-normal distributions. A single parameter $\theta_k$ with uncertainty factor $\kappa$ has the constraint:

$$\mathcal{C}_k(\theta_k) = \frac{1}{\theta_k \ln(\kappa) \sqrt{2\pi}} \exp\left(-\frac{\left[\ln(\theta_k)\right]^2}{2[\ln(\kappa)]^2}\right)$$

This is equivalent to a scale factor on the process yield: $s_k = e^{\theta_k}$, ranging from $1/\kappa$ to $\kappa$ with uncertainty factor $\kappa$.

### CLs Hypothesis Test

For testing a hypothesized signal strength $\mu$:

1. **Fit the data** with $\mu$ fixed to test value
2. **Compute test statistic**: $t_{\mu} = -2 \ln(\mathcal{L}_{\mu} / \mathcal{L}_{\text{best}})$
3. **Generate toys** under background-only ($\mu = 0$) and signal+background hypotheses
4. **Compute p-values**:
   - $p_{\mu}$ = fraction of $\mu$ toys with $t_{\mu}^{\text{toy}} \geq t_{\mu}^{\text{obs}}$
   - $p_b$ = fraction of background toys with $t_{\mu}^{\text{toy}} \geq t_{\mu}^{\text{obs}}$
5. **CLs upper limit** at 90% CL is the $\mu$ value where:
   $$\text{CL}_s(\mu) = \frac{p_{\mu}}{p_b} = 0.10$$

## Input: YAML Datacards

The sensitivity limit setter accepts YAML datacards (typically generated from fit snapshots):

```yaml
# Observables and fit ranges
observables:
  mom: [97.0, 110.0]        # Momentum (MeV/c)
  time: [475.0, 1650.0]     # Time (ns)

# Expected yields (nominal values)
expected_yields:
  CE: 0.57                  # Signal (target)
  DIO: 2612.3               # Decay in orbit
  Cosmic: 145.2             # Cosmic induced
  RPC: 82.1                 # Radiative pion capture

# Process definitions with fitted parameters
processes:
  CE:
    type: "signal"          # Signal process (scale with μ)
    pdf_type: "ce_spectrum"
    parameters:
      ce_norm: {value: 0.57, uncertainty: 0.05}
      lifetime: {value: 864.0, fixed: true}
  
  DIO:
    type: "background"
    pdf_type: "dio_poly"
    parameters:
      dio_coeff_1: {value: 1.23, uncertainty: 0.08}
      dio_coeff_2: {value: -0.45, uncertainty: 0.12}
      lifetime: {value: 864.0, fixed: true}
  
  Cosmic:
    type: "background"
    pdf_type: "uniform"
    parameters: {}
  
  RPC:
    type: "background"
    pdf_type: "gaussian"
    parameters:
      rpc_mean: {value: 104.5, uncertainty: 0.3}
      rpc_sigma: {value: 2.0, uncertainty: 0.15}

# Systematic uncertainties (log-normal format)
systematics:
  DIO_shape_uncertainty:
    type: "lnN"
    affects: ["DIO"]
    kappa: 1.10              # 10% uncertainty (factor of 1.10)
  
  Cosmic_theory:
    type: "lnN"
    affects: ["Cosmic"]
    kappa: 1.20              # 20% uncertainty
  
  RPC_efficiency:
    type: "lnN"
    affects: ["RPC"]
    kappa: 1.15              # 15% uncertainty
```

## Basic Usage

### 1D Sensitivity Scan (Momentum Only)

```bash
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_result.yaml \
    --dim 1d \
    --toys 2000 \
    --seed 42 \
    --output sensitivity_1d.json
```

### 2D Sensitivity Scan (Momentum × Time)

```bash
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_result.yaml \
    --dim 2d \
    --toys 5000 \
    --seed 42 \
    --output sensitivity_2d.json
```

### Key Command-Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--datacard` | required | Path to YAML datacard |
| `--dim` | `2d` | Fit dimensionality: `1d` or `2d` |
| `--toys` | 1000 | Number of toy datasets per μ value |
| `--mu-points` | 50 | Number of signal strength (μ) values to scan |
| `--mu-range` | [0, 10] | Range of μ values to scan |
| `--seed` | random | Random seed for reproducibility |
| `--output` | `limits.json` | Output file for results |
| `--jobs` | 1 | Parallel toy generation jobs |
| `--verbosity` | 1 | Logging verbosity level |

## Output: Sensitivity Results

The output JSON file contains the sensitivity curve with expected limits and confidence bands:

```json
{
  "mu_points": [0.0, 0.2, 0.4, ..., 10.0],
  "expected_limit": [0.85, 0.90, 1.10, ..., 8.5],
  "sigma_1_up": [1.15, 1.20, 1.40, ..., 10.2],
  "sigma_1_down": [0.65, 0.68, 0.85, ..., 6.8],
  "sigma_2_up": [1.45, 1.52, 1.75, ..., 12.0],
  "sigma_2_down": [0.50, 0.52, 0.65, ..., 5.2],
  "observed_limit": [0.92, 0.98, 1.15, ..., 8.8]
}
```

### Interpretation

| Quantity | Meaning |
| :--- | :--- |
| `expected_limit` | 90% CL limit assuming data is consistent with background |
| `sigma_1_up/down` | ±1σ band: 68% of background-only experiments yield limits within this range |
| `sigma_2_up/down` | ±2σ band: 95% of background-only experiments yield limits within this range |
| `observed_limit` | Actual 90% CL limit from real data (if available) |

## Advanced Features

### Asymptotic Approximation

For fast preliminary results, use the asymptotic approximation (no toys):

```python
from sensitivity_scan.simple_limit_combine_12d import asymptotic_cls_limit

mu_test = 1.0
observed_limit = asymptotic_cls_limit(
    datacard_path='fit_result.yaml',
    mu_test=mu_test,
    dim='2d'
)
```

This runs in seconds compared to minutes for toy-based scans.

### Custom Systematic Treatment

Modify systematic uncertainties programmatically:

```python
from datacard import DataCard

card = DataCard.from_yaml('fit_result.yaml')

# Change DIO uncertainty from 10% to 5%
card.systematics['DIO_shape_uncertainty']['kappa'] = 1.05

# Freeze a nuisance parameter
card.get_parameter('dio_coeff_1').fixed = True

# Re-run limit scan
card.to_yaml('modified_card.yaml')
```

### Parallelization

Use multiprocessing to speed up toy generation:

```bash
python sensitivity_scan/simple_limit_combine_12d.py \
    --datacard fit_result.yaml \
    --dim 2d \
    --toys 5000 \
    --jobs 8 \
    --output sensitivity_parallel.json
```

This distributes toy generation across 8 processes, typically 6-7× faster than serial execution.

## Visualization and Analysis

### Plot Sensitivity Curves

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('sensitivity_1d.json') as f:
    results = json.load(f)

mu_pts = results['mu_points']
expected = results['expected_limit']
s1_up = results['sigma_1_up']
s1_down = results['sigma_1_down']

# Plot with 1σ band
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(mu_pts, expected, 'b-', linewidth=2, label='Expected (±1σ)')
ax.fill_between(mu_pts, s1_down, s1_up, alpha=0.3, color='blue')
ax.axhline(y=1.0, color='r', linestyle='--', label='Observed Limit = 1')
ax.set_xlabel('Signal Strength (μ)')
ax.set_ylabel('90% CL Upper Limit (μ)')
ax.set_title('Expected Sensitivity: CE Search')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sensitivity_curve.pdf')
```

### Compare Multiple Scenarios

```python
import json
import matplotlib.pyplot as plt

scenarios = {
    '1D Momentum': 'sensitivity_1d.json',
    '2D Momentum×Time': 'sensitivity_2d.json',
    'With Systematics': 'sensitivity_with_syst.json',
    'Systematics Frozen': 'sensitivity_syst_frozen.json',
}

fig, ax = plt.subplots(figsize=(12, 7))

for label, filepath in scenarios.items():
    with open(filepath) as f:
        results = json.load(f)
    ax.plot(results['mu_points'], results['expected_limit'], 
            label=label, linewidth=2)

ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Signal Strength (μ)')
ax.set_ylabel('90% CL Upper Limit')
ax.set_title('Sensitivity Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparison.pdf')
```

## Comparison with ROOT Combine

The Python implementation reproduces ROOT Combine results:

| Feature | Python `simple_limit_combine_12d.py` | ROOT Combine |
| :--- | :--- | :--- |
| **CLs Method** | Cowan et al. 2011 asymptotic | ✓ Equivalent |
| **Log-normal Systematics** | ✓ Full support | ✓ `lnN` format |
| **Nuisance Parameter Profiling** | ✓ Joint ML fit | ✓ Equivalent |
| **Toy Generation** | ✓ With systematic variations | ✓ Equivalent |
| **Dimensionality** | 1D, 2D | Any |
| **Speed** | Fast (seconds-minutes) | Moderate (minutes) |
| **Debugging** | Excellent (Python) | Limited (C++) |

## Troubleshooting

### Fit fails during toy generation
- **Cause**: Unstable PDFs or flat likelihoods
- **Solution**: Increase statistics, regularize with Bayesian penalties, or freeze nuisance parameters

### Limit curve is flat or unphysical
- **Cause**: Insufficient toy statistics or weak signal
- **Solution**: Increase `--toys`, check that signal PDF is sufficiently distinct from background

### Parallel jobs hang
- **Cause**: Shared zfit session state
- **Solution**: Use `--seed` to ensure independent random states, or reduce `--jobs`

### Datacard loading fails
- **Cause**: YAML format mismatch
- **Solution**: Validate with `datacard_validator.py`, check observable ranges and systematic keys

## See Also

- [Likelihood Definition](LikelihoodDefinition.md) — Mathematical foundation
- [Parquet Fit Builder](ParquetFitBuilder.md) — Generating datacards from fits
- [Weighted Fits and Limits](WeightedFitsAndLimits.md) — Broader framework
- ROOT Combine documentation: https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/
