# Uncertainty Analysis — Implementation Guide

The pyfitter framework includes a complete systematic uncertainty propagation pipeline. This document describes the two main workflows and how to use them.

## Overview

Systematic uncertainties are propagated through the fit using multiple methods:
- **Profile Likelihood**: Fix parameter to nominal ± 1σ, refit N_CE (publication-ready)
- **Systematic Variation**: Shift data by ±1σ, refit all parameters (diagnostic)
- **Constraint-based**: Gaussian constraints on nuisance parameters in the fit
- **Template-based**: Alternative PDF shapes for background models

All 17 systematically uncertain sources are catalogued in `uncertainties/model/sysunc_components.py` with specifications for each variation method.

## Quick Start

### Method 1: Profile Likelihood  ⭐ Recommended

Profile likelihood is the standard frequentist approach. For each systematic, we fix the parameter to nominal ± 1σ and refit N_CE to measure the impact.

**Run all implemented systematics:**

```bash
cd uncertainties
python run_profile_systematics.py --all-implemented --data ../MDS3c_mom_mag.npz --fittype 2D
```

**Run a single systematic (both +/- directions):**

```bash
python run_profile_systematics.py --systematic DIO_Theory --data ../MDS3c_mom_mag.npz --fittype 2D
```

**Collect results and generate impact summary:**

```bash
python run_profile_systematics.py --impact-summary
```

**Output**: `uncertainties/outputs/impact_table.json` with Δ N_CE for each systematic.

### Method 2: Systematic Variation (Diagnostic)

Run all implemented systematic variations by shifting the data and refitting:

```bash
python run_systematic_variation.py --all-implemented --data ../MDS3c_mom_mag.npz --fittype 2D
```

**Run a single systematic:**

```bash
python run_systematic_variation.py --systematic Abs_Mom_Scale --direction plus --data ../MDS3c_mom_mag.npz
```

**Output**: Varied data (NPZ) and fit results (JSON) for each systematic.

### Comparison: Profile Likelihood vs. Systematic Variation

| Aspect | Profile Likelihood | Systematic Variation |
|--------|-------------------|----------------------|
| **Best for** | Final results, publications | Diagnostics, understanding impact |
| **How it works** | Fix parameter at ±1σ, refit N_CE | Shift data by ±1σ, refit all parameters |
| **Parameter behavior** | Fixed (not varied) | Constrained (like in baseline fit) |
| **Background yields** | Constrained (stable) | Constrained (stable) |
| **Interpretation** | Δ N_CE from fixing nuisance | Δ N_CE from data modification |
| **Computation** | Faster: 2 fits/systematic | Slower: data variation + full pipeline |
| **Correlations** | Automatic (proper profile) | Via constraints on backgrounds |
| **Publication status** | ✓ Standard in HEP | ✓ Useful for exploration |

## Understanding Profile Likelihood

Profile likelihood is a standard frequentist method to measure systematic impacts:

**The procedure:**
1. **Fix the nuisance parameter** to nominal + 1σ (or - 1σ) using `treat_params='fix'` in physics_components
2. **Refit the signal yield N_CE** with other parameters floating (backgrounds constrained)
3. **Record the new N_CE value** and compare to baseline
4. **The difference is your systematic impact**: Δ N_CE = N_CE(fixed) - N_CE(nominal)

**Why this is powerful:**
- **Accounts for correlations** between the systematic parameter and N_CE automatically
- **Stable backgrounds** thanks to constraints on yields/efficiencies
- **Standard in HEP**: matches what journal editors and reviewers expect
- **Publication-ready**: single impact number per systematic
- **Frequentist**: no priors needed, straightforward interpretation

**Example: DIO Theory yield uncertainty**
- Nominal: DIO theory predicts 6400 ± 450 events (7% uncertainty)
- Profile +1σ: Fix N_DIO = 6850, refit → N_CE decreases by 0.5 events
- Profile -1σ: Fix N_DIO = 5950, refit → N_CE increases by 0.6 events
- Impact: Δ N_CE = ±0.5-0.6 events from DIO uncertainty

## Systematic Uncertainties Catalog

The framework defines 17 systematic sources, organized by component:

### Momentum/Detector (5 systematics)
| Name | Type | Variation | Status |
|------|------|-----------|--------|
| `Abs_Mom_Scale` | Momentum scale | ±3% | Implemented |
| `Mom_Res_Scale` | Momentum resolution | ±10% | Planned |
| `Track_Efficiency` | Track reco efficiency | ±2% | Planned |
| `PID_Efficiency` | Particle ID efficiency | ±5% | Planned |
| `Timing_Cal` | Timing calibration | ±50 ps | Planned |

### Yield/Cross-section (5 systematics)
| Name | Type | Variation | Status |
|------|------|-----------|--------|
| `DIO_Theory` | Yield constraint | ±7% | Implemented |
| `RPC_Rate` | Yield constraint | ±50% | Implemented |
| `Pion_Rate` | Yield constraint | ±20% | Implemented |
| `CRV_Efficiency` | Veto efficiency | ±10% | Implemented |
| `Cosmic_Rate` | Cosmic background | ±30% | Planned |

### Background PDFs (5 systematics)
| Name | Type | Variation | Status |
|------|------|-----------|--------|
| `DIO_PDF_Variant` | Shape uncertainty | Alternative shape | Planned |
| `Cosmic_Generator` | Shape uncertainty | Alternative generator | Implemented |
| `c1_Cosmic` | Shape parameter | ±50% | Implemented |
| `c2_Cosmic` | Shape parameter | ±20% | Implemented |
| `RPC_PDF` | Shape uncertainty | Alternative shape | Planned |

### Physics/Theory (2 systematics)
| Name | Type | Variation | Status |
|------|------|-----------|--------|
| `CE_Signal_Norm` | Theory prediction | ±2% | Planned |
| `Pileup_Model` | Readout effects | ±5% | Planned |

## Implementation Architecture

### Profile Likelihood Runner

**File**: `uncertainties/run_profile_systematics.py`

Measures systematic impacts using proper profile likelihood (publication-ready):

```python
from uncertainties.run_profile_systematics import ProfileLikelihoodRunner

runner = ProfileLikelihoodRunner(
    data_file='../MDS3c_mom_mag.npz',
    fit_type='2D',
    output_dir='uncertainties/outputs'
)

# Run single systematic
runner.run_systematic('DIO_Theory')

# Run all implemented
runner.run_all_implemented()

# Generate impact table
impacts = runner.collect_impact_summary()
print(impacts)  # {systematic_name: {'plus': delta, 'minus': delta}, ...}
```

### Systematic Variation Runner

**File**: `uncertainties/run_systematic_variation.py`

Applies data variations and refits (diagnostic):

```bash
# Run all implemented
python run_systematic_variation.py --all-implemented --data ../MDS3c_mom_mag.npz

# Run momentum systematics only
python run_systematic_variation.py --all-momentum --data ../MDS3c_mom_mag.npz

# Run single systematic
python run_systematic_variation.py --systematic Abs_Mom_Scale --direction plus --data ../MDS3c_mom_mag.npz
```

### Constraint Generation

**File**: `uncertainties/generate_constraints.py`

Creates constraint JSON for background yields and efficiencies:

```bash
python generate_constraints.py --fittype 2D --data ../MDS3c_mom_mag.npz
```

Produces: `uncertainties/outputs/constraints_combined.json`

## Typical Workflow

```
Step 1: Generate baseline fit
  $ python process.py --data data.txt --location baseline

Step 2a: Profile each systematic (publication results)
  $ cd uncertainties
  $ python run_profile_systematics.py --all-implemented --data ../MDS3c_mom_mag.npz --fittype 2D
  $ python run_profile_systematics.py --impact-summary
  → impact_table.json

Step 2b (optional): Diagnostic variations
  $ python run_systematic_variation.py --all-implemented --data ../MDS3c_mom_mag.npz

Step 3: Analyze impacts
  $ python post_process.py --impact-table outputs/impact_table.json
  → Waterfall plot, ranking, total uncertainty
```

## Output Formats

### Profile Likelihood Results

**Per-systematic output** (`MDS3c__sys-{NAME}__profile-{DIR}.json`):

```json
{
  "systematic": "DIO_Theory",
  "direction": "plus",
  "param_value": 6850,
  "baseline_N_CE": 18.5,
  "profiled_N_CE": 18.0,
  "delta_N_CE": -0.5,
  "fit_quality": "good"
}
```

**Impact summary** (`impact_table.json`):

```json
{
  "DIO_Theory": {"plus": -0.5, "minus": 0.6},
  "RPC_Rate": {"plus": -0.1, "minus": 0.1},
  "Abs_Mom_Scale": {"plus": -0.3, "minus": 0.2},
  ...
}
```

### Systematic Variation Results

**Varied data** (`MDS3c__sys-{NAME}__shift-{DIR}.npz`):
- Modified momentum array with applied systematic

**Fit results** (`MDS3c__sys-{NAME}__fit-{DIR}.json`):
```json
{
  "systematic": "Abs_Mom_Scale",
  "direction": "plus",
  "baseline_N_CE": 18.5,
  "varied_N_CE": 18.2,
  "delta": -0.3,
  "parameters": {...}
}
```

## Metrics and Interpretation

**Δ N_CE (shift in signal yield):**
- Small (< 0.2): Systematic has minimal impact
- Medium (0.2-1.0): Moderate constraint effect
- Large (> 1.0): This systematic dominates fit sensitivity

**Fit quality:**
- Check if profiling/variation causes convergence issues
- Larger shifts may indicate correlations with background components

**Comparison table (all systematics):**
- Rank by |Δ N_CE| to identify which systematics matter most
- Total uncertainty (assuming independence): √(Σ (Δ N_CE)²)

## Common Workflows

### Generate Full Impact Table

```bash
cd uncertainties

# Profile all implemented systematics
python run_profile_systematics.py --all-implemented --data ../MDS3c_mom_mag.npz --fittype 2D

# Collect summary
python run_profile_systematics.py --impact-summary

# Review results
cat outputs/impact_table.json | python -m json.tool
```

### Compare Profile vs. Variation

```bash
# Profile method
python run_profile_systematics.py --systematic Abs_Mom_Scale --data ../MDS3c_mom_mag.npz

# Variation method
python run_systematic_variation.py --systematic Abs_Mom_Scale --data ../MDS3c_mom_mag.npz

# Compare Δ N_CE values to understand method differences
```

### Diagnostic: Single Systematic Detail

```bash
# Profile with verbose output
python run_profile_systematics.py --systematic DIO_Theory --data ../MDS3c_mom_mag.npz --verbose 2

# View fit results
ls -la outputs/MDS3c__sys-DIO_Theory__*.json
cat outputs/MDS3c__sys-DIO_Theory__profile-plus.json | python -m json.tool
```

## References

- `uncertainties/run_profile_systematics.py` — Profile likelihood implementation
- `uncertainties/run_systematic_variation.py` — Systematic variation runner
- `uncertainties/generate_constraints.py` — Constraint generation
- `uncertainties/model/sysunc_components.py` — 17 systematic specifications
- `uncertainties/QUICKSTART.md` — Quick reference

