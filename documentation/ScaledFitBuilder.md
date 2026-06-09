# Scaled Sample Fitting Guide

## Overview

The **ScaledFitBuilder** enables fitting to scaled MC component samples using the highest-stats approach. Instead of combining all MC data into a single dataset, you can:

1. Load each MC component independently (e.g., DIO, cosmic, RPC, CE separately)
2. Apply cuts to each component consistently
3. Scale each component by its expected physics yield
4. Combine the scaled samples
5. Execute fits using the existing fitting infrastructure

This approach is useful when:
- You want to balance MC statistics across components
- Each component has different available statistics
- You need to test sensitivity to component scaling
- You want to maintain reproducibility of component contributions

## Architecture

### Data Flow

```
Raw ROOT Files (separate by component)
        ↓
Apply Cuts (via AnaProcessor)
        ↓
Extract Fit Variable (momentum, time, etc.)
        ↓
Scale by Expected Yields
        ↓
Combine Scaled Samples
        ↓
Create Component Category Array
        ↓
Unbinned Fit (existing infrastructure)
```

### Key Classes

**ScaledFitBuilder**: Main class that orchestrates the pipeline
- Loads and processes component files
- Applies scaling
- Combines data
- Executes fits

## Usage

### Method 1: Python API (Recommended for workflows)

```python
from scaled_fit_builder import ScaledFitBuilder

# Initialize builder
builder = ScaledFitBuilder(verbosity=2, jobs=4)

# Define component files
component_files = {
    'dio': 'file_lists/MDS3c_1e-13.txt',
    'cosmic': 'file_lists/CM_Cosmic.txt',
    'rpc_ext': 'file_lists/ExtRPC_MDC2025an_nomix.txt',
    'rpc_int': 'file_lists/IntRPC_MDC2025an_nomix.txt',
    'ce': 'file_lists/signal_ce.txt',
}

# Load and scale using physics default yields
builder.load_and_scale_components(
    component_files,
    sign='minus',
    location='disk'
)

# Run 1D fit
result = builder.fit_mom_1d(
    variable='recomom_ttfront',
    fit_range=(95, 110),
    plot=True
)

# Or run 2D fit
result = builder.fit_mom_time_2d(
    mom_variable='recomom_ttfront',
    time_variable='trkfit.trksegpars_lh.t0',
    fit_range_mom=(95, 110),
    fit_range_time=(475, 1650),
    plot=True
)
```

### Method 2: Command-Line Interface

```bash
# 1D momentum fit
python scaled_fit_builder.py \
    --fit-type 1d \
    --components dio=file_lists/MDS3c_1e-13.txt \
                 cosmic=file_lists/CM_Cosmic.txt \
                 ce=file_lists/signal_ce.txt \
    --variable recomom_ttfront \
    --fit-range 95 110 \
    --jobs 4

# 2D momentum + time fit
python scaled_fit_builder.py \
    --fit-type 2d \
    --components dio=file_lists/MDS3c_1e-13.txt \
                 cosmic=file_lists/CM_Cosmic.txt \
    --variable recomom_ttfront \
    --time-variable trkfit.trksegpars_lh.t0 \
    --fit-range 95 110 \
    --time-range 475 1650 \
    --constraints-dir uncertainties/outputs
```

### Method 3: Examples

Run provided example scripts:

```bash
# Basic 1D fit
python fit_scaled_examples.py --example basic

# Custom yields
python fit_scaled_examples.py --example custom_yields

# 2D fit
python fit_scaled_examples.py --example 2d

# MC truth variable
python fit_scaled_examples.py --example mc_truth

# All examples
python fit_scaled_examples.py --example all
```

## Configuration

### Component Yields

By default, physics-based expected yields are used:

| Component | Default Yield | Purpose |
|-----------|---------------|---------|
| DIO | 5.87e3 | Radiative DIO above 95 MeV |
| Cosmic | 500.5 | Cosmic-induced tracks |
| RPC External | 1.18 | External RPC backgrounds |
| RPC Internal | 1.49 | Internal RPC backgrounds |
| RMC External | Auto | External rare-process backgrounds |
| RMC Internal | Auto | Internal rare-process backgrounds |
| IPA | Auto | Inclasive pion absorption |
| CE | 65 | Signal (Charged Lepton Exchange) |

### Custom Yields

Override defaults:

```python
builder = ScaledFitBuilder(verbosity=1)
builder.set_component_yields({
    'dio': 6000,      # Custom value
    'cosmic': 450,
    'ce': 70,
    'rpc_ext': None,  # Auto-scale (use max of other components)
})
builder.load_and_scale_components(component_files)
```

Or command-line:

```bash
python scaled_fit_builder.py \
    --components ... \
    --yields dio=6000 cosmic=450 ce=70 \
    ...
```

### Scaling Modes

**1. Physics-Based (Default)**
```python
builder.load_and_scale_components(
    component_files,
    normalize_to_max=False  # Use component_yields
)
```

**2. Normalize to Max**
```python
builder.load_and_scale_components(
    component_files,
    normalize_to_max=True  # All scale to largest component
)
```

## Fit Variables

### Supported Standard Variables

Any field accessible in the processed data structure:

```python
# Momentum variables
'recomom_ttfront'           # Reconstructed momentum at TT_Front
'recomom_mc_ttfront'        # MC true momentum at TT_Front
'trkfit.trksegpars_lh.p'    # Total momentum (LH hypothesis)
'trk.pt'                    # Transverse momentum

# Time variables
'trkfit.trksegpars_lh.t0'   # Time-of-arrival
'trkfit.t0'                 # Alternative time field

# Other variables
'trk.nactive'               # Active tracker planes
'trk.ndof'                  # Degrees of freedom
```

### Special Variables

The builder includes preprocessing for:

- **`recomom_ttfront`**: Extracts momentum at tracker front surface
  - Automatically selects TT_Front surface
  - Handles segment masking
  - Returns magnitude

- **`recomom_mc_ttfront`**: MC truth momentum at tracker front
  - Uses MC segment information (trksegsmc)
  - Extracts from MC truth branches

## Cut Configuration

ScaledFitBuilder applies standard cuts to all components (same as plot_scaled_mom.py):

| Index | Cut | Active |
|-------|-----|--------|
| 0 | is_reco_electron | ✓ |
| 1 | has_downstream | ✓ |
| 2 | has_trk_front | ✓ |
| 3 | good_trkqpid | ✓ |
| 4 | good_trkqual | ✓ |
| 5 | within_t0err | ✓ |
| 6 | has_hits | ✓ |
| 7 | within_lhr_maxl | ✗ |
| 8 | within_d0 | ✗ |
| 9 | within_pitch_angle | ✗ |
| 10 | has_st | ✓ |
| 11 | no_opa | ✓ |
| 12 | no_crv_veto | ✓ |
| 13 | no_crv_quality | ✓ |
| 14 | no_crv_timewindow | ✓ |
| 15 | pz/pt | ✓ |
| 16 | triggers | ✓ |
| 17 | in_mom_range | ✓ |
| 18 | within_t0_early | ✗ |
| 19 | no_reflected | ✗ |
| 20 | within_t0 | ✓ |
| 21 | signal_region | ✗ |

To customize cuts, create a subclass or modify the `_process_file_list` method.

## Integration with Existing Infrastructure

### Reused Components

The ScaledFitBuilder is designed to maximize reuse:

| Component | Module | Reused |
|-----------|--------|--------|
| Fit engine | `fit_module.py` | ✓ Full |
| PDF builders | `momentum_pdf_builder.py` | ✓ Full |
| Plotting | `plot_module.py` | ✓ Full |
| Data preparation | `data_prep.py` | ✓ Full |
| Physics models | `model/physics_components.py` | ✓ Full |
| Cuts & selection | `pyutils/` | ✓ Full |
| File processing | `process.py` (AnaProcessor) | ✓ Full |

### No Changes Needed

The following modules are used unchanged:
- `fit_module.Unbinned_fit_mom`
- `fit_module.Unbinned_2d_fit_mom_time`
- All plotting and PDF infrastructure
- All constraint/uncertainty handling

## Advanced Usage

### Using Real Data for Comparison

```python
builder = ScaledFitBuilder()
builder.load_and_scale_components(component_files)

# Load real data (optional)
builder.load_real_data(
    'file_lists/data.txt',
    sign='minus',
    location='local'
)

# Data can be overlaid on plots if plot infrastructure is extended
result = builder.fit_mom_1d(...)
```

### Different File Locations

```python
# From disk (default)
builder.load_and_scale_components(
    component_files,
    location='disk'
)

# From local filesystem
builder.load_and_scale_components(
    component_files,
    location='local'
)
```

### Multi-Job Processing

```python
builder = ScaledFitBuilder(jobs=8)  # 8 parallel jobs
builder.load_and_scale_components(component_files)
```

### Custom Constraints

```python
result = builder.fit_mom_1d(
    variable='recomom_ttfront',
    fit_range=(95, 110),
    constraints_dir='uncertainties/outputs'  # Custom constraint location
)
```

## Output and Results

### Fit Results

The fit functions return result objects containing:
- Fitted parameter values and uncertainties
- Likelihood information
- Covariance matrices (if available)
- Generated plots (ROOT files and PDFs)

### Plot Output

Generated plots include:
- Fit quality plots showing data and model predictions
- Component contributions
- Residual plots (if included in plotting module)
- Correlation matrices
- Parameter error summaries

## Performance Considerations

### Memory Usage

- Each component is loaded separately to enable parallel processing
- Scaling creates duplicated events (event resampling)
- Total memory: ~(N_events_per_component × N_components × 8 bytes) × scale_factors

### Processing Time

- Component loading: Depends on file I/O and size
- Cut application: ~1-2 seconds per component (with 1 job)
- Scaling: Negligible
- Fitting: Depends on total events and model complexity

### Optimization Tips

```python
# Use parallel jobs for faster loading
builder = ScaledFitBuilder(jobs=min(N_components, available_cores))

# Process fewer components
component_files = {
    'dio': 'file_lists/MDS3c_1e-13.txt',
    'cosmic': 'file_lists/CM_Cosmic.txt',
    # Skip less significant components for quick tests
}
```

## Troubleshooting

### Common Issues

**Issue: "No scaled components loaded"**
- Ensure component files exist and are readable
- Check file paths are relative to current directory
- Verify AnaProcessor can process the files

**Issue: Empty variable array**
- Check variable name spelling
- Verify the variable exists in processed data
- Try simpler variables like 'trk.pt' for debugging

**Issue: Fit doesn't converge**
- Check fit range is appropriate for variable
- Ensure you have enough statistics
- Try reducing number of components for debugging
- Check constraints are reasonable

**Issue: Slow processing**
- Check available disk I/O bandwidth
- Consider using local files instead of remote
- Reduce jobs if memory-bound
- Process fewer components first

### Debug Mode

Enable verbose logging:

```python
builder = ScaledFitBuilder(verbosity=3)  # Maximum verbosity
builder.load_and_scale_components(...)
```

## Examples and References

### Complete Example: Physics Study

```python
"""Study signal efficiency variation with scaled MC."""
from scaled_fit_builder import ScaledFitBuilder

# Component files - highest stats available
components = {
    'dio': 'file_lists/DIO_highstats.txt',
    'cosmic': 'file_lists/Cosmic_highstats.txt',
    'rpc': 'file_lists/RPC_combined.txt',
}

# Try different yield assumptions
for dio_yield in [5000, 5500, 6000]:
    builder = ScaledFitBuilder(verbosity=1)
    builder.set_component_yields({
        'dio': dio_yield,
        'cosmic': 500,
        'rpc': 2,
    })
    
    builder.load_and_scale_components(components)
    
    result = builder.fit_mom_1d(
        variable='recomom_ttfront',
        fit_range=(95, 110),
        plot=True
    )
    
    # Analyze result
    print(f"DIO yield {dio_yield}: N_CE = {result.params['N_ce'].value()}")
```

### Complete Example: Comparison Study

```python
"""Compare single combined fit vs. highest-stats scaled fit."""
from fit_module import Unbinned_fit_mom
from scaled_fit_builder import ScaledFitBuilder

# Single combined fit (current approach)
# ... [load combined data]
result_combined = Unbinned_fit_mom(combined_data, ...)

# Highest-stats scaled fit
builder = ScaledFitBuilder()
components = {
    'dio': 'file_lists/MDS3c_1e-13.txt',
    'cosmic': 'file_lists/CM_Cosmic.txt',
    # ... other components
}
builder.load_and_scale_components(components)
result_scaled = builder.fit_mom_1d(...)

# Compare results
print("Combined fit N_CE:", result_combined.params['N_ce'].value())
print("Scaled fit N_CE:", result_scaled.params['N_ce'].value())
```

## See Also

- [Existing Fitting Infrastructure](../documentation/fitmodule_py.md)
- [PDF Builder Reference](../documentation/momPDF_module_py.md)
- [Physics Components Model](../documentation/code-specifics/physics_components.md)
- [Uncertainty Analysis](../documentation/UncertaintyAnalysis.md)
- [Getting Started Guide](../documentation/GettingStarted.md)
