# 🧪 Component Fit Modules (`component_fits/`)

The `component_fits/` directory contains **standalone unbinned maximum likelihood fitters** designed for **testing individual particle physics component shapes and conducting focused, component-specific studies** in the Mu2e analysis. These modules are separate from the main `fit_module.py` analysis chain and are used for:

- **Shape validation**: Testing individual PDF models against Monte Carlo or data
- **Control region studies**: Fitting background components in sideband regions
- **Systematic studies**: Investigating component parameter sensitivity
- **Parameter optimization**: Fine-tuning PDF parameters for specific analyses
- **Publication plots**: Generating high-quality diagnostic plots with fitted shapes overlaid

## 🏗️ Design Philosophy

Each component fitter:
- **Operates independently**: No dependency on the main likelihood framework
- **Single-component focus**: Fits one physics process in isolation
- **Extended likelihood**: Uses extended unbinned ML fits (event yield + shape parameters)
- **Publication-ready output**: Automatic Okabe-Ito color scheme and publication styling
- **Flexible I/O**: Accepts parquet, numpy arrays, or awkward arrays; exports PDF plots
- **Pull plots**: Includes residuals and pull distributions for goodness-of-fit assessment

## 📦 Modules

### 1. **`ce.py` — Conversion Electron (CE) Fitter**

**Purpose**: Fits the conversion electron signal-like distribution using a Double-Sided Crystal Ball (DSCB) PDF.

**Physics**: The CE signal in momentum space is characterized by a sharp edge at the kinematic endpoint (104.97 MeV), with asymmetric tails from detector resolution and radiative effects.

**PDF Model**: `zfit.pdf.DoubleCB`
- **Parameters**:
  | Parameter | Description | Typical Value | Floating? |
  | :--- | :--- | :--- | :--- |
  | `mu` | Peak position | 104.0 MeV/c | ✓ |
  | `sigma` | Core width | 0.35 MeV/c | ✓ |
  | `alphal` | Left tail threshold | 1.5 | ✓ |
  | `nl` | Left tail power | 3.0 | ✓ |
  | `alphar` | Right tail threshold | 1.5 | ✓ |
  | `nr` | Right tail power | 3.0 | ✓ |
  | `N_Flat` | Event yield | Data-dependent | ✓ |

**Main Class**: `ConversionElectronFitter`

**Key Methods**:
- `fit_CELL_momentum_dscb(data_list, start, end, opt, label, nbins, out_file)` — Performs DSCB fit on momentum data
  - Accepts a list of data arrays with corresponding labels
  - Produces 2-panel plot: (top) histogram + fitted curve; (bottom) pull distribution
  - Supports normalization to a target yield

**Usage Example**:
```python
from component_fits.ce import ConversionElectronFitter

fitter = ConversionElectronFitter()
ce_data = [numpy_array_1, numpy_array_2]
fitter.fit_CELL_momentum_dscb(
    data_list=ce_data,
    start=95.0,
    end=115.0,
    opt="default",
    label=["MC Signal 1", "MC Signal 2"],
    nbins=50,
    out_file="CE_dscb_fit.pdf"
)
```

---

### 2. **`rmc.py` — Radiative Muon Capture (RMC) Fitter**

**Purpose**: Fits the RMC background shape in momentum space using a custom `GammaPolyHybrid` PDF.

**Physics**: RMC produces a radiative muon capture background with a low-energy turn-on, polynomial bulk, and exponential decay tail.

**PDF Model**: `GammaPolyHybrid` (custom `zfit.pdf.ZPDF`)
$$\text{pdf}(x) \propto (x - x_0)^\alpha \cdot (120 - x)^\beta \cdot e^{-\lambda x}$$

**Parameters**:
| Parameter | Description | Typical Value | Range | Floating? |
| :--- | :--- | :--- | :--- | :--- |
| `x0` | Low threshold edge | 90.0 MeV/c | [70, 96.5] | ✓ |
| `alpha` | Low-edge turn-on power | 2.0 | [0.01, 10] | ✓ |
| `beta` | Bulk curvature power | 1.0 | — | ✗ (fixed) |
| `lam` | Tail exponential decay | 0.5 | [0.001, 10] | ✓ |
| `N_RMC` | Event yield | Data-dependent | — | ✓ |

**Main Class**: `RMC`

**Key Methods**:
- `fit_momentum(data_list, labels, pdf_func, out_file, normalize, target_yield)` — Performs extended unbinned ML fit
  - Supports custom PDF via `pdf_func` callback
  - Computes $\chi^2 / \text{ndf}$ and displays in legend
  - Handles normalization with optional `target_yield`

**Command-Line Interface**:
```bash
python component_fits/rmc.py \
    --parquet <path_to_parquet> \
    --label "Internal RMC" \
    --tag v1 \
    --outdir ./results \
    --normalize \
    --target-yield 100000
```

**Usage Example**:
```python
from component_fits.rmc import RMC, make_gamma_poly_pdf

rmc_fitter = RMC()
rmc_data = awkward_array  # or numpy array
fitted_yield = rmc_fitter.fit_momentum(
    data_list=[rmc_data],
    labels=["RMC MC Sample"],
    out_file="rmc_fit.pdf",
    normalize=True,
    target_yield=100000
)
print(f"Fitted RMC yield: {fitted_yield:.1f}")
```

---

### 3. **`dio.py` — Dielectric Interaction (DIO) Fitter**

**Purpose**: Fits the DIO background shape using a physics-motivated `poly58` PDF based on muon decay kinematics.

**Physics**: DIO is the dominant background in the Mu2e analysis. The spectrum follows from the energy-dependent muon decay rate in aluminum:

$$\text{pdf}(\delta) \propto \sum_{n=5}^{8} a_n \delta^n$$

where $\delta = M_\mu - p - \frac{p^2}{2M_{\text{Al}}}$ is the recoil energy.

**PDF Model**: `poly58` (custom `zfit.pdf.ZPDF`)

**Physics Constants** (Mu2e Standards 2025):
```python
E_MAX = 104.97          # MeV (CE endpoint)
ALPHA = 1.0 / 137.036   # Fine structure constant
M_E = 0.510998          # Electron mass [MeV]
M_MU = 105.194          # Muon mass [MeV]
m_Al = 25133.0          # Aluminum mass [MeV]
```

**Parameters**:
| Parameter | Description | Typical Value | Range |
| :--- | :--- | :--- | :--- |
| `a5` | 5th-order coefficient | 8.98e-17 | [1e-17, 1e-16] |
| `a6` | 6th-order coefficient | 1.17e-17 | [1e-18, 1e-16] |
| `a7` | 7th-order coefficient | -1.07e-19 | [-1e-18, -1e-19] |
| `a8` | 8th-order coefficient | 8.14e-20 | [1e-20, 1e-19] |
| `N_DIO` | Event yield | Data-dependent | — |

**Main Class**: `DIO`

**Key Methods**:
- `fit_momentum(data_list, labels, pdf_func, out_file, normalize, target_yield)` — Similar to RMC fitter
- `make_poly58_pdf(obs, name)` — Factory function for poly58 PDF instantiation

**Usage Example**:
```python
from component_fits.dio import DIO, make_poly58_pdf
import zfit

dio_fitter = DIO()
obs = zfit.Space("x", limits=(95.0, 110.0))
custom_pdf = make_poly58_pdf(obs, name="dio_v1")

dio_fitter.fit_momentum(
    data_list=[dio_data],
    labels=["DIO MC"],
    pdf_func=lambda obs: custom_pdf,
    out_file="dio_fit.pdf"
)
```

---

### 4. **`cosmics.py` — Cosmic Background Fitter**

**Purpose**: Fits the cosmic-ray background in **time-of-arrival** space using a Chebyshev polynomial.

**Physics**: Cosmic backgrounds in Mu2e are relatively flat in the time window and are best characterized by low-order polynomial shapes.

**PDF Model**: `zfit.pdf.Chebyshev`
- Parameterized by Chebyshev polynomial coefficients
- Default degree: 1 (linear)

**Parameters**:
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `c0, c1, ...` | Chebyshev coefficients | 0.0 |
| `deg` | Polynomial degree | 1 |
| `N_Cosmic` | Event yield | Data-dependent |

**Main Class**: `Cosmics`

**Fit Range**: 480 ns to 700 ns (time-of-arrival window)

**Key Methods**:
- `fit_time(data_list, labels, deg, normalize, out_file, target_yield)` — Fits time distribution with Chebyshev polynomial
  - `deg` controls polynomial complexity

**Usage Example**:
```python
from component_fits.cosmics import Cosmics

cosmic_fitter = Cosmics()
cosmic_fitter.fit_time(
    data_list=[cosmic_time_data],
    labels=["Cosmic MC"],
    deg=2,  # 2nd-order polynomial
    normalize=True,
    out_file="cosmic_fit.pdf"
)
```

---

### 5. **`rpc.py` — RPC Background Fitter**

**Purpose**: Fits the RPC (Radiative Pion Capture) background in **time-of-arrival** space using an exponential shape.

**Physics**: RPC backgrounds are characterized by an exponential decay in time, reflecting the muon lifetime convolved with detector acceptance.

**PDF Model**: `zfit.pdf.Exponential`
$$\text{pdf}(t) \propto e^{-\lambda t}$$

**Parameters**:
| Parameter | Description | Typical Value |
| :--- | :--- | :--- |
| `lam` | Exponential decay rate | 0.001 /ns |
| `N_RPC` | Event yield | Data-dependent |

**Main Class**: `RPC`

**Fit Range**: 475 ns to 550 ns (time-of-arrival window)

**Key Methods**:
- `fit_time(data_list, labels, out_file, normalize, target_yield)` — Fits time distribution with exponential

**Usage Example**:
```python
from component_fits.rpc import RPC

rpc_fitter = RPC()
rpc_fitter.fit_time(
    data_list=[rpc_time_data],
    labels=["RPC MC"],
    out_file="rpc_fit.pdf",
    normalize=True
)
```

---

## 🔄 Common Features Across All Modules

### Data Input Formats
All fitters accept:
- **Numpy arrays**: `numpy.ndarray`
- **Awkward arrays**: `ak.Array` (auto-flattened and cleaned)
- **Parquet files**: Via command-line interface

### Data Cleaning
Automatic handling of:
- NaN values: `ak.nan_to_none()` → `ak.drop_none()`
- Nested arrays: Flattened to 1D with `ak.flatten()`
- Observable range: Data automatically clipped to fit range

### Normalization
Two modes:
- **Relative**: `normalize=True` → rescale to 100,000 events (default)
- **Explicit**: `target_yield=N` → rescale to exactly N events (overrides `normalize`)

### Output Visualization
Each fitter produces a **2-panel publication-ready PDF plot**:

**Panel 1 (top, 3:1 height):**
- Data histogram with Poisson error bars (hollow markers)
- Fitted curve overlaid
- Legend with fit statistics ($N$, $\chi^2/\text{ndf}$)
- Okabe-Ito color scheme for accessibility

**Panel 2 (bottom, 1:1 height):**
- Pull distribution: $\text{pull} = (\text{data} - \text{fit}) / \sigma_\text{data}$
- Blue bars centered at zero
- ±3σ reference lines

### Styling
All plots use:
- **Font**: DejaVu Serif (or Times New Roman fallback)
- **DPI**: 150 (publication quality)
- **Color scheme**: Okabe-Ito palette (colorblind-friendly)
- **Mu2e watermark**: Added to top-left corner

---

## 📋 Workflow Examples

### Example 1: Quick CE Shape Validation
```python
import pandas as pd
from component_fits.ce import ConversionElectronFitter

# Load MC sample
df = pd.read_parquet("ce_mc.parquet")
ce_mom = df["momentum"].to_numpy()

fitter = ConversionElectronFitter()
fitter.fit_CELL_momentum_dscb(
    data_list=[ce_mom],
    start=95.0,
    end=115.0,
    opt="default",
    label=["CE Signal MC"],
    nbins=50,
    out_file="ce_shape_check.pdf"
)
```

### Example 2: Comparing Multiple Background Samples
```python
from component_fits.rmc import RMC

fitter = RMC()

# Load multiple RMC datasets
rmc_sample_1 = pd.read_parquet("rmc_v1.parquet")["momentum"].to_numpy()
rmc_sample_2 = pd.read_parquet("rmc_v2.parquet")["momentum"].to_numpy()

fitter.fit_momentum(
    data_list=[rmc_sample_1, rmc_sample_2],
    labels=["RMC v1", "RMC v2"],
    out_file="rmc_comparison.pdf",
    normalize=True
)
```

### Example 3: Control Region DIO Fit
```bash
# DIO control region study from parquet
python component_fits/dio.py \
    --parquet sideband_dio_cr.parquet \
    --label "DIO Sideband CR" \
    --tag cr_study \
    --outdir ./cr_results \
    --target-yield 50000
```

### Example 4: Time-Domain Cosmic Background
```python
from component_fits.cosmics import Cosmics

fitter = Cosmics()

# Fit cosmic time distribution with 2nd-order polynomial
cosmic_time = df["time_of_arrival"].to_numpy()

fitter.fit_time(
    data_list=[cosmic_time],
    labels=["Cosmic Background"],
    deg=2,
    normalize=True,
    out_file="cosmic_time_fit.pdf"
)
```

---

## 🔗 Integration with Main Framework

These standalone fitters are **independent of** the main `fit_module.py` likelihood chain but can:
- **Provide initial PDF parameters** for use in `physics_components.py`
- **Validate component shapes** against MC expectations
- **Study systematic variations** by refitting with modified parameters
- **Generate diagnostic plots** for collaboration reviews

The `fit_module.py` itself uses more complex combined fits (multiple components simultaneously), while these modules focus on **isolated component validation**.

---

## 📚 Related Documentation

- [**`fit_module.py`**](fitmodule_py.md) — Main likelihood fitter with combined components
- [**`physics_components.py`**](dictionaries.md) — Component dictionary definitions
- [**`custom_models.py`**](dictionaries.md) — Custom physics models (trunc_landau, res_components)
- [**`mu2e_plot_style.py`**](../../../component_fits/mu2e_plot_style.py) — Publication styling utilities
