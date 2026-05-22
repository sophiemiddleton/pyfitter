# Dictionary Classes

The likelihood construction depends on several input configuration dictionaries and physics models, helping keep track of possible models for each component:

# ⚙️ `physics_components.py` - Momentum & Time Fit Configuration

This Python module consolidates all physics component dictionaries and constants for the fit, importing them into a single configuration source. It is used by `fit_module.py`, `momentum_pdf_builder.py`, and other core modules.

## 📜 `mom_components` Dictionary Structure

The top-level keys are the process names (e.g., `'CE'`, `'DIO'`), and the values are dictionaries containing the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| **`pdf`** | `str` | Name of the PDF model (e.g., `'dscb'`, `'uniform'`) as defined in `momentum_pdf_builder.py`. |
| **`pars`** | `dict`/`None` | Initial values, lower bounds, and upper bounds for the PDF parameters, excluding the yield $N$ (which is automatically added). |
| **`treat_params`** | `str` | Defines how to handle parameters: `'float'`, `'fix'`, `'constrain'`, `'param'`, or `'simul'`. |
| **`startCode`** | `list[int]` | Monte Carlo (MC) start codes (or process IDs) used for plotting truth-level categorization. |
| **`genCode`** | `list[int]` | Monte Carlo (MC) generator codes used for plotting truth-level categorization. |
| **`lineColor`**, `lineStyle` | `str` | Plotting styles for the PDF component line. |
| **`catColor`** | `str` | Color used to shade the MC truth component if categorization is enabled. |

### Default Components Example

| Process | PDF Model | `treat_params` | Notes |
| :--- | :--- | :--- | :--- |
| `'Cosmic'` | `'uniform'` | `'float'` | Models the flat component of cosmic ray backgrounds. |
| `'CE'` | `'dscb'` | `'float'` | **Signal** (Conversion Electron) modeled by a Double-Sided Crystal Ball function, with all shape parameters floating. |
| `'DIO'` | `'dio_custom_model_2025'` | `'fix'` | **Background** (Decay-in-Orbit) modeled by a custom theoretical spectrum, with shape parameters potentially fixed to theory. |
| `'RPC'` | `'Gauss'` | `'float'` | **Background** (Radiative Pion Capture) modeled by a Gaussian. |

## 📐 Advanced: Convolutional PDF (`'theo_exp'`)

The configuration includes complex, commented-out logic for defining the PDF of the Conversion Electron (CE) signal and the DIO background using the **`'theo_exp'`** model, which performs a convolution of several components.

This approach defines the final measured momentum distribution $P_{\text{meas}}(p)$ as:
$$P_{\text{meas}}(p) = P_{\text{theo}}(p) \otimes P_{\text{eff}}(p) \otimes P_{\text{res}}(p) \otimes P_{\text{loss}}(p)$$
where $\otimes$ denotes convolution.

### Key Advanced Components:

1.  **Theoretical Lineshape (`'theo'`):** The intrinsic momentum spectrum for the process (e.g., from `binned_spectrum_CeLL()` or a text file for DIO).
2.  **Efficiency (`'eff'`):** A function or histogram representing the momentum-dependent selection efficiency.
3.  **Resolution and Energy Loss (`'res'`, `'loss'`):** These are the experimental effects:
    * **Resolution:** The detector's intrinsic measurement uncertainty (`flat_res`).
    * **Energy Loss:** Energy loss as the particle traverses the detector material (`flat_loss`).

### Parameter Treatment in Advanced Fits

When using `'theo_exp'`, the shape parameters of the resolution and loss functions are managed by the `res_components` class in `custom_models.py`.

| `treat_params` Setting | Description |
| :--- | :--- |
| **`'param'`** | Used when the parameters are defined **for the first time** (e.g., by loading a pre-fit `res_components` object). These parameters are added to the list of total fit parameters. |
| **`'simul'`** | Used when the parameters have already been defined by a previous component (e.g., CE) and are being **reused/shared** by the current component (e.g., DIO). This is common for shared resolution and energy loss models. |
| **`'nll'`** | If the resolution/loss parameters are fit independently (e.g., from a sideband control sample), the `res_components` sources are listed here. Their NLLs are added to the main fit's loss function for a **simultaneous, constrained fit**. |


# ⏱️ `time_components.py` - Time Fit Configuration

This dictionary defines all physics processes included in the time-of-arrival likelihood fit, specifying the PDF model, initial parameters (specifically the decay rate $\lambda$), and visualization settings for each.

## 🕰️ `time_components` Dictionary Structure

The structure of this dictionary is analogous to `mom_components.py`, but the physics model is centered on exponential decay times.

| Field | Type | Description |
| :--- | :--- | :--- |
| **`pdf`** | `str` | Name of the PDF model (e.g., `'muexp'`, `'uniform'`) from `momentum_pdf_builder.py`. |
| **`pars`** | `dict`/`None` | Initial values and bounds for the PDF's decay rate parameter $\lambda$. The decay rate is typically $\lambda = -1/\tau$, where $\tau$ is the characteristic lifetime (in ns). |
| **`startCode`**, **`genCode`** | `list[int]` | Monte Carlo (MC) codes used for truth-level categorization and plotting. |
| **`lineColor`**, `lineStyle` | `str` | Plotting styles for the PDF component line. |
| **`catColor`** | `str` | Color used to shade the MC truth component. |

### Component Definitions

The time-of-arrival fit is dominated by the lifetime of the stopped muon ($\mu^-$) in the target, which governs both the signal and the largest background.

| Process | PDF Model | Decay Rate Parameter $\lambda$ | Notes |
| :--- | :--- | :--- | :--- |
| `'Cosmic'` | `'uniform'` | N/A | Cosmic ray backgrounds that are distributed uniformly across the measurement time. |
| `'CE'` | `'muexp'` | $\lambda_{\mu} \approx -1/864$ ns$^{-1}$ | **Signal** (Conversion Electron) and its time distribution are governed by the parent muon's lifetime (approx. $864\text{ ns}$ in Aluminum).  |
| `'DIO'` | `'muexp'` | $\lambda_{\mu} \approx -1/864$ ns$^{-1}$ | **Background** (Decay-in-Orbit) is also a decay of a stopped muon, so it shares the same exponential time distribution as the CE signal. |
| `'Pion'` | `'piexp'` | $\lambda_{\pi} \approx -1/26$ ns$^{-1}$ | **Background** (Radiative Pion Capture/Pion Decay) from stopped pions, which have a much shorter lifetime ($\tau_{\pi} \approx 26\text{ ns}$). |

### Shared Muon Lifetime

Both the **CE Signal** and the **DIO Background** are sourced from muons stopping in the target and subsequently decaying. Therefore, they are both modeled by the `'muexp'` PDF and, crucially, share the same physical decay rate parameter, $\lambda_{\mu}$. In a simultaneous fit (not explicitly shown here but implied by physics consistency), this parameter would often be shared or highly constrained across both components.

# 💻 `custom_models.py` - Custom Physics Models

This consolidated module provides theoretical spectrum calculations, custom PDFs, and detector resolution/loss parameterization. It contains:
- Spectrum calculation functions (`LeadingLog`, `binned_spectrum_CeLL`)
- Custom Landau PDF (`trunc_landau`) for energy loss
- Detector response handler (`res_components` class)

## ⚛️ Conversion Electron (CE) Spectrum

The CE process ($\mu^- N \to e^- N$) is a flavor-violating decay that produces a mono-energetic electron in the two-body limit. However, due to radiative corrections, the spectrum acquires a radiative tail.

### Constants and Parameters

* $e_{\text{Max}} = 104.97$ MeV: The kinematic endpoint of the CE electron momentum.
* $\alpha = 1/137.036$: The fine-structure constant.
* $m_e = 0.511$ MeV: The electron mass.

### Function: `LeadingLog(E)`

This function implements the theoretical calculation for the normalized differential momentum spectrum of the CE electron, including the dominant $\mathcal{O}(\alpha)$ **Leading Logarithm (LL)** radiative correction terms.

* The formula used is derived from quantum electrodynamics (QED) and describes the probability of the electron having momentum $E$ while radiating a photon.
* **Formula:**
    $$\text{LL}(E) \propto \frac{1}{e_{\text{Max}}} \frac{\alpha}{2\pi} \left( \ln \left( \frac{4E^2}{m_e^2} \right) - 2 \right) \frac{E^2 + e_{\text{Max}}^2}{e_{\text{Max}} (e_{\text{Max}} - E)}$$
    * The term $(e_{\text{Max}} - E)$ in the denominator causes the spectrum to peak sharply and drop off steeply near the endpoint, $e_{\text{Max}}$.
    * A cut-off is applied (`if val < 0.0: val = 0.0`) to ensure a physical PDF.

### Function: `binned_spectrum_CeLL(binwidth=0.1)`

This function discretizes the theoretical `LeadingLog` spectrum into a binned histogram format suitable for use in the convolution fitting scheme (`'theo_exp'` PDF in `momentum_pdf_builder.py`).

* It calculates the bin `edges` and the spectrum `values` (PDF value at the bin centers) across the momentum range from $0$ to $e_{\text{Max}}$.
* It performs a final normalization step to ensure the sum of the binned probabilities is exactly unity (or close to unity, allowing the last bin to absorb small numerical deviations).
* **Output:** Returns a tuple `(values, edges)` representing the binned histogram of the theoretical spectrum.

# 🔍 Resolution and Energy Loss Parameterization (in `custom_models.py`)

The `res_components` class in `custom_models.py` manages the modeling of detector effects—specifically, momentum resolution and energy loss—as a function of the reconstructed momentum. This class is designed to integrate these effects into the overall likelihood fit (via the `'theo_exp'` convolution PDF).

## 🎚️ `class res_components`

This class handles the creation and management of parameters for detector response functions, typically parameterized in bins of true momentum.

### Initialization (`__init__`)

The initializer sets up the parameterization based on whether the parameters are being loaded from a previous fit or are meant to be floated in a new simultaneous fit.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`p_bins`** | `list[float]` | Edges of momentum bins used for parameterization (e.g., $95-97$ MeV/c). |
| **`params`** | `str` / `dict` / `None` | If provided, parameters are loaded from a pickle file (`str`) or dictionary (`dict`) and are typically **fixed or constrained** in the main fit. |
| **`simul_source`** | `tuple[ak.Array]` / `None` | If provided, this is the truth-level and reconstructed-level data. The parameters are treated as **floating** and will be fit simultaneously using their own NLL loss term. |
| **`res_type`** | `str` | Specifies the type of physical effect: `'res'` (resolution) or `'loss'` (energy loss). |
| **`pdf`** | `str` | The PDF used for the response function: `'gcb'` (Generalized Crystal Ball for Resolution) or `'landau'` (Truncated Landau for Energy Loss). |

### Parameter Creation Logic

1.  **Simultaneous Fit (`simul_source` is set):**
    * Floating `zfit.Parameter` objects are created for each PDF parameter (e.g., `mu`, `sigmaL`, `nL`, etc.) within *each* momentum bin defined by `p_bins`.
    * The parameter names are constructed to include the bin index and the `res_type` (e.g., `mu0_res`).
2.  **Fixed/Constrained Fit (`params` is set):**
    * Parameters are loaded from the external dictionary or pickle file.
    * `zfit.Parameter` objects are created with initial values and bounds derived from the loaded dictionary, ready to be fixed or constrained in the main fit.

### Method: `get_params(self)`

* Returns the dictionary `self.fitpars`, which contains the parameter objects and the binning information. This dictionary is passed to `mom_components.py` to be used in the convolution model.

### Method: `get_nll(self, params_tot)`

This is the core method used when performing a **simultaneous fit** to the response function alongside the main physics fit.

1.  **Data Slicing:** The input data (`true_mom`, `reco_mom`) is sliced into subsets corresponding to the defined momentum bins.
2.  **Residual Calculation:** The relevant residual is calculated:
    * **Resolution (`'res'`):** Residual $\approx \text{Reco Momentum} - \text{True Momentum}$.
    * **Energy Loss (`'loss'`):** Residual $\approx \text{Reco Momentum} - \text{True Momentum}$ (though physically loss is $\text{True} - \text{Reco}$, the convention here is for the residual fit).
3.  **Local Extended NLL:** For each momentum bin:
    * A small, local extended NLL is constructed using the appropriate PDF (`GeneralizedCB` or `TruncatedLandau`) and the corresponding data slice (`data_res`).
    * A floating yield parameter ($N_{\text{slice}}$) is also included.
    * This local NLL is minimized **simultaneously** with the main physics fit, effectively constraining the response parameters.
4.  **Output:** Returns a list of `zfit.loss.ExtendedUnbinnedNLL` objects, one for each momentum bin, which are added to the main fit's loss function in `fit_module.py`.

# 📊 `sysunc_components.py` - Systematic Uncertainty Configuration

This configuration file defines the dictionary `sysunc_components`, which itemizes and quantifies the major sources of systematic uncertainty (sysunc) relevant to the momentum and time likelihood analysis.

## 📝 `sysunc_components` Dictionary Structure

Each entry in the dictionary represents a distinct source of uncertainty (e.g., `'Abs_Mom_Scale'`). Its value is a dictionary containing the properties of that uncertainty:

| Field | Type | Description |
| :--- | :--- | :--- |
| **`type`** | `str` | Defines how the uncertainty is applied: |
| | `'shift'` | An absolute uncertainty applied as a fixed $\pm \text{Value}$ (e.g., $\pm 0.1$ MeV/c). |
| | `'frac'` | A fractional uncertainty applied as a percentage on the yield (e.g., $\pm 2.5\%$ on the total number of events). |
| | `'shape'` | Not currently used, but indicates uncertainty on a specific PDF shape parameter (e.g., $\sigma$ of a Gaussian). |
| **`sim`** | `bool` | `True` indicates this value is currently derived from simulation studies. The expectation is that many of these will be replaced or constrained by data-driven measurements in later analysis stages. |
| **`process`** | `str` | The physics process affected by the uncertainty. Can be a specific component (e.g., `'DIO'`, `'RPC'`, `'Cosmics'`) or `'all'`. |
| **`component`** | `list[str]` | The observable(s) affected: `'mom'`, `'time'`, or both. |
| **`value`** | `list[float]` | The magnitude of the uncertainty, typically a list of two values representing the asymmetric lower and upper bounds $[\text{minus}, \text{plus}]$. |

## 📈 Defined Systematic Uncertainties

The configuration groups uncertainties by the process they affect:

### General Uncertainties (Affecting All Processes)

| Key | `type` | `process` | `value` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `'Abs_Mom_Scale'` | `'shift'` | `'all'` | $[0.1, 0.1]$ MeV/c | Uncertainty on the absolute momentum calibration of the detector. This affects the peak position/endpoint of all momentum distributions.  |
| `'Mom_Res'` | N/A | N/A | N/A | **Placeholder** for momentum resolution uncertainty, which would affect the width of the momentum distributions. |

### DIO (Decay-in-Orbit) Background

| Key | `type` | `process` | `value` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `'DIO_Theory'` | `'frac'` | `'DIO'` | $[0.025, 0.025]$ | A $\pm 2.5\%$ fractional uncertainty on the total predicted DIO yield, reflecting uncertainties in the theoretical calculation of the decay rate. |

### RPC (Radiative Pion Capture) Background

| Key | `type` | `process` | `value` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `'RPC_rate'` | `'frac'` | `'RPC'` | $[0.093, 0.093]$ | $\pm 9.3\%$ fractional uncertainty on the RPC rate, likely due to uncertainties in nuclear capture or target composition (e.g., presence of magnesium). |
| `'pion_rate'` | `'frac'` | `'RPC'` | $[0.27, 0.09]$ | An asymmetric uncertainty from $-27\%$ to $+9\%$ on the total pion stopping rate, derived from Geant4 simulation studies. |
| `'internalconv_rate'` | `'frac'` | `'RPC'` | $[0.0045, 0.0045]$ | $\pm 0.45\%$ fractional uncertainty on the yield of backgrounds associated with internal conversions following pion capture. |

### Cosmics Background

| Key | `type` | `process` | `value` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `'CRV_eff'` | `'frac'` | `'Cosmics'` | $[0.04, 0.04]$ | $\pm 4\%$ fractional uncertainty on the efficiency of the Cosmic Ray Veto (CRV) system. |
| `'generator'` | `'frac'` | `'Cosmics'` | $[0.20, 0.20]$ | A large $\pm 20\%$ fractional uncertainty reflecting differences between various Monte Carlo event generators used to simulate cosmic ray backgrounds. |

---

> **Missing/Future Uncertainties:** The file notes that crucial uncertainties such as the Out-of-Time RPC, Beam Extinction uncertainty, and CRV aging effects are yet to be included.


