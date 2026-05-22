# 📉 `momentum_pdf_builder.py` - PDF Builders for Momentum & Time

This consolidated module defines the core PDF builder classes used to construct Probability Density Functions (PDFs) for both **momentum** and **time-of-arrival** observables in unbinned maximum likelihood fits. It includes custom DIO models, standard `zfit` PDFs, parameter initialization, and the combined 2D builders used by `fit_module.py`.

## 🏗️ Core Builder Classes

### `class MomPDFBuilder(PDFBuilder)`

Constructs momentum PDFs for individual physics processes (signal and backgrounds). Inherits from the abstract `PDFBuilder` template class.

**Key Methods:**
- `build(model, obs_space, params_dict, treat_params='float', **kwargs)` - Main entry point to build a momentum PDF component
  - Returns: `(zfit_pdf, yield_param, params_tot)` tuple
  - Supports models: `dscb`, `Gauss`, `uniform`, `kde`, `poly58`, `theo_exp`, `DIO_custom_model_2025`
  
**Parameter Treatment Options:**
| `treat_params` | Action | Resulting `zfit` Object |
| :--- | :--- | :--- |
| `'float'` | Parameter is free to float during minimization. | `zfit.Parameter` |
| `'fix'` | Parameter is held constant to its initial value. | `zfit.Parameter(floating=False)` |
| `'constrain'` | Parameter is floated with a Gaussian penalty if it moves far from initial value. | `zfit.constraint.GaussianConstraint` |

### `class TimePDFBuilder(PDFBuilder)`

Constructs time-of-arrival PDFs for individual physics processes, modeling exponential decay and flat backgrounds.

**Key Methods:**
- `build(model, obs_space, params_dict, treat_params='float', **kwargs)` - Main entry point to build a time PDF component
  - Returns: `(zfit_pdf, yield_param, params_tot)` tuple
  - Supports models: `'muexp'`, `'piexp'`, `'cosmicexp'`, `'uniform'`

**Available Time Models:**

| Model Name | Physics Process | PDF Class | Decay Parameter (Default) |
| :--- | :--- | :--- | :--- |
| `'muexp'` | Muon decay (CE/DIO) | `zfit.pdf.Exponential` | $\lambda_{\mu} \approx -0.001157$ ns$^{-1}$ |
| `'piexp'` | Pion decay | `zfit.pdf.Exponential` | $\lambda_{\pi} \approx -0.03846$ ns$^{-1}$ |
| `'cosmicexp'` | Cosmic ray background | `zfit.pdf.Exponential` | $\lambda_{\text{CRV}} \approx -0.037$ ns$^{-1}$ |
| `'uniform'` | Prompt/flat background | `zfit.pdf.Uniform` | N/A |

> **Note on Decay Rate:** The `zfit.pdf.Exponential` parameter $\lambda$ represents the decay rate. Since PDFs use $\exp(\lambda \cdot t)$, the decay constant is the negative inverse of the lifetime: $\lambda = -1/\tau$.

### `class MomTimePDFBuilder`

Combines momentum and time PDFs into a 2D `ProductPDF` for simultaneous fitting of both observables.

**Key Methods:**
- `build(mom_pdf, time_pdf, obs_space_2d)` - Constructs the 2D combined PDF
  - Returns: Combined `zfit.pdf.ProductPDF` object

## ⚛️ Custom DIO Models

### `class poly58(zfit.pdf.ZPDF)`

Implements the classic theoretical DIO spectrum parameterization from **Czarnecki et al. (2016)**, as used in the Mu2e simulation.

* **Formula:** $$\text{PDF}(x) \propto a_5 \Delta^5 + a_6 \Delta^6 + a_7 \Delta^7 + a_8 \Delta^8$$
    where $x$ is electron momentum and $\Delta = E_{\mu} - x - x^2 / (2 m_{\text{Al}})$
* **Parameters:** $a_5, a_6, a_7, a_8$ (typically constrained or fixed)
* **Endpoint Handling:** Uses `tf.nn.relu` to zero the PDF above the theoretical endpoint

### `class DIO_custom_model_2025(zfit.pdf.ZPDF)`

A modern, flexible parameterization for the DIO spectrum allowing for nuclear effects and experimental uncertainties.

* **Formula:** $$\text{PDF}(x) \propto (\text{Endpoint} - x)^{5 + \text{shift}} \cdot \exp \left( \beta \cdot \ln^2 \left( \frac{\text{Endpoint} - x}{m_{\mu}} \right) \right)$$
* **Floating Parameters:**
    * `DIO_endpoint`: Kinematic endpoint of the DIO decay
    * `beta`: Logarithmic correction term strength
    * `degree_shift`: Modifies the power-law behavior

## 📊 Standard Momentum Models

| `model` | Description | `zfit` Class | Component |
| :--- | :--- | :--- | :--- |
| `'dscb'` | Double-Sided Crystal Ball | `zfit.pdf.DoubleCB` | signal |
| `'Gauss'` | Gaussian (Normal) | `zfit.pdf.Gauss` | RPC |
| `'uniform'` | Flat distribution | `zfit.pdf.Uniform` | Cosmics/RPC |
| `'kde'` | Kernel Density Estimator | `zfit.pdf.KDE1DimGrid` | signal |
| `'poly58'` | Czarnecki DIO model (custom) | `poly58` | DIO |
| `'theo_exp'` | Convolution of theoretical spectrum with resolution/loss | `zfit.pdf.TruncatedPDF` | custom |
| `'DIO_custom_model_2025'` | Modern DIO parameterization (custom) | `DIO_custom_model_2025` | DIO |
