# 📉 `momPDF_module.py` - Momentum Probability Density Functions

This module defines the custom and standard Probability Density Functions (PDFs) used to model the momentum spectrum of physics processes in the maximum likelihood fit. It handles parameter initialization, parameter treating (fixing/floating/constraining), and model selection using the **`zfit`** framework.

## ⚛️ Custom DIO Models

The momentum spectrum of the Decay-in-Orbit (DIO) background is complex, dropping sharply near the endpoint, it requires specialized theoretical parameterizations. Given there are no experimentally verified parameterizations there remains significant uncertainty in the exact shape of the tail. Therefore, specific emphasis is taking place to properly understand the DIO shape.

### `class poly58(zfit.pdf.ZPDF)`

This class implements the classic theoretical parameterization for the DIO spectrum, as calculated by **2016 Czarnecki et al.**. It is a binner version of this parameterization that is used within the Mu2e simulation framework.

* **Logic:** The unnormalized PDF is defined as a polynomial of the form:
    $$\text{PDF}(x) \propto a_5 \Delta^5 + a_6 \Delta^6 + a_7 \Delta^7 + a_8 \Delta^8$$
    where $x$ is the electron momentum and $\Delta = E_{\mu} - x - x^2 / (2 m_{\text{Al}})$ is the energy released (endpoint energy minus electron energy).
* **Parameters:** $a_5, a_6, a_7, a_8$ are the fit parameters, typically highly constrained or fixed to theoretical values.
* **Endpoint Cut:** The use of `tf.nn.relu` ensures the PDF is zero for momentum values above the theoretical endpoint ($x > E_{\mu}$).


### `class DIO_custom_model_2025(zfit.pdf.ZPDF)`

A modern, more flexible custom parameterization for the DIO spectrum often used to account for potential nuclear effects or experimental uncertainties beyond the standard theoretical prediction.

* **Logic:** The unnormalized PDF is of the form:
    $$\text{PDF}(x) \propto (\text{Endpoint} - x)^{5 + \text{shift}} \cdot \exp \left( \beta \cdot \ln^2 \left( \frac{\text{Endpoint} - x}{m_{\mu}} \right) \right)$$
* **Parameters:**
    * `DIO_endpoint`: Floating parameter for the kinematic endpoint of the decay.
    * `beta`: Controls a logarithmic exponential correction term.
    * `degree_shift`: Modifies the nominal $\Delta^5$ power-law behavior.

## 🏭 Core Function: `MomModel(...)`

This function is the main entry point called by `fit_module.py` to construct a combined momentum PDF component and its yield parameter.

### 1. Yield Parameter ($N$) Initialization

* A floating `zfit.Parameter` named `N_{process}` is created to represent the extended likelihood yield (number of events) for the component (e.g., $N_{\text{CE}}, N_{\text{DIO}}$).
* Initial values are taken from the input `pardict` or sensible defaults (`default_norms`).

### 2. PDF Parameter Initialization (`zpars`)

* It determines the parameters required for the requested `model` (e.g., `mu`, `sigma` for a Gaussian).
* It applies a **treatment** specified in `treat_params` from `mom_components.py`:
    | `treat_params` | Action | Resulting `zfit` Object |
    | :--- | :--- | :--- |
    | `float` | Parameter is free to float during the minimization. | `zfit.Parameter` |
    | `fix` | Parameter is held constant to its initial value. | `zfit.Parameter(floating=False)` |
    | `constrain` | Parameter is floated, but penalized by a $\chi^2$ term if it moves far from its initial value. | `zfit.constraint.GaussianConstraint` |

### 3. PDF Construction

Based on the `model` string, the appropriate `zfit` PDF is instantiated:

| `model` | Description | `zfit` Class | Component |
| :--- | :--- | :--- | :--- |
| `dscb` | Double Sided Crystal Ball | `zfit.pdf.DoubleCB` | signal |
| `gcb` | Generalized Crystal Ball | `zfit.pdf.GeneralizedCB` | signal |
| `Gauss` | Gaussian (Normal) | `zfit.pdf.Gauss` | RPC |
| `uniform` | Flat distribution | `zfit.pdf.Uniform` | Cosmics/RPC |
| `kde` | Kernel Density Estimator (non-parametric) | `zfit.pdf.KDE1DimGrid` | signal |
| `poly58` | Czarnecki DIO model (custom) | `poly58` | DIO |
| `theo_exp` | Convolution of a theoretical PDF with resolution terms. | `zfit.pdf.TruncatedPDF` | custom |

* **Output:** Returns the instantiated `zfit.pdf.ZPDF` object and its corresponding yield parameter $N$.
