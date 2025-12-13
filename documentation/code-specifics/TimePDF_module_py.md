# ⏳ `timePDF_module.py` - Time Probability Density Functions

This module defines the PDFs used to model the reconstructed **track time-of-arrival** spectrum, primarily focusing on exponential decay components characteristic of unstable particle lifetimes. It provides the **`TimeModel`** function to construct individual PDF components for the likelihood fit using the `zfit` framework.

## 🕰️ Time-Dependent Models

In many particle physics experiments, backgrounds associated with unstable particles or delayed processes are modeled by exponential distributions in time.

| Model Name | Physics Process | Description | `zfit` Class | Decay Constant (Default) |
| :--- | :--- | :--- | :--- | :--- |
| `muexp` | Muon decay in orbit (DIO/CE) | Models the decay of a stopped muon ($\mu^-$). | `zfit.pdf.Exponential` | $\lambda_{\mu} \approx -0.001157$ ns$^{-1}$ |
| `piexp` | Pion decay in orbit ($\pi$) | Models the decay of stopped pions ($\pi^-$). | `zfit.pdf.Exponential` | $\lambda_{\pi} \approx -0.03846$ ns$^{-1}$ |
| `cosmicexp` | Cosmic Ray Background (CRV) | Often used to model the exponential tail of delayed cosmic-ray induced background. | `zfit.pdf.Exponential` | $\lambda_{\text{CRV}} \approx -0.037$ ns$^{-1}$ |
| `uniform` | Prompt or uniform background | Models backgrounds that are flat across the fit time range. | `zfit.pdf.Uniform` | N/A |

> **Note on Decay Rate:** The `zfit.pdf.Exponential` takes a $\lambda$ parameter. Since the PDFs here are defined as $\exp(\lambda \cdot t)$, the decay rate parameter is the negative inverse of the characteristic lifetime: $\lambda = -1/\tau$. The default values reflect the measured or expected particle lifetimes (e.g., muon lifetime $\tau_{\mu} \approx 864$ ns in the aluminum target).

> **Note** Given the DIO and CE have the same time distribution this introduces one less time component compared to the momentum components. This makes the 2D fit tricky.


## 🛠️ Core Function: `TimeModel(...)`

This function is the main interface for building a single time PDF component, its decay parameters, and its yield. It is called repeatedly by `fit_module.py`.

### 1. Yield Parameter ($N$) Initialization

* A floating `zfit.Parameter` named `N_{process}` is created to represent the extended likelihood yield (number of events) for the component (e.g., $N_{\text{Cosmic}}, N_{\text{DIO}}$).
* Initial values are taken from the input `pardict` or sensible defaults (`default_norms`). The yield is appended to `params_tot`.

### 2. PDF Construction

* The function selects the appropriate decay model (`muexp`, `piexp`, `uniform`, etc.) and retrieves its parameters from `default_model_params`.
* **Parameter Instantiation:** A floating `zfit.Parameter` is created for the decay rate of the component (e.g., `decay_rate_mu`).
* **PDF Instantiation:** The corresponding `zfit.pdf.Exponential` or `zfit.pdf.Uniform` object is created, passing the observable (`obs_time`), the decay rate parameter, and the yield parameter (`N`).

* **Output:** Returns the instantiated `zfit.pdf.ZPDF` object and its corresponding yield parameter $N$.
