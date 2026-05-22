## 📑 Mu2e Analysis Likelihood Definition Summary

The analysis group for the muon-to-electron conversion search is conducting a "shape" analysis using a maximum likelihood fitter written in Python leveraging the **zfit** package. This approach is paired with a Bayesian analysis using the **BAT.jl** framework.

The core of the analysis is an **extended likelihood** function, $\mathcal{L}$:

$$\mathcal{L}=\mathcal{P}_{\text{poisson}}(N_{\text{obs}};N_{\text{exp}})\cdot\prod_{i}^{N_{\text{obs}}}\left[\sum_{j}w_{j}f_{j}(p_{i},t_{i})\right]$$

### 1. Components of the Likelihood

| Component | Mathematical Term | Description |
| :--- | :--- | :--- |
| **Poisson Term** | $\mathcal{P}_{\text{poisson}}$ | Normalizes the likelihood. $N_{\text{exp}} = N_{s} + N_{b}$, where $N_{s}$ and $N_{b}$ are the expected signal and background events. This makes the total event count a parameter of the fit.  |
| **Product Term** | $\prod_{i}^{N_{\text{obs}}}$ | A product over all observed track events, $i$. |
| **Sum Term** | $\sum_{j}w_{j}f_{j}(p_{i},t_{i})$ |A weighted sum of the probability density functions (PDFs) for each process $j$ (e.g., DIO, RPC, signal), evaluated at the measurement $i$. |

### 2. The Model for a Single Process ($f_j$)

The model $f_j$ for a given process $j$ is defined as a function of momentum ($p$) and time ($t$). The fit is conducted in a **2D space** of momentum and time.

$$f_{j}=\left[f_{\text{theory},j}(p,t)\times(\mathcal{A}\times\epsilon_{\text{trig.}}\times\epsilon_{\text{reco}})_{j}(p,t)\times\epsilon_{\text{sel}.j}(p,t)\right]*(L\times r)_{j}$$

This model includes theoretical spectra and a series of efficiency and response terms:

| Term | Symbol | Description |
| :--- | :--- | :--- |
| **Theoretical Spectrum** | $f_{\text{theory},j}(p,t)$ | The theoretical assumed spectrum for the process. |
| **Acceptance** | $\mathcal{A}$ | The implicit tracker geometric acceptance. |
| **Trigger Efficiency** | $\epsilon_{\text{trig.}}$ | Includes the trigger and digitization efficiency, which has a time dependence. Cannot digitize for $t<450$ ns. |
| **Reconstruction Efficiency** | $\epsilon_{\text{reco}}$ | The efficiency to reconstruct a viable track. |
| **Selection Efficiency** | $\epsilon_{\text{sel.}}$ | The efficiency of pre-selection criteria, if applicable. |
| **Response Function** | $R(p) = L(p) \times r(p)$ | The total detector response, consisting of two sub-components: |
| $\quad$ Energy Loss | $L(p)$ | Momentum-dependent energy loss, naturally described by a Landau function. |
| $\quad$ Detector Response | $r(p)$ | Momentum-dependent detector response/resolution, often parameterized by a Generalized or Double-Sided Crystal Ball function. |

The total efficiency terms ($\mathcal{A}$, $\epsilon_{\text{trig.}}$, $\epsilon_{\text{reco}}$, $\epsilon_{\text{sel.}}$) are often combined into a single function, $\epsilon(p)$, parameterized by a Chebyshev polynomial function.

### 3. Key Process Models

The fit is conducted in the $\mathbf{95 < p < 115 \text{ MeV/c}}$ momentum region.

| Process | Momentum Component | Time Component | Notes |
| :--- | :--- | :--- | :--- |
| **Signal (CE)** | $f_{\text{CE}}(E)$ (Complex spectrum, $E=\sqrt{p^{2}+m_{e}^{2}}$) | $f_{\text{CE}}(t)=e^{-t/\tau_{\text{Al}_{\mu}}}$ | $\tau_{\text{Al}_{\mu}}$ (muonic aluminum mean lifetime) is assumed to be 864 ns. |
| **Decay in Orbit (DIO)** | $f_{\text{DIO}}(p)$ (Polynomial fit to theoretical spectrum) | $f_{\text{DIO}}(t)=e^{-t/\tau_{\text{Al}_{\mu}}}$ | Assumed to have the same time component as CE. |
| **Cosmic Induced** | Uniform, $U(a,b)$ with $a=95, b=115$ MeV/c | Uniform, $U(a,b)$ with $a=640, b=1650$ ns | Uniform distribution is a basic starting point; off-spill data is planned for a data-driven distribution. |
| **Radiative Pion Capture (RPC)** | Gaussian, $\mathcal{N}(\mu,\sigma^{2})$ | Exponential, $f_{\text{RPC}}(t)=e^{-t/\tau_{\pi}}$ | Gaussian is a simplification; $\tau_{\pi}$ is the free pion lifetime, as the pionic aluminum lifetime is unknown. Data-driven estimates are preferred. |

### DIO Custom Model (2025)

The analysis includes a custom, physics-motivated momentum PDF for Decay-In-Orbit (DIO) implemented as `DIO_custom_model_2025` in `py-fitter/momentum_pdf_builder.py`.

Form (unnormalized):
$$
f_{\mathrm{DIO}}(p) \propto
\begin{cases}
(\Delta E)^{5+\delta}\,\exp\!\big(\beta\,[\log(\Delta E/m_{\mu})]^2\big) & \Delta E\equiv E_{\mathrm{endpoint}}-p > 0,\\
0 & \text{otherwise.}
\end{cases}
$$

Parameters
- `DIO_endpoint`: endpoint energy (MeV). Typical code fallback: ~104.97 MeV.
- `beta`: coefficient of the log-squared correction (typical fallback: ~-0.002).
- `degree_shift` (\(\delta\)): small shift applied to the power (typical fallback: 0).

Notes
- The model is implemented as a TensorFlow/ZFit PDF and is constructed as an extended PDF via the `MomPDFBuilder.build()` method (yield handled via `N` in zfit).
- The implementation uses `tf.where`/safe-values to avoid taking `log` of non-positive arguments, improving numerical stability near the endpoint.
- When supplying `pardict` for this model via the `MomPDFBuilder.build()` method, initialize the parameters (endpoint, beta, degree_shift) or rely on the code defaults.

Example (simple usage):
```python
# build the momentum PDF for DIO using the builder
mom_builder = MomPDFBuilder()
PDF, N, params = mom_builder.build(obs_mom, 'DIO_custom_model_2025',
                                    {'endpoint': (104.97, 103.5, 106.0), 'beta': (-0.002, -0.01, 0.01), 'degree_shift': (0, -1, 1)},
                                    treat_params='float')
```

Implementation caveat
- The class `DIO_custom_model_2025` declares parameters internally named `['DIO_endpoint','beta','degree_shift']`, while the `MomPDFBuilder.build()` method may use alternate keys when constructing the TF/ZFit parameters. If you see unexpected defaults being used, ensure the `pardict` keys match the parameter lookup names (or provide both forms) so the ZFit-parameters propagate to the PDF constructor.

See `py-fitter/momentum_pdf_builder.py` for the exact implementation and `py-fitter/physics_components.py` for where the model name `DIO_custom_model_2025` may be referenced in `mom_components`.

### 4. Including Constraints and Uncertainties

The full likelihood is often combined with likelihoods from Control Regions (CRs) in a **simultaneous fit**:

$$\mathcal{L}_{\text{combined}}=\mathcal{L}_{\text{SR}}\times\prod_{k}\mathcal{L}_{\text{CR}_{k}}$$

This allows data from CRs to constrain shared parameters ($\theta$) in the Signal Region (SR).

Systematic uncertainties are incorporated into the full likelihood, $\mathcal{L}(\theta, \vec{\nu})$, through the inclusion of **nuisance parameters** ($\vec{\nu}$). These are typically constrained by Gaussian or Log Normal terms, reflecting external or subsidiary measurements.

* **Normalization Uncertainties** are handled by simple Gaussian/Log Normal constraint terms on nuisance parameters that scale yields.
* **Shape Uncertainties** (e.g., the DIO tail, resolution, and efficiency shapes) are complex and may be handled using:
    * **Templates/Morphing:** Describing the shape as a linear combination or interpolation between templates, with a mixing coefficient/morphing parameter acting as a nuisance parameter.
    * **Parametric Nuisance:** For functional forms (e.g., Double-Sided Crystal Ball for resolution), the uncertainty on the function's parameters is characterized by a nuisance parameter with a Gaussian constraint.
    
> Note: we are still finalizing our uncertainty analyses 

Most uncertainties will be input through subsidiary measurments. Each uncertainty requires careful study and is an analysis task in itself. In some cases we can do it ourselves, in other cases a dedicated other WG might provide us their final result.


