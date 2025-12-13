# 🧪 `fit_module.py` - Unbinned Maximum Likelihood Fitting

The `fit_module.py` is the primary interface for performing unbinned maximum likelihood fits to particle kinematic variables (momentum and time-of-arrival) using the **`zfit`** fitting library. It defines the loss function, initializes the PDF components, and executes the minimization routine (`Minuit`).

## 📚 Dependencies

This module imports and utilizes several key custom and third-party modules:

* **`zfit`**: The core library for unbinned maximum likelihood fitting.
* **`momPDF_module.py` / `timePDF_module.py`**: Custom classes (`MomModel`, `TimeModel`, `MomTimeModel`) that build the individual Probability Density Functions (PDFs) and their normalization yields (`norms`) for each physics process.
* **`mom_components` / `time_components`**: Dictionaries defining the PDF type (e.g., Gaussian, polynomial) and initial parameters for each background and signal process.
* **`recoplot_module.py`**: Functions (`plotmom_fit`, `plottime_fit`) used for visualizing the fit results overlaid on the data histogram.

## 📐 Core Function: `Unbinned_fit_mom(...)`

Performs an unbinned maximum likelihood fit to the **track momentum magnitude**.

### 1. Initialization and Data Preparation

1.  **Observable Space (`zfit.Space`):** Defines the fit range (e.g., $100$ to $115$ MeV/c) for the momentum observable, $x$.
2.  **Data Conversion:** The input `awkward.Array` (`mom_mag`) is flattened, cleaned of NaNs, and converted into a `zfit.Data` object.
    > The conversion of the input data from a ragged `awkward.Array` to a flat `numpy` array is a necessary step before passing it to `zfit`, which operates on flat data structures.
3.  **PDF Components:** It iterates over the `mom_components` dictionary, initializing an individual PDF model (e.g., Signal, DIO, RMC) for each physics process using `MomModel()`. Each model contributes a PDF and a floating yield/normalization parameter.

### 2. Model Construction and Loss Function

1.  **Combined PDF:** The individual PDF components are combined into a single total PDF using `zfit.pdf.SumPDF`:
    $$\text{PDF}_{\text{total}}(x) = \sum_{i} N_{i} \cdot \text{PDF}_{i}(x)$$
    where $N_i$ is the yield (normalization) for process $i$.
2.  **Extended Negative Log-Likelihood (NLL):** The loss function is defined as the extended NLL, which simultaneously fits the shape and the yield of each component.
    $$\text{NLL} = -\sum_{j} \ln \left( \text{PDF}_{\text{total}}(x_j) \right) + \sum_{i} N_{i}$$
    Constraint terms (`nlls`) from external likelihoods (e.g., normalization from sideband studies) are added to the loss.
    

### 3. Minimization and Results

1.  **Minimizer:** An instance of the `Minuit` minimizer is used to find the parameters that minimize the NLL.
2.  **Error Estimation (`minos`):** Optionally, the asymmetric `minos` error estimation is performed for more accurate parameter uncertainties.
3.  **Visualization:** The fit result is plotted using `plotmom_fit` to display the data histogram alongside the fitted components and the total PDF curve.
4.  **NLL Scan (Optional):** Includes optional logic to perform an NLL scan over the signal yield parameter ($N_{\text{sig}}$) to visualize the likelihood minimum and calculate confidence intervals (e.g., $1\sigma$ or $\Delta \text{NLL}=0.5$).

## ⏰ Core Function: `Unbinned_fit_time(...)`

Performs an unbinned maximum likelihood fit to the **track time-of-arrival**.

* This function follows the exact same structure as `Unbinned_fit_mom`, but uses the `TimeModel` and iterates over the `time_components` dictionary, fitting to the time observable instead of momentum.
* The output is visualized using `plottime_fit`.

## 2D Fit: `Unbinned_2d_fit_mom_time(...)` **FIXME**

Performs a combined, simultaneous unbinned maximum likelihood fit to **both momentum and time** by treating them as independent observables in a 2D space.

* **Observable Space:** Defines a combined `zfit.Space` for the two observables: `obs_2D = obs_mom * obs_time`.
* **Model:** Uses the `MomTimeModel` to build combined 2D PDFs for each process $i$:
    $$\text{PDF}_{i}(p, t) = \text{PDF}_{\text{mom}, i}(p) \cdot \text{PDF}_{\text{time}, i}(t)$$
    This assumes statistical independence between the momentum shape and the time shape of a given process.
* **Data Conversion:** The input momentum and time arrays are combined into a 2D NumPy array (`np.column_stack`) before being passed to `zfit.Data`.
* **Result Visualization:** After the 2D fit, separate 1D projections (marginal fits) are plotted for time and momentum.
