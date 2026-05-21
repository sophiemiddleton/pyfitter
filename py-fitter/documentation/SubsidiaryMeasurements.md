# 🚀 Full Fit Framework with Subsidiary Measurement Defintions

## 1. Core Architecture Overview
The codebase is structured to allow a **Simple Baseline Fit** (fast, standard shapes) and an **Advanced Physics Fit** (computationally intensive, convolution-based) to coexist in the same dictionary.

### Key Logic Toggle
The behavior of the entire pipeline is controlled by the `use_advanced` flag in `fit_module.py`:
*   **`use_advanced=False`**: Uses standard analytical PDFs (DSCB, Gauss, Custom Theory formulas).
*   **`use_advanced=True`**: Uses `theo_exp` (Theoretical lineshapes convolved with Experimental resolution/loss).

---

## 2. The Component Dictionary (`physics_components.py`)
Each physics process (CE, DIO, Cosmic, RPC) is defined as a dictionary with two primary blocks:

### A. Simple Block (Standard)
These keys define the behavior for standard analytical fits.
- **`pdf`**: The name of the Zfit PDF class (e.g., `'dscb'`, `'Gauss'`).
- **`pars`**: Start values, limits, or fixed values for the PDF parameters.
- **`treat_params`**: Control logic for parameters:
    - `'float'`: Standard floating fit parameter.
    - `'fix'`: Parameter value is locked.
    - `'constrain'`: Adds a Gaussian constraint to the NLL.

### B. Advanced Block (`advanced_pars`)
This block is activated only when `use_advanced=True`.
- **`pdf_theo`**: Set to `'theo_exp'` to trigger the convolution logic.
- **`treat_params_adv`**: 
    - `'param'`: Defines the primary resolution/loss parameters.
    - `'simul'`: Shares parameters with another process (e.g., DIO sharing CE's resolution).
- **`fitpars_in_formatted`**: A pre-processed bundle containing the theoretical lineshape and experimental response objects.
- **`nll_sources`**: A list of objects (like `flat_res`) that provide auxiliary likelihood terms for a **simultaneous fit**.

---

## 3. Simultaneous Fit Options
The framework automatically detects if a "Simultaneous Fit" should be performed based on the data provided to the resolution objects.

|Mode | Configuration |	Behavior|
| :--- | :--- | :--- |
| Fixed|  `res_components(params='path.pkl')` (from `custom_models.py`) | Uses the fixed parameters from the pickle file. No extra NLL terms added.|
| Simultaneous	| `res_components(simul_source=(gen, mc))` (from `custom_models.py`) | Performs a simultaneous fit of the resolution/loss using the provided data tuples. Adds get_nll to the total loss. |
---

## 4. Key Modules Functionality

### `momentum_pdf_builder.py`
The "Factory" module. It contains the logic to:
1.  Unpack the `advanced_pars` block and route to the correct logic branch.
2.  Assign parameters to `zfit.Parameter` (independent) or `zfit.ComposedParameter` (shared/simultaneous).
3.  Instantiate custom physics models like `poly58` or `DIO_custom_model_2025`.

### `fit_module.py`
The "Orchestrator" module. It handles:
1.  **Data Processing**: Converting Awkward arrays to `zfit.Data` objects.
2.  **NLL Construction**: Summing the primary Extended Unbinned NLL with any auxiliary NLL terms from `nll_sources`.
3.  **Minimization**: Running the Minuit optimizer and calculating MINOS errors.

### `helper.py`
The "Math" module. Contains:
1.  **`doConv`**: High-performance FFT convolution logic using TensorFlow.
2.  **`make_HistogramPDF`**: Converts theoretical tables (like Czarnecki DIO) into differentiable Zfit objects using vectorized bin lookup.

---

## 5. Typical Workflow Example

### Running a fast signal check:
In `process.py`, set:
```python
Unbinned_fit_mom(..., use_advanced=False)
```

---

## 6. Data Inputs and File Handling
The framework relies on specific file formats to bridge the gap between offline simulation and the Zfit optimization environment.

### A. Parameter Pickles (`.pkl`)
These are serialized Python dictionaries (using `pickle` or `dill`) containing previous fit results.
*   **Usage**: Used in `res_components` (in `custom_models.py`) to initialize the starting values or to fix parameters for resolution (`gcb`) and energy loss (`landau`).
*   **Mechanism**: The `res_components` class extracts the `'best'` or `'params'` keys to populate `zfit.Parameter` objects.
*   **Example**: `fitpars_flat_res_entrance_gcb.pkl` provides the $\mu, \sigma, \alpha, n$ values for the experimental response.

### B. Theoretical Tables (`.tbl` / `.txt`)
Flat text files containing raw physics theory data (e.g., Czarnecki DIO spectrum).
*   **Format**: Two columns: `[Momentum (MeV/c), Probability Density]`.
*   **Mechanism**: `load_lineshape` in `helper.py` reads these files and `make_HistogramPDF` converts them into a differentiable TensorFlow-based PDF.
*   **Example**: `czarnecki_szafron_Al_2016.tbl` is used for the high-fidelity DIO theoretical model.

### C. Efficiency Histograms (`.pkl`)
Pickle files containing a tuple of `(values, edges)` representing the momentum-dependent reconstruction efficiency.
*   **Mechanism**: During the `theo_exp` generation, the theoretical spectrum is multiplied bin-by-bin by the efficiency histogram to create a "reconstructible" theory shape before convolution.

### D. Flat Simulation Samples (`.pkl`)
Large Awkward or NumPy arrays containing $(Gen, MC, Reco)$ momentum triplets.
*   **Usage**: Required for **Simultaneous Fits**.
*   **Mechanism**: The arrays are passed as `simul_source` to `res_components`. The `get_nll` method then constructs a secondary likelihood to "anchor" the resolution parameters to the simulation data while the primary fit is running.

---

## 7. Resolution and Loss Parameterization


###  Primary Physics Yields (`N_...`)
These represent the "extended" part of the likelihood—the total number of events attributed to each physics process in your fit range.

* `N_CE`: Number of Conversion Electron (signal) events.
* `N_Cosmic`: Number of Cosmic Ray background events.
* `N_DIO`: Number of Decay-in-Orbit background events.
* `N_RPC`: Number of Radiative Pion Capture events.

### Signal Shape Parameters (`..._CE`)

These define the Double Crystal Ball (dscb) shape of your signal.

* `mu_CE`: The peak position (mean) of the signal in MeV/c.
* `sigma_CE`: The width of the Gaussian core.
* `alphaL` / `alphaR`: The point where the "tail" begins on the Left (low energy) and Right (high energy) sides, in units of sigma.
* `nL` / `nR`: The power-law slope of the Left and Right tails (larger values mean the tail drops off faster).

### Resolution Parameters (`..._res`)
These describe the momentum resolution (reconstruction error). The numbers 0 through 4 indicate that the resolution is being measured in 5 distinct momentum bins (e.g., 95-97, 97-99 MeV/c, etc.).

* `N0_res` to `N4_res`: The number of simulated events in each momentum bin used to determine the resolution.
* `mu0_res`: The "bias" or shift in the peak for that bin.
* `sigmaL` / `sigmaR`: The widths of the asymmetric Gaussian core (Generalized Crystal Ball).
* `alpha` / `n`: The tail parameters for the resolution function in that specific bin.

### Energy Loss Parameters (`..._loss`)

These describe the Landau distribution of energy lost by the electron as it traverses the experimental material before being measured.

* `N0_loss` to `N4_loss`: The statistics used to fit the energy loss in each bin.
* `loc(0-4)_loss`: The "location" (most probable value) of the energy loss.
* `scale(0-4)_loss`: The width of the energy loss distribution (related to material thickness).

### Summary

| Prefix/Suffix | Meaning | Example |
| :--- | :--- | :--- |
| `N_` | 	Yield (Number of events)	| N_CE |
| `mu_` |	Mean / Peak Position |	`mu_CE`|
|`sigma_`|	Core Width	| `sigma_CE`|
|`loc_`	|Landau Peak (Most Probable)|	`loc0_loss` |
|`scale_`|	Landau Width	|`scale0_loss` |
| `0, 1, 2, 3, 4`|	Momentum Bin Index	|`mu2_res` |
| `_res`	|Experimental Resolution |	`alphaL0_res` |
| `_loss`	|Energy Loss Component |	`scale3_loss` |

## 8. Troubleshooting
- **`FileNotFoundError`**: Check that the paths in `mom_components.py` (e.g., `../common/...`) are relative to the execution directory of `process.py`.
- **`AttributeError: 'list' object has no attribute 'items'`**: This occurs if the loop in `fit_module.py` expects a dictionary but receives a list. The logic should check for `isinstance(sources, list)`.
- **`TypeError: cannot unpack non-iterable NoneType object`**: Occurs when `use_advanced=True` but no `simul_source` (data) was provided to the resolution objects while they were added to `nll_sources`. 
    - **Fix**: Either provide the data tuple `(gen, mc)` or modify the loop to check `if sim_data is not None`.

