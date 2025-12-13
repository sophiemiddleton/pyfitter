# 📈 `results_module.py` - Analysis and Hypothesis Testing

This module defines the **`ResultsClass`**, which is responsible for taking the output of the maximum likelihood fit performed with `zfit` and performing downstream statistical analysis, primarily focusing on the search for the Conversion Electron (CE) signal. It heavily utilizes the Scikit-HEP library `hepstats` for rigorous hypothesis testing.

## 🔬 `class ResultsClass`

This class wraps the final fit result and provides methods for calculating key physics metrics, such as significance and upper limits, and handles data serialization.

### Initialization (`__init__`)

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`result`** | `zfit.FitResult` | The output object from the likelihood minimization, containing best-fit parameter values and uncertainties. |
| **`data`** | `ak.Array` | The input data array (e.g., flattened momentum values) used in the fit. |
| **`verbose`** | `int` | Controls the level of output detail during execution. |
| `rmue`, `pvalue`, `sigma` | `float` | Storage for calculated physics results (initialized to 0). |

### Method: `CalculateRmue(self, n_ce, n_dio)`

* **Purpose:** Estimates the $\text{R}_{\mu \text{e}}$ ratio (the branching ratio for $\mu^- N \to e^- N$ conversion) by normalizing the fitted signal yield $N_{\text{CE}}$ to the number of available stopped muons.
* **Formula (Text Equivalent):** The $\text{R}_{\mu \text{e}}$ ratio is proportional to:
    
$$\text{R}_{\mu \text{e}} \propto \frac{N_{\text{CE}}}{\text{Number of Stopped Muons}}$$
    
* **Status:** Marked as `#FIXME`, indicating it requires complex work to incorporate experimental efficiencies, acceptance, and normalization constants (like POT).

## 📊 Hypothesis Testing (Using `hepstats`)

### Method: `GetSignifcance(self, par, loss, opt='freq')`

* **Purpose:** Computes the statistical significance of the observed signal yield, testing the signal presence ($H_1$) against the background-only hypothesis ($H_0$: signal yield is zero).
* **Calculators (`opt`):**
    * **`'asym'` (AsymptoticCalculator):** Uses the asymptotic approximation (Wilks' theorem) for fast results.
    * **`'freq'` (FrequentistCalculator):** Uses Monte Carlo toys (simulated experiments) to calculate the $p$-value and test statistic distribution, which is typically more accurate but requires more computational resources.
* **Output:** Returns the $p$-value and the significance (in number of $\sigma$) for discovery.

### Method: `GetUL(self, par, loss, ..., CL=0.90, opt='freq')`

* **Purpose:** Calculates the **Upper Limit (UL)** on the signal yield $N_{\text{CE}}$ at a specified Confidence Level (CL) in cases where no significant signal is observed.
* **Methodology:** Uses the **$\text{CL}_{\text{s}}$ method**, standard in high-energy physics, where the limit is set by scanning signal yields and finding the point where the test statistic satisfies:

$$\text{CL}_{\text{s}} = 1 - \text{CL}$$


Note: 

$$\text{CL}_{\text{s}} = p_{\text{clsb}} / p_{\text{clb}}$$

where $p_{\text{clsb}}$ is the p-value of the signal-plus-background hypothesis, and $p_{\text{clb}}$ is the p-value of the background-only hypothesis.)

* **Input Data/Model:** The method reconstructs the combined PDF and loss function using parameters from the initial best fit, then uses the `UpperLimit` class from `hepstats`.
* **Output:** Returns the `UpperLimit` object and generates a plot showing the $\text{CL}_{\text{s}}$ scan.
***(Visualization Note: This generates the standard limit plot showing the observed $\text{CL}_{\text{s}}$ curve, the expected median, and the $\pm 1\sigma$ and $\pm 2\sigma$ uncertainty bands.)***

---
*The `plotlimit` function is included within the class definition, which renders the standard CLs plot showing the observed CLs curve, the expected median, and the $\pm 1\sigma$ and $\pm 2\sigma$ uncertainty bands.*
---

## 💾 Data Output Methods

These methods serialize the fit results and input data for external analysis or archiving.

### Method: `WriteFittedData(self, min_v, max_v)`

* **Purpose:** Writes the raw input data (e.g., filtered momentum values within the range `[min_v, max_v]`) to a CSV file (`output_data.csv`).
* **Note:** The output is intended to be in a format usable by external tools like `BAT.jl`.

### Method: `WriteResult(self)`

* **Purpose:** Exports the final best-fit values of all `zfit.Parameter` objects (Name and Value) to a CSV file (`output_fitresult.csv`).

### Method: `WritePkl(self)`

* **Purpose:** Dumps the fit result parameter metadata (names and values) into a Python pickle file (`output_fitresult.pkl`). This is useful for loading fit results in subsequent analysis steps or for comparison.
