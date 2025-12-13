# 📊 `recoplot_module.py` - Visualization and Plotting Tools

This module contains utility functions for generating histograms, overlaying Monte Carlo (MC) truth categorization, and plotting the final fit results alongside data and normalized residuals for both momentum and time spectra.

## 🎨 Functions

### 1. `plot_variable(...)`

* **Purpose:** Plots the distribution of any single reconstructed variable (e.g., `rmax`, track quality) and overlays multiple data/MC samples, splitting the data based on process code (`mc_count`).
* **Input Data:** `val_overlay` (list of reconstructed variable arrays), `mc_count` (array of process codes).
* **Visualization:** Creates a single plot with a logarithmic y-scale, stacked histograms for MC components, and draws vertical dashed lines to indicate selection cuts (`cut_lo`, `cut_hi`).
* **Output:** Saves the plot as `[filenames]_selection.pdf`.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| **`val_overlay`** | `list` of `ak.Array` | List of values (e.g., $R_{\text{max}}$) to plot, often used to compare data/MC. |
| **`val_label`** | `str` | X-axis label (e.g., "Reconstructed $R_{\text{max}}$"). |
| **`lo`, `hi`** | `float` | Lower and upper bounds for the plot range. |
| **`cut_lo`, `cut_hi`** | `float` | Values at which to draw vertical cut lines. |
| **`mc_count`** | `ak.Array` | Array containing truth process codes for categorization. |
| **`columns`** | `list` of `str` | Labels for different sets of overlayed histograms in the legend. |
| **`density`** | `bool` | If `True`, the histogram is normalized to form a probability density. |

### 2. `plotmom_fit(mom_mag, mc_count, fit_range, list_pdfs, cat=None)`

* **Purpose:** Draws the momentum spectrum histogram with the combined fit result overlaid. It includes a sub-plot showing normalized residuals.
* **Structure:** Uses a `matplotlib` figure with two subplots: the main histogram (top, 3:1 ratio) and the residuals (bottom, 1:1 ratio). 
* **MC Categorization:** If `cat` is enabled, it separates the input data (`mom_mag`) into true physics processes (Signal, DIO, Cosmic, RPC, etc.) based on `mc_count` and displays them as a stacked histogram.
* **Fit Overlay:** Plots each individual PDF component (from `list_pdfs`) and the final total fit (`'Total'`) curve against the binned data.
* **Residuals Plot (`ax2`):** Plots the normalized residuals, calculated as (Data - Fit) / $\sqrt{\text{Data}}$, showing the pull of the fit from the observed data points.

### 3. `plottime_fit(time, mc_count, fit_range, list_pdfs, cat=None)`

* **Purpose:** Draws the time-of-arrival spectrum histogram with the combined fit result overlaid, including a sub-plot showing normalized residuals.
* **Structure:** Functionally identical to `plotmom_fit`, but tailored for time distribution.
* **Fit Overlay:** Plots each time PDF component (using colors/styles from `time_components`) and the total fit curve.
* **Residuals Plot:** Plots the normalized residuals for the time fit.

### 4. `plotmom_fit_old(...)` and `plot_time_fit_old(...)` **Deprecated**

* **Purpose:** Older implementations of the momentum and time fit plotting functions.
* **Note:** These versions use a different method for categorizing MC truth (`track_cats`) and iterating over the components, but serve the same ultimate purpose as the primary `plotmom_fit` and `plottime_fit` functions. They are likely retained for backward compatibility or debugging.
