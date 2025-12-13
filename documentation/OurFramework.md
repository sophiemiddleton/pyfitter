# 🏗️ The `py-fitter` Framework

The `py-fitter` directory contains the core analysis framework, which orchestrates the data processing, model construction, fitting, and result presentation. The code is organized into modules to maximize reusability and adhere to the separation of concerns.

## 📦 Core Execution and Data Management

These scripts manage the overall execution flow, handle data preparation, and run the main analysis loop.

| File | Description | Role in Analysis |
| :--- | :--- | :--- |
| [`process.py`](code-specifics/process_py.md) | **Main Data Processor** | Handles reading the input `EventNtuple` data, applying selection cuts via `cut_manager.py`, and preparing the data into the appropriate format for `zfit` fitting. |
| [`analyze.py`](code-specifics/analyze_py.md) | **Analysis Execution Script** | utilizes the `cut_manager.py` applies a list of simple selection cuts |
| [`cut_manager.py`](code-specifics/cutmanager_py.md) | **Selection Manager** | Contains the logic for applying sequential data selection cuts (e.g., energy, time, momentum windows) to isolate the signal region. |

## 📐 Likelihood and Model Definition

These modules are responsible for defining the specific components (PDFs) that make up the total likelihood function, using the `zfit` package.

| File | Description | Role in Analysis |
| :--- | :--- | :--- |
| [`fit_module.py`](code-specifics/fitmodule_py.md )| **Main Fitting Orchestrator** | Central module that takes the defined PDFs from the component modules, combines them into the full model (Signal + Backgrounds), defines the **Negative Log-Likelihood (NLL)** loss function, and executes the `zfit` minimization. |
| [`momPDF_module.py`](code-specifics/momPDF_module_py.md) | **Momentum PDF** | Defines the probability density functions (PDFs) for the **reconstructed momentum** observable. Includes signal and various background shapes. |
| [`timePDF_module.py`](code-specifics/TimePDF_module_py.md) | **Time PDF** | Defines the PDFs for the **reconstructed time** observable, accounting for time distribution differences between prompt and delayed components. |
| `mom_components.py` | **Momentum PDF Definitions** | Contains the explicit mathematical implementations for momentum component shapes (e.g., Crystall Ball, Gaussian, Exponential functions) used by `momPDF_module.py`. |
| `time_components.py` | **Time PDF Definitions** | Contains the explicit mathematical implementations for time component shapes (e.g., lifetime model, prompt beam shape) used by `timePDF_module.py`. |
| `landau_pdf.py` | **Specific PDF** | contains the implementation for a non-standard landau for the energy loss of a particle |
| `helper.py` | **General Utility Functions** | Contains miscellaneous functions used across multiple modules, such as common math operations, error handling, or simple data manipulation routines. |

## 📊 Results, Systematics, and Theoretical Inputs

These modules handle post-fit analysis, visualization, and the incorporation of inputs that constrain the fit.

| File | Description | Role in Analysis |
| :--- | :--- | :--- |
| `results_module.py` | **Results Handler** | Processes the `zfit` `FitResult` object, calculates final yields and confidence intervals, and formats the output. |
| `recoplot_module.py` | **Visualization** | Contains functions for generating plots of the fitted model overlayed on the data and various diagnostic plots. |
| `res_components.py` | **Detector Resolution Inputs** | Manages inputs related to the detector's finite resolution, which are crucial for modeling the observed PDFs accurately. |
| `sysunc_components.py` | **Systematic Uncertainties** | Manages the constraints and parameters related to systematic uncertainties (nuisance parameters) that are included in the overall likelihood function. |
| `theo_components.py` | **Theoretical Inputs** | Manages fixed or constrained parameters related to theoretical predictions (e.g., branching ratios, physics constants) used in the fit. |


