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
| [`momentum_pdf_builder.py`](code-specifics/momPDF_module_py.md) | **Momentum & Time PDF Builders** | Consolidated module defining probability density functions (PDFs) for both **reconstructed momentum** and **time** observables. Includes signal and background shapes with custom models. |
| [`physics_components.py`](code-specifics/dictionaries.md) | **Physics Component Definitions** | Pure configuration file containing momentum and time component dictionaries with parameters, line styles, and configuration for all physics processes. |
| [`custom_models.py`](code-specifics/dictionaries.md) | **Custom Physics Models** | Consolidated implementation module containing: `trunc_landau` PDF for energy loss, spectrum calculation functions (`LeadingLog`, `binned_spectrum_CeLL`), and `res_components` class for detector resolution. |
| [`sysunc_components.py`](code-specifics/dictionaries.md) | **Systematic Uncertainties** | Manages the constraints and parameters related to systematic uncertainties (nuisance parameters) that are included in the overall likelihood function. |
| `helper.py` | **Utility Functions** | Core utilities for lineshape loading, convolution, PDF generation, and data operations. |
| `data_prep.py` | **Data Preparation Manager** | Centralized safe data cleaning, conversion, and validation for awkward arrays and zfit. |

## 📊 Results, Systematics, and Theoretical Inputs

These modules handle post-fit analysis, visualization, and the incorporation of inputs that constrain the fit.

| File | Description | Role in Analysis |
| :--- | :--- | :--- |
| [`results_module.py`](code-specifics/results_module_py.md) | **Results Handler** | Processes the `zfit` `FitResult` object, calculates final yields and confidence intervals, and formats the output. |
| [`recoplot_module.py`](code-specifics/recoplot_module_py.md) | **Visualization** | Contains functions for generating plots of the fitted model overlayed on the data and various diagnostic plots. |



