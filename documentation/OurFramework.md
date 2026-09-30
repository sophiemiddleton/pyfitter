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

## 🧪 Component Fit Modules (Testing & Diagnostics)

The [`component_fits/`](../../component_fits/) directory contains **standalone unbinned maximum likelihood fitters** for testing individual physics component shapes and conducting focused, component-specific studies.

| File | Component | Observable | PDF Model | Role |
| :--- | :--- | :--- | :--- | :--- |
| [`ce.py`](code-specifics/component_fits_py.md#1-cepy--conversion-electron-ce-fitter) | Conversion Electron | Momentum | Double-Sided Crystal Ball | Signal shape validation and optimization |
| [`rmc.py`](code-specifics/component_fits_py.md#2-rmcpy--radiative-muon-capture-rmc-fitter) | RMC Background | Momentum | GammaPolyHybrid (custom) | Background shape validation |
| [`dio.py`](code-specifics/component_fits_py.md#3-diopy--dielectric-interaction-dio-fitter) | DIO Background | Momentum | poly58 (physics-motivated) | Physics-based background fitting |
| [`cosmics.py`](code-specifics/component_fits_py.md#4-cosmicspy--cosmic-background-fitter) | Cosmic Background | Time-of-Arrival | Chebyshev Polynomial | Control region studies |
| [`rpc.py`](code-specifics/component_fits_py.md#5-rpcpy--rpc-background-fitter) | RPC Background | Time-of-Arrival | Exponential | Control region studies |

These modules are **independent of the main likelihood chain** and are used for:
- **Shape validation** against Monte Carlo expectations
- **Parameter optimization** for PDF definitions
- **Control region fits** in sideband regions
- **Systematic studies** of individual components
- **Publication plots** with diagnostic overlays

See [**`component_fits/` Documentation**](code-specifics/component_fits_py.md) for detailed API and usage examples.



