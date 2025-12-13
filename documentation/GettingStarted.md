# 🚀 Getting Started

`py-fitter` is a custom **Python-based physics analysis tool** designed for fitting and interpreting reconstructed Mu2e data and Monte Carlo (MC) simulations.

## Philosophy

Our analysis framework is built to integrate seamlessly with the standard tools developed by the Mu2e Analysis Tools Group (led by Andy Edmonds and Sophie Middleton). We leverage the established **`EventNtuple`** data structure and the **`pyutils`** package for common utilities. The code is compatible with the standard mu2e **`pyana`** python environment and can be ran on the **Elastic Analysis Facility**.

## Environment Setup

There are two primary options for setting up the required analysis environment.

### Option 1: Mu2e Supported Machines (Recommended)

Running on the Mu2e General Purpose Virtual Machines (GpVMs) or the Elastic Analysis Facility is the **recommended** approach. This offers a centrally maintained, pre-installed Python environment and access to the vast computing resources of the Elastic Analysis Facility.

All required Python package dependencies are pre-installed in the standard Mu2e Python environment. You can activate it using one of the following methods:

**Method A: Direct Source**

```
source /cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/activate
```
**Method B: Mu2e Aliases**

```
mu2einit
pyenv ana
```

### Option 2: Local Development Environment

If you prefer to develop on a personal machine, the code can be run locally as it does not directly depend on the Mu2e source code (only standard Python packages).

#### Prerequisites

* **Python 3.8+**
* **`pip`** (Python package installer)
* **`pyutils`** package (from the Mu2e Analysis Tools Group)

To ensure a clean and reproducible environment, using a virtual environment (`virtualenv` or similar) is highly recommended.

#### Installation Steps

**1. Create and Activate a Virtual Environment**

```
# Create a new virtual environment
virtualenv myzfitenv

# Activate the environment
source myzfitenv/bin/activate
```

**2. Install Core Dependencies**

The required third-party Python packages are defined in the `requirements/current.txt` file.

```
pip install -r requirements/current.txt
```

**3. Install `pyutils`**

Install the Mu2e-specific utility package directly from the repository:

```
pip install git+https://github.com/Mu2e/pyutils.git
```

> **Note:** The current version of PyFitter is compatible with the current version of the Mu2e `pyutils` package.

# Mock Data

This analysis code is designed to process data formatted according to the Mu2e standard Ntuple framework (EventNtuple).

In the absence of real physics data, development and testing are performed using the simulated Mock Data (MDS). More details regarding the available data ensembles (e.g., MDC2025) can be found on the Mu2e Wiki.

## Running  in `py-fitter`

Before running the fitting code you should familarize yourself with the remaining documentation. For a quick start:

### 1. **Get Mock Data** current mock data samples for MDS2c (nominal Mu2e Geometry, with MDC2020 assumptions):

```

ls /pnfs/mu2e/persistent/users/mu2epro/ensembles/MDS2c/1e-13_1month_10exps/merged_files_1/*.root &> MDS2c_1e-13_1month_exp1.txt
```

these are examples of up to 10 random samples for 1month of data with a $R_{\mu e} = 1 \times 10^{-13}$. The `i` in the `merged_files_i` represents the experiment number.

### 2. ** Run `py-fitter`

To run `py-fitter` in default: 

```
python process.py --file MDS2c_1e-13_1month_exp1.txt --loc "local"

```

where the `.txt` file is the file list you made above and `--loc` tells the processor to look locally.
