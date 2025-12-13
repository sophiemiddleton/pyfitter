# Getting Started

**Python-based analysis tool** for analyzing reconstructed Mu2e data or Monte Carlo (MC) simulations, leveraging the `EventNtuple` framework.

## Prerequisites

This tool requires **Python 3.8+** and several external libraries managed via `pip`. It is also dependent on the `pyutils` package developed by the Mu2e analysis tools group.

The current version of PyFitter is compatible with the current version of the Mu2e specific `pyutils` package.

## Building the Python Environment

You have two primary options for setting up the required environment: on a personal device using a virtual environment, or using the standard environment provided on the Mu2e GpVMs.

### On Your Own Device

To ensure reproducibility, we use a standard set of Python packages defined in the `requirements/current.txt` file. Using a virtual environment is highly recommended:


# 1. Create a new virtual environment
```
virtualenv myzfitenv
```

# 2. Activate the environment
```
source myzfitenv/bin/activatehttps://mu2ewiki.fnal.gov/wiki/Mock_Data_(MDS)#MDC2025_ensembles
```

# 3. Install core dependencies from the requirements file
```
pip install -r requirements/current.txt
```

# 4. Install `pyutils`:

```
pip install git+https://github.com/Mu2e/pyutils.git
```

## On the Mu2e gpvm's:

We have all our package dependecies installed in the standard Mu2e python environment

```
source /cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/activate

```

## Mock Data

The code is designed to run using the Mu2e standard Ntuple framework (EventNtuple). 

Please see: https://mu2ewiki.fnal.gov/wiki/Mock_Data_(MDS)#MDC2025_ensembles

The input to the `py-fitter` interface is a text file list of EventNtupled Mock Data.
