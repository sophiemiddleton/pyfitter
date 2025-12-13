# Getting Started

`py-fitter` is a custom **Python-based physics analysis tool** for analyzing reconstructed Mu2e data or mock data (MC).
 
Our philosophy is to utilize Mu2e's standard analysis tools devloped by the Mu2e Analysis Tools Group led by Andy Edmonds and Sophie Middleton, leveraging `EventNtuple` and `pyutils` framework.

To get started there are two primary options for setting up the required environment: 

* (**recommended**) using the standard environment provided on the Mu2e GpVMs or elastic analysis facility;
* on a personal device using a virtual environment. 

## On the Mu2e Machines or the Elastic Analysis Facility

All required python package dependecies are installed in the standard Mu2e python environment that can be sourced directly:


```
source /cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/activate

```
or 

```
mu2einit
pyenv ana
```

Running on the Mu2e supported machines has the advantage of the ready installed python enviroment, which is centrally maintained by the Mu2e Analysis Tools group, and the opportunity to leverage the vast resources of the Elastic Analysis Facility. 

## Building the Python Environment On Your Own Device

We appreciate users may wish to develop locally, since the code does not depend on any mu2e source code (only python), this is possible:

### Prerequisites

This tool requires **Python 3.8+** and several external libraries managed via `pip`. It is also dependent on the `pyutils` package developed by the Mu2e analysis tools group.

The current version of PyFitter is compatible with the current version of the Mu2e specific `pyutils` package.




To ensure reproducibility, we use a standard set of Python packages defined in the `requirements/current.txt` file. Using a virtual environment is highly recommended:


#### 1. Create a new virtual environment
```
virtualenv myzfitenv
```

#### 2. Activate the environment
```
source myzfitenv/bin/activatehttps://mu2ewiki.fnal.gov/wiki/Mock_Data_(MDS)#MDC2025_ensembles
```

#### 3. Install core dependencies from the requirements file
```
pip install -r requirements/current.txt
```

#### 4. Install `pyutils`:

```
pip install git+https://github.com/Mu2e/pyutils.git
```


## Mock Data

The code is designed to run using the Mu2e standard Ntuple framework (EventNtuple). 

In the absence of real physics data development is currently taking place using the simulated Mock Data. More details can be found: https://mu2ewiki.fnal.gov/wiki/Mock_Data_(MDS)#MDC2025_ensembles.


