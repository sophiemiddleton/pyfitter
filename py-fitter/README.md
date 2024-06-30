# Mu2e Analysis

Python based analysis tool for analysis of reconstructed Mu2e data or MC.

# Developers

The current code base has been initially developed by Leo Borrel and Sophie Middleton. Moving forward it will be part of the joint Mu2e Analysis Working Group. The current members include: Caltech, LBNL, Berkeley, Northwestern and Andy Edmonds (from CUNY but acting in on his own).

# Legacy Branch: MDC2018

The MDC2018 branch  was developed using MDC2018 TrkAna NTuples. See the Mu2e wiki page for more information on MDC2018.

# Building the python environment:

## On your own device:

If you do want to do this you can simple pull the "requirements.txt":

```
$ virtualenv myzfitenv
$ source myzfitenv/bin/activate
(myzfitenv)$ pip install -r path/to/requirements.txt
```

## On the Mu2e gpvm's:

We have implemented a venv, with version control on the al9 based mu2egpvm's. To activate it:

```
source /exp/mu2e/app/users/sophie/pyana-tests/zfit-env-v1/bin/activate

```

# Mock Data Samples:

The code is current imagined to run using the Mu2e standard Ntuple EvtNtuple (formally trkana).

It has been tested with the latest MDC2024 mock data samples, listed here: https://mu2ewiki.fnal.gov/wiki/MDC2024:_Mock_Data#MDC_2024:_Mock_Data_samples

# Running the Code:

The code is currently object orientated with a set of distinct classes:

* Main.py - the driver function. The user can define several input parameters, check the default settings (at the bottom of the Main.py script). 

To run for example:

```
python Main.py --filelist "MDS0a.root" --treename "TrkAna" --branchname "trkana"
```

* The Main function imports the given root NTuple via the ImportClass defined in Import_module.py. The code currently assumes the Mu2e/TrkAna will be in input NTuple but the user parameters allow some flexibility.

# The Fitting Code:

## zfit

zfit is likelihood fitting code, it is very similar to the popular RooFit code base. The zft source code can be found: https://github.com/zfit/zfit.

zfit allows for custom (and predefined) -log likelihood maximizaton. Underneath it interfaces with iminuit and TensorFlow and is purely python based.

We import the Mu2e ntuples using uproot and awkward array.

## Our interface:


The Fit_module.py calls zfit and the various parameterizations of the signal (CE) and backgrounds (currently DIO, Cosmics only). The PDFs are written only for the momentum parameter currently.
We plan to expand to a 2D momentum and time fit eventually.

### The Momentum PDF Parameterizations

* conversion e- signal (CE) is currently parameterized as a Double Sided Crystal Ball, assuming there has been multiple scattering, energy losses and detector distortions;
* decay in orbit (DIO) is currently parameterized using the work of Czernecki et al and the polynomial functional form derived in [Phys. Rev. D 94, 051301];
* cosmic induced background is currently parameterized as a uniform distribution;
* RPC will be characterized as a Gaussian, centered on 100MeV/c.

These distributions are all defined inside of the "Mom_PDF" script. This infrastructure needs to evolve as we develop more complexity.

### The Time PDF Parameterizations

* TODO

### 2D fits

* TODO

### Resoluton and Efficency parameterizations

* TODO

# Characterizing Uncertainties

## Systematics and Nusiance Parameters

* TODO

## Shape uncertainties

* TODO

# Results

The Results module store the final fit results in terms of expected yield from each of the sources of events. This should be adapted to interface with our Bayesian tools eventually.
