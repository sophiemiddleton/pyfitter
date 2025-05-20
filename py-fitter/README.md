# Mu2e Analysis

Python based analysis tool for analysis of reconstructed Mu2e data or MC.

# Developers

The current code base has been developed by Leo Borrel, Susan Dittmer, Sophie Middleton and Sam Zhou as part of the joint Mu2e Analysis Working Group.

# Legacy Branches

As the code base has been applied to several mock data sets over the years we have two legacy branches:

* MDC2018 branch  was developed using MDC2018 TrkAna NTuples. See the Mu2e wiki page for more information on MDC2018.
* MDS0 branch was developed using the MDS0 samples of MDC2020.

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
source /exp/mu2e/data/users/sophie/mu2e_env.v1.2.0/bin/activate 

```

# Mock Data Samples:

The code is current imagined to run using the Mu2e standard Ntuple EvtNtuple (formally trkana).

It has been tested with the latest MDC2024 mock data samples, listed here: https://mu2ewiki.fnal.gov/wiki/MDC2024:_Mock_Data#MDC_2024:_Mock_Data_samples

# The Code:

The code is currently object orientated with a set of distinct classes:

* main.py - the driver function. The user can define several input parameters, check the default settings (at the bottom of the Main.py script).
* cut_module.py - takes in an opt to a list of cuts, applies cuts
* fit_module.py - runs unbinned ML fits to input data
* recoplot_module.py - plots reconstructed information
* *PDF_module.py - sets of PDFs to be input into the fit module
* *component.py - user input list of chosen PDF names, parameters and draw options
* results_module.py - TODO

# Running:

To run for example:

```
python main.py --file "/pnfs/mu2e/tape/phy-nts/nts/mu2e/ensembleMDS1dOnSpillTriggered/MDC2020ai_perfect_v1_3/root/d3/6f/nts.mu2e.ensembleMDS1dOnSpillTriggered.MDC2020ai_perfect_v1_3.0.root" --dirname "EventNtuple" --treename "ntuple" --cat 1 --mismatch 1 --fitrange_low=98. --fitrange_hi=113.
```

* The Main function imports the given root NTuple via the ImportClass defined in import_module.py. The code currently assumes the Mu2e/EventNtuple will be an input NTuple but the user parameters allow some flexibility.

Here is a list of the current arguments and what they represent:

* file - filename (include path if not local), required
* dirname - defaults to "EventNtuple"
* treename - defaults to "ntuple"
* fittype - implented opts: "mom1D", "time1D", "momtime2D"
* fit range (low, hi) - range to fit over
* showMC - set to 1 if "MC infor is present and I want to use it to help my analysis"
* cuts - cut list to use, default is SU2020 cuts
* categorize - uses MC process code to find true nature of the particles making the tracks
* verbose - has the usual meaning, prints debug statements as desired, off by default

If the verbose option is set then arguments are printed out before running the main.

# Fitting:

## zfit

zfit is likelihood fitting code, it is very similar to the popular RooFit code base. The zft source code can be found: https://github.com/zfit/zfit.

zfit allows for custom (and predefined) -log likelihood maximizaton. Underneath it interfaces with iminuit and TensorFlow and is purely python based.

We import the Mu2e ntuples using uproot and store it as an awkward array.

## Our Fitting Interface:

The fit_module.py is our interface to zfit and the various parameterizations of the signal (CE) and backgrounds (currently DIO, RPC, and Cosmics). 1D PDFs are written for the time and momentum distributions, as well as a 2D PDF for time vs momentum (in progress).

There are three functions in the fit module:

1) Unbinned_fit_mom - a 1D unbinned fit for momentum (default)
2) Unbinned_fit_time - a 1D unbinned fit for time (probably not used on its own)
3) Unbinned_2d_fit_mom_time - 2D fit for both momentum and time

The user can specify the fit functions using the *components.py files, below is discussion of the parameters defined in these files and how to use them:

### Components specification

The signal and backgrounds considered in the fit are specified in a dictionary within components.py . This dictionary specifies the following:
* **pdf** -- PDF which describes the component; this will be one of the options described below
* **pars** -- Optional user-defined values for the PDF values and lower/upper limits in the fit. If not given, default values for the parameters will be used.
* **startCode**, **genCode**, **catColor** -- When the --categorize option is used, tracks are categorized based on the true particle type when plotting (for better comparison with fit results). The true particle type is defined by the corresponding startCode and genCode, and the component is plotted with color catColor.
* **lineColor**, **lineStyle** -- The line color and style when drawing the PDF component
  
### The Momentum PDF Parameterizations

* **dscb** -- The conversion e- signal is expected to follow a Double Sided Crystal Ball distribution, assuming there has been multiple scattering, energy losses and detector distortions;
* **poly58** -- Decay in orbit (DIO) is currently parameterized using the work of Czernecki et al and the polynomial functional form derived in [Phys. Rev. D 94, 051301];
* **uniform** -- Cosmic induced background is currently parameterized as a uniform distribution;
* **Gauss** -- RPC is characterized as a Gaussian, centered on 100MeV/c following studies outline in mu2e-doc-db: 36503.

These distributions are all defined inside of momPDF_module.py. While the different distributions were developed to describe specific processes, this is not hardcoded in the script.

### The Time PDF Parameterizations

Our final goal is to conduct a 2D fit in momentum and time. The time component helps remove in-time RPC.

The time fit currently parameterizes things as follows:

* **muexp** -- for all muon processes (CE, DIO, RMC) these are parameterized as an exponential with a rate according to the mean lifetime in Al (864ns)
* **piexp**  -- for in time RPC
* **uniform** -- Cosmic induced background is assumed uniform in time

### 2D fits

* The 2D fit combines the momentum and time 1D fits to provide a combined momentum time fit. The individual components are parameterized in the same way as the 1D fits.

### Resoluton and Efficiency parameterizations

* Further study is required to help us parameterize the momentum resolution and tracker acceptance.

# Characterizing Uncertainties

## Systematics and Nusiance Parameters

## Shape uncertainties

# Results

The Results module store the final fit results in terms of expected yield from each of the sources of events. This should be adapted to interface with our Bayesian tools eventually.
