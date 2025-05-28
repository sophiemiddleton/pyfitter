# $\mu^{-} \rightarrow e^{-}$ Analysis

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
* XPDF_module.py - sets of PDFs to be input into the fit module (X = mom, time)
* Xcomponent.py - user input list of chosen PDF names, parameters and draw options (X = mom, time)
* results_module.py - TODO

The latest version of the code >= v2_00_00 requires Mu2e's pyutils be within the users working directory.

# Running:

To run for example:

```
python main.py --file "/pnfs/mu2e/tape/phy-nts/nts/mu2e/ensembleMDS1dOnSpillTriggered/MDC2020ai_perfect_v1_3/root/d3/6f/nts.mu2e.ensembleMDS1dOnSpillTriggered.MDC2020ai_perfect_v1_3.0.root" --dirname "EventNtuple" --treename "ntuple" --cat 1 --mismatch 1 --fitrange_low=98. --fitrange_hi=113.
```
for a single file.

With a file list, pass the files (with full paths) to a text file and run as:

```
python main.py --file filelist.txt --dirname "EventNtuple" --treename "ntuple" --cat 1 --mismatch 1 --fitrange_low=98. --fitrange_hi=113. --singlefile 0

```

the singlefile args should be switched off for this. Eventually this will probably become default.

* The Main function imports the given root NTuple (s) via the use of Mu2e's pyutils (maintained by the Mu2e Analysis tools group). This therefore assumes input is an up-to-date EventNtuple file or list of files.

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

If the verbose option is set then arguments are printed out before running the main. The verbose arg is sent to sub-functions, and allows the user to track any failure modes. We suggest a verbose > 0 for  development users.

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

Signal (detailed below):

* **dscb** -- Double Sided Crystal Ball distribution;
* **gcb** -- generalized crystal ball
* **kde** -- kernal density estimation
* **gcb_gen_res** or **gcb_mc_res** -- use lineshape assumptions

Backgrounds

* **poly58** -- Decay in orbit (DIO) is currently parameterized using the work of Czernecki et al and the polynomial functional form derived in [Phys. Rev. D 94, 051301];
* **uniform** -- Cosmic induced background is currently parameterized as a uniform distribution;
* **Gauss** -- RPC is characterized as a Gaussian, centered on 100MeV/c following studies outline in mu2e-doc-db: 36503. Could also use uniform for the signal region we are looking at.

These distributions are all defined inside of momPDF_module.py. While the different distributions were developed to describe specific processes, this is not hardcoded in the script.

### The Time PDF Parameterizations

Our final goal is to conduct a 2D fit in momentum and time. The time component helps remove in-time RPC.

The time fit currently parameterizes things as follows:

* **muexp** -- for all muon processes (CE, DIO, RMC) these are parameterized as an exponential with a rate according to the mean lifetime in Al (864ns)
* **piexp**  -- for in time RPC
* **uniform** -- Cosmic induced background is assumed uniform in time

### 2D fits

* The 2D fit combines the momentum and time 1D fits to provide a combined momentum time fit. The individual components are parameterized in the same way as the 1D fits.

### Signal (CE) Fit Options

Susan Dittmer has carried out detailed work to parameterize the signal shape, taking into account resolution (i.e. reconstructed shape). Her work can be found in our meeting slides archive: https://drive.google.com/drive/u/0/folders/1o6gYW_gWHGtaAmZ7zZDWhDj8GhhTWddb

A number of possible signal shapes can be considered:

```
default_model_params = {'dscb'   : {'mu'     : (104,           103,   107),
                                    'sigma'  : (0.5,           0.08,  2.0),
                                    'alphaL' : (0.422,         0,     10),
                                    'nL'     : (25.1,          0,     100),
                                    'alphaR' : (2.227,         0,     100),
                                    'nR'     : (5.954,         0,     100)},
                        'gcb'    : {'mu'     : (104,           103,   107),
                                    'sigmaL' : (0.5,           0.08,  2.0),
                                    'sigmaR' : (0.5,           0.08,  2.0),
                                    'alphaL' : (0.422,         0,     10),
                                    'nL'     : (25.1,          0,     100),
                                    'alphaR' : (2.227,         0,     100),
                                    'nR'     : (5.954,         0,     100)},
                        'kde' : None,
                        'gcb_gen_res' : None,
                        'gcb_mc_res' : None,
                        }
```
Where:

* **gcb** = "Fully asymmetric Crystalball function" --> default (implicitly assumes resolution)
* **dscb** = "double sided crystal ball" (implicitly assumes resolution)

Parameters can be floated by setting the following in the components:

```
'treat_params' : 'float'
```
other ways to treat the parameters are:
* ```'fix'``` (fixed)
* ```'simul' ```(simultaneous fits)

In addition there is the option to use kernal density estimation:

* **kde** = "kernal density estimator" derived from fits to primary CeMLL sample

The latter two use the lineshape * momentum concept, indepdently fitting to extract the resolution.

* **gcb_gen_res**
* **gcb_mc_res**


# Characterizing Uncertainties

Uncertainties can appear in two forms:

* normalization/yield uncertainties effect the overal derived number of events (e.g. luminosity uncertainties)
* shape uncertainties move events around within the distribution, with the total yield staying the same e.g. uncertainty in a given theoretical description.

## Normalization uncertainties

## Shape uncertainties

# Results

The Results module store the final fit results in terms of expected yield from each of the sources of events. The results module can also print out the list of momenta or times used in the fit (passing all cuts). This can be in put into BAT.jl for Bayesian studies.

Another important goal of the "results" module is to have various statistical tests here e.g. for understanding the significance or pvalue of a result or deriving a frequentist limit in the event of low or no signal yields.

The current version of this code is underdevelopment, but the concept is taking shape. The functions work, but have not been used to produce viable results due to missing external infrastructure.

## GetSignificance (underdevelopment)

The aim of this function is to take the output of a fit and understand the p-value on the derived signal yield and the significance (in n*sigma). It will be useful for understanding if we have a discovery.

## GetUL (underdevelopment)

This function will be used to derive frequentist UL at a chosen CL. It should be used with smaller yields. There is much work to do to understand how to run with large number of toys.
