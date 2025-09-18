# $\mu^{-} \rightarrow e^{-}$ Analysis

Python based analysis tool for analysis of reconstructed Mu2e data or MC.

# Developers

The current code base has been developed by Leo Borrel, Susan Dittmer, Sophie Middleton and Sam Zhou as part of the joint Mu2e Analysis Working Group.

# Legacy Branches

As the code base has been applied to several mock data sets over the years we have two legacy branches:

* MDC2018 branch  was developed using MDC2018 TrkAna NTuples. See the Mu2e wiki page for more information on MDC2018.
* MDS0 branch was developed using the MDS0 samples of MDC2020.

# Building the python environment:

## ```pyutils```

In v2 onwards the code is dependent on pyutils, the python interface to EventNtuple developed by the Mu2e analysis tools group. This should be included in the python environment or you can include it on your own device using the instructions below.

The current pyfitter is compatible with v01_01_00 of pyutils.

## On your own device:

In order to ensure reproducibility, we have a standard set of python packages which should be used with each intall. This is stored in the requirements/current.txt

On your own device you can use pip (or other means) to produce a virtual environment for your work:

```
$ virtualenv myzfitenv
$ source myzfitenv/bin/activate
(myzfitenv)$ pip install -r requirements/current.txt
```

In addition to work with the latest pyutils:

```
pip install hist 
pip install tqdm 
pip install git+https://github.com/Mu2e/pyutils.git 
```

## On the Mu2e gpvm's:

We have all our package dependecies installed in the standard Mu2e python environment

```
source /cvmfs/mu2e.opensciencegrid.org/env/ana/current/bin/activate


```

# Mock Data Samples:

The code is current imagined to run using the Mu2e standard Ntuple framework EventNtuple and will continue to assume that.

It has been tested with the latest MDC2020 MDS mock data samples, listed here: https://mu2ewiki.fnal.gov/wiki/MDC2024:_Mock_Data#MDC_2024:_Mock_Data_samples.

# The Code:

The code is currently object orientated with a set of distinct classes:

* `process.py` and `analyze.py`- the driver functions. The user can define several input parameters, check the default settings (at the bottom of the Main.py script).
* `cut_module.py` - takes in an opt to a list of cuts, applies cuts
* `fit_module.py` - runs unbinned ML fits to input data
* `recoplot_module.py` - plots reconstructed information
* `_PDF_module.py` - sets of PDFs to be input into the fit module ( mom, time)
* `_component.py` - user input list of chosen PDF names, parameters and draw options (mom, time)
* `results_module.py`- runs functions to interpret the result (e.g. significance tests and limit setting)

The latest version of the code >= v2_00_00 requires Mu2e's pyutils be within the users working directory.

# Running:

To run for example:

```
python process.py --file /exp/mu2e/app/users/sophie/analysis/LikelihoodAnalysis/py-fitter/filelist.txt --cat 1 --mismatch 1 --interpret 1
```

With a file list stored in the mentioned .txt file.

* The process function imports the given root NTuple (s) via the use of Mu2e's pyutils (maintained by the Mu2e Analysis tools group). This therefore assumes input is an up-to-date pyuyils. To get pyuyils in your python env:

```
pip install git+https://github.com/Mu2e/pyutils.git 

```

this is already installed in the standard python env mentioned above.

# Main Code Documentation

## `process.py`

The user runs the analysis through the process python code. There are a number of input arguments that the user uses to control the analysis.

The process module contains a class AnaProcessor which inherits from the Skeleton ```pyutils/pyproces.py```. For more information see the pyutils documentation.



<details>
<summary>For our specific instance the class is detailed here</summary>
    
```
NAME
    process.py
MODULE
      process.py
     |      User input args to module:
     |          * file - filename (include path if not local), required
     |           * jobs - should be the same as number of files to be as optimal as possible (number of worker threads for import)
     |           * fittype - implented opts: "mom1D", "time1D", "momtime2D"
     |           * fit range (low, hi) - range to fit over
     |           * cat - uses MC process code to find true nature of the particles making the tracks
     |           * mismatch - a work around (FIXME)
     |           * verbose - has the usual meaning, prints debug statements as desired, off by default
     |           * interpret - will look for pvalue and signficance
     |           * setlimit  - assumes small or no signal and will try to set limit
CLASSES
    builtins.object
        AnaProcessor inherits from pyprocess Skeleton class
     |
     |  process_file(): 
     |     Process a single ROOT file
     |     This method will be called for each file in our list.
     |    It extracts data, processes it, and returns a result.
     |     Args:
     |         file_name: Path to the ROOT file to process   
     |     Returns:
     |         A tuple containing the histogram (counts and bin edges)
     |  
     | combine_arrays():
     |  Combine filtered arrays from multiple files
     |  Args:
     |    results: list of returned filtered data
     |  Returns:
     |    concatanted results array
     |  ----------------------------------------------------------------------
```

</details>

---

## `analyse.py`

The user interacts with the analyze via the processor class. The analyze module contains 

<details>
<summary>For our specific instance the class is detailed here</summary>
    
```
NAME
    analyse.py
CLASSES
    builtins.object
        analyse.
     |      Class to handle analysis functions
     |        Args:
     |          * verbosity = verbosity level
```

</details>

---

# `fit_module.py`:

## zfit

zfit is likelihood fitting code, it is very similar to the popular RooFit code base. The zft source code can be found: https://github.com/zfit/zfit.

zfit allows for custom (and predefined) -log likelihood maximizaton. Underneath it interfaces with iminuit and TensorFlow and is purely python based.

We import the Mu2e ntuples using uproot and store it as an awkward array.

## Our Fitting Interface:

The ```fit_module.py``` is our interface to zfit and the various parameterizations of the signal (CE) and backgrounds (currently DIO, RPC, and Cosmics). 1D PDFs are written for the time and momentum distributions, as well as a 2D PDF for time vs momentum (in progress).

There are three functions in the fit module:

1) Unbinned_fit_mom - a 1D unbinned fit for momentum (default)
2) Unbinned_fit_time - a 1D unbinned fit for time (under development)
3) Unbinned_2d_fit_mom_time - 2D fit for both momentum and time (under development)

The user can specify the fit functions using the *components.py files, below is discussion of the parameters defined in these files and how to use them:

### `_components.py`

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

### Signal Momentum (CE) Shape Characteristics

Detailed work has been carried out to parameterize the signal shape, taking into account resolution (i.e. reconstructed shape). Her work can be found in our meeting slides archive: https://drive.google.com/drive/u/0/folders/12jnMJh-Hg7eg-WNqawPMq2lZ15e9xwQB

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

The latter two use the lineshape convoluted with momentum concept, indepdently fitting to extract the resolution:

* **gcb_gen_res**
* **gcb_mc_res**

Full definitions are provided in https://drive.google.com/drive/u/0/folders/12jnMJh-Hg7eg-WNqawPMq2lZ15e9xwQB.

### DIO Momentum Shape Characteristics

The DIO shape is a convolution of the theoretical DIO spectrum taken from https://arxiv.org/abs/1505.05237 and doc-db 6309 with an efficiency and resolution parameterization derived from flat spectra.


<details>
<summary>The efficiency and resolution are included optionally in the momPDF module</summary>
    
```
elif model == 'poly58':
        if dio_resolution is not None:
            if dio_efficiency is None:
                raise Exception("ERROR: dio_resolution can only be used if dio_efficiency is also defined")
            else: # Both efficiency and resolution are defined
                # Load the PDFs
                efficiency_pdf = _load_pdf(dio_efficiency)
                resolution_pdf = _load_pdf(dio_resolution)

                # Adjust the PDFs
                efficiency_pdf = efficiency_pdf.to_truncated(obs=obs_mom)
                resolution_pdf = resolution_pdf.copy(obs=zfit.Space('mom', limits=(-8, 1)))

                # Multiply the efficiency PDF with the poly58 PDF and convolve with the resolution PDF
                poly58_pdf = poly58(obs=obs_mom, a5=zpars['a5'], a6=zpars['a6'], a7=zpars['a7'], a8=zpars['a8'])
                poly58_efficiency_product = zfit.pdf.ProductPDF([poly58_pdf, efficiency_pdf])
                PDF = zfit.pdf.FFTConvPDFV1(poly58_efficiency_product, resolution_pdf, obs=obs_mom, extended=N, n=1000)
        else: 
            if dio_efficiency is not None: # just efficiency, no resolution
                # Load the efficiency PDF
                efficiency_pdf = _load_pdf(dio_efficiency)

                # Adjust the efficiency PDF to the observation space
                efficiency_pdf = efficiency_pdf.to_truncated(obs=obs_mom)

                # Multiply the efficiency PDF with the poly58 PDF
                poly58_pdf = poly58(obs=obs_mom, a5=zpars['a5'], a6=zpars['a6'], a7=zpars['a7'], a8=zpars['a8'])
                PDF = zfit.pdf.ProductPDF([poly58_pdf, efficiency_pdf], extended=N)
            else: # no resolution or efficiency, just poly58 PDF
                PDF = poly58(obs=obs_mom, a5=zpars['a5'], a6=zpars['a6'], a7=zpars['a7'], a8=zpars['a8'], extended=N)
```

</details>

---



# Characterizing Uncertainties

Uncertainties can appear in two forms:

* normalization/yield uncertainties effect the overal derived number of events (e.g. luminosity uncertainties)
* shape uncertainties move events around within the distribution, with the total yield staying the same e.g. uncertainty in a given theoretical description.

## Normalization uncertainties (underdevelopment)

## Shape uncertainties (underdevelopment)

# `results_module.py`

The Results module store the final fit results in terms of expected yield from each of the sources of events. The results module can also print out the list of momenta or times used in the fit (passing all cuts). This can be in put into BAT.jl for Bayesian studies.

Another important goal of the "results" module is to have various statistical tests here e.g. for understanding the significance or pvalue of a result or deriving a frequentist limit in the event of low or no signal yields.

The current version of this code is underdevelopment, but the concept is taking shape. The functions work, but have not been used to produce viable results due to missing external infrastructure.

## ```GetSignificance```

The aim of this function is to take the output of a fit and understand the p-value on the derived signal yield and the significance (in n*sigma). It will be useful for understanding if we have a discovery.

The code uses the hepstats package ```hepstats.hypotests```

First it computes a null hypothesis, using the fit parameters and assuming 0 signal yield:

```
    # the null hypothesis
    sig_yield_poi = POI(par, 0)
    minimizer = zfit.minimize.Minuit()
```

and the builds the chosen calculator from the zfit loss function where the self.result is the result of the combined fit:

```  
    if opt == 'freq':
      calculator = FrequentistCalculator(input=loss, minimizer=minimizer)
      calculator.bestfit = self.result
      calculator = FrequentistCalculator(input=self.result, minimizer=minimizer)
    elif opt == 'asym':
      calculator = AsymptoticCalculator(input=loss, minimizer=minimizer)
      calculator.bestfit = self.result
      calculator = AsymptoticCalculator(input=self.result, minimizer=minimizer)
```

The 'freq' option uses a full frequentist procedure for sampling the test statistic distribution whereas the 'asym' generates the Asimov histogram using a model and dictionary of parameters (uses  Eur. Phys. J., C71:1–19, 2011).

The asympotic formula is significantly faster than the Frequentist calculator, as it does not require the calculation of the frequentist p-value, which involves the calculation of toys 


The significance is calculated using the Discovery class:

```
  discovery = Discovery(calculator=calculator, poinull=sig_yield_poi)
  significance = discovery.result()
```
The output is in units of sigma.


## ```GetUL```

In the event that we have small/no signal we may want to derive an upper limit on the signal yield. The GetUL function performs this task using the hepstats hypotests package.

The GetUL function has a number of parameters:

```
    Parameters
    ----------
      par : zfit parameters
      loss : zfit loss function
      combine_pdf: zfit combined pdf
      fitlow, fithigh : fit range
      sig_yield : observed CEs from fit
      CL : confidence level for limit default is 90%
      opt : option for how to compute (either frequentist (freq) or asymptotic (asym)
```

The function proceeds as follows.

First the parameter of interest for the null hypothesis is taken from the derived signal yield (input)

```
    sig_yield_poi = POI(par, 0)
    minimizer = zfit.minimize.Minuit()
    # Sets the values of the parameters to the self.result of the simultaneous fit
    zfit.param.set_values(loss.get_params(), self.result)
```
A sampler is created to  sample the combined pdf (input)
```
    # Creates a sampler that will draw events from the model
    sampler = combine_pdf.create_sampler()
```

The loss is computed and the resampler samples with sig_yield. Since the model is extended the number of signal generated is drawn from a poisson distribution with lambda = sig_yield.

```
    sampler.resample({par: sig_yield})
```

Then calculators are called with the low signal nlls as inputs and the discovery significance calculated

Then we look at the background only hypothesis:

``` 
    #Background only hypothesis.
    bkg_only = POI(par, 0)
    # Range of Nsig values to scan.
    sig_yield_scan = POIarray(par, np.linspace(0,570,550))#FIXME - hardcoded

    ul = UpperLimit(calculator=calculator_low_sig, poinull=sig_yield_scan, poialt=bkg_only)
    ul.upperlimit(alpha=1-CL);

```
Further development of this limit setting interface is to be done with MDS2.


## Passing to BAT.jl

The ```WriteOuput``` function results in a text file format of the post-cut momentum and can be input into BAT.jl. We also write out the fit parameters using the ```WriteResult``` function.

