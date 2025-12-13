# Results and Interpretation

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

