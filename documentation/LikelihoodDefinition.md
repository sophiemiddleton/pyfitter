# Defining our Likelihood

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

