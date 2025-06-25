# Define custom PDF for momentum

import numpy as np
import tensorflow as tf
import zfit
import math
import hist as hist

class poly58(zfit.pdf.ZPDF):
    """ for DIO parameterization """
    _N_OBS = 1
    _PARAMS = ['a5', 'a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x) # convert the zfit Data object into a tf tensor
        a5 = self.params['a5']
        a6 = self.params['a6']
        a7 = self.params['a7']
        a8 = self.params['a8']
        E_mu = 105.194 # mass of the muon [MeV]
        m_Al = 25133 # mass of the Aluminum atom [MeV]
        delta = tf.nn.relu(E_mu - x - x**2 / (2 * m_Al))    # Use relu function to make pdf = 0 for x > E_mu

        return a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8

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
                        'poly58' : {'a5'     : (8.97879e-17,    0,     1e-16),
                                    'a6'     : (1.17169e-17,   0,     1e-16),
                                    'a7'     : (-1.06599e-19, -1e-18, 0),
                                    'a8'     : (8.14251e-20,   0,     1e-19)},
                        'Gauss'  : {'mu'     : (100,           95,    115),
                                    'sigma'  : (0.5,           1e-3,  1e3)},
                        'uniform' : {}
                        }

default_norms = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 200, 'RPC' : 1} #FIXME - should we make these relative?

def MomModel(obs_mom, params_tot, process, model, pardict, treat_params, fit_range, constraints):
    """ 
    Build momentum fit model
      obs_mom = zfit parameter for reco momentum
      params_tot = list of fit parameters
      process = type of physics process
      model = fit model
      pardict = dictionary of parameters
      trat_params = fixed or float, defined in componets script
      fit_range = min, max to fit over
      constraints= parameter specific constraints
    """
    if isinstance(pardict,dict) and 'N' in pardict:
        N = zfit.Parameter('N_'+process, pardict['N'][0], pardict['N'][1], pardict['N'][2])
    elif process in list(default_norms.keys()):
        N = zfit.Parameter('N_'+process, default_norms[process], 0, 1e6)
    else:
        N = zfit.Parameter('N_'+process, 10,                     0, 1e6)
    params_tot.append(N)

    # Start with default parameters for model
    params_ini = default_model_params[model]
    # If any parameters are specified in mom_components.py, override
    if pardict is not None:
        if params_ini is None:
            params_ini = pardict
        else:
            for par in pardict.keys(): 
                if par in params_ini.keys() : params_ini[par] = pardict[par]
    zpars = {}

    if isinstance(params_ini,dict) and params_ini:
        lineshape = params_ini.pop('lineshape',None) # lineshape X resolution fits

        # If input parameters is a dict of dicts (ex. for p-binned resolution) merge the values
        # Parameters should have unique names -- none should be lost
        params = {}
        if isinstance(list(params_ini.values())[0],dict):
            for pardict in params_ini.values():
                params.update(pardict)
        else:
            params = params_ini

        for p in params.keys():
            if treat_params == 'constrain':
                zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][0]+5*params[p][1], params[p][0]+5*params[p][2],step_size=0.0001)
                params_tot.append(zpars[p])
                constraints.append(zfit.constraint.GaussianConstraint(zpars[p],observation=params[p][0],uncertainty=max(abs(params[p][1]),abs(params[p][2]))))
            elif treat_params == 'fix':
                zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][0]-0.005, params[p][0]+0.005, floating=False)
            elif treat_params == 'simul':
                zpars[p] = zfit.ComposedParameter(p+'_'+process, lambda x : 1*x, params=params[p])
            else:
                if treat_params != 'float':
                    print("Supported values for treat_params are 'float','fix','constrain', or 'simul'. You are using {} -- will be treated as 'float'")
                zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][1], params[p][2])
                params_tot.append(zpars[p])

    # For KDE, pass input data in place of params
    else: 
        params = params_ini

    if model == 'dscb': # Crystalball function with asymmetric tails
        PDF = zfit.pdf.DoubleCB(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], alphal=zpars['alphaL'], nl=zpars['nL'], alphar=zpars['alphaR'], nr=zpars['nR'], extended=N)

    elif model == 'gcb': # Fully asymmetric Crystalball function
        PDF = zfit.pdf.GeneralizedCB(obs=obs_mom, mu=zpars['mu'], sigmal=zpars['sigmaL'], sigmar=zpars['sigmaR'], alphal=zpars['alphaL'], nl=zpars['nL'], alphar=zpars['alphaR'], nr=zpars['nR'], extended=N)

    elif model == 'kde': # Kernel Density Estimator
        data = zfit.Data.from_numpy(array=params, obs=obs_mom)
        PDF = zfit.pdf.KDE1DimGrid(data, num_grid_points=256, binning_method='linear', extended=N, bandwidth='adaptive_zfit')
        
    elif model == 'gcb_gen_res' or model == 'gcb_mc_res':
        if lineshape is None:
            raise Exception("ERROR: '*_res' models can only be used if 'lineshape' is defined in parameters dict")

        binwidth_eval = 0.1
        
        # Lineshape is either an array of momentum values or a true lineshape e.g. list of (p,pdf) values.
        # In either case this will be used to fill a histogram -- pdf used as weights for the latter case
        # TODO for true lineshape, check that binning is appropriate / values are filled in correct bins
        if isinstance(lineshape[0],tuple):
            gen_mom = np.array([l[0] for l in lineshape])
            weights = np.array([l[1] for l in lineshape])
        else:
            gen_mom = np.array(lineshape)
            weights = np.ones_like(gen_mom)
        
        obs_res  = zfit.Space('x',-10,10) if 'gen_res' in model else zfit.Space('x',-1,1)
        obs_gen  = zfit.Space('x',math.floor(max(float(np.min(gen_mom)),fit_range[0]-float(obs_res.v1.upper))),math.ceil(min(float(np.max(gen_mom)),fit_range[1]-float(obs_res.v1.lower))))
        obs_full = zfit.Space('x',float(obs_gen.v1.lower+obs_res.v1.lower),float(obs_gen.v1.upper+obs_res.v1.upper))
        nbins_gen = int((obs_gen.v1.upper - obs_gen.v1.lower)/binwidth_eval)
        nbins_res = int((obs_res.v1.upper - obs_res.v1.lower)/binwidth_eval)

        fracs = []
        pdfs = []
        
        for ip,pbin in enumerate(params_ini.keys()):
            # Get lineshape part
            plow  = pbin[0] if ip != 0                        else float(obs_gen.v1.lower)
            phigh = pbin[1] if ip != len(params_ini.keys())-1 else float(obs_gen.v1.upper)
            mask = (gen_mom >= plow) & (gen_mom < phigh)
            gen_slice = gen_mom[mask]
            w_slice = weights[mask]
            if len(gen_slice) == 0: continue
            h_gen_slice = hist.Hist(hist.axis.Regular(bins=nbins_gen, start=obs_gen.v1.lower, stop=obs_gen.v1.upper, name="x"))
            h_gen_slice.fill(x=gen_slice,weight=w_slice)
            lineshape_pdf = zfit.pdf.UnbinnedFromBinnedPDF(zfit.pdf.HistogramPDF(h_gen_slice),obs=obs_gen)

            # Get resolution part
            res = zfit.pdf.GeneralizedCB(obs=obs_res, mu=zpars[f'mu{ip}'], sigmal=zpars[f'sigmaL{ip}'], sigmar=zpars[f'sigmaR{ip}'], alphal=zpars[f'alphaL{ip}'], alphar=zpars[f'alphaR{ip}'], nl=zpars[f'nL{ip}'], nr=zpars[f'nR{ip}'])

            # Do convolution
            func   = lineshape_pdf if (nbins_gen >= nbins_res) else res
            kernel = res           if (nbins_gen >= nbins_res) else lineshape_pdf
            conv = zfit.pdf.FFTConvPDFV1(func, kernel, n=nbins_res, obs=obs_mom, norm=obs_full)
            pdfs.append(conv)
            fracs.append(len(gen_slice))

        PDF = zfit.pdf.SumPDF(pdfs, fracs=[f/sum(fracs) for f in fracs[:-1]], obs=obs_mom, norm=obs_full, extended=N)

    elif model == 'poly58':
        PDF = poly58(obs=obs_mom, a5=zpars['a5'], a6=zpars['a6'], a7=zpars['a7'], a8=zpars['a8'], extended=N)

    elif model == 'uniform':
        PDF = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N)

    elif model == 'Gauss':
        PDF = zfit.pdf.Gauss(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], extended=N)

    else:
        raise Exception(f"ERROR: model {model} not defined!")

    return PDF, N
  
