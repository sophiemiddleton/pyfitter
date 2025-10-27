# Define custom PDF for momentum

import numpy as np
import tensorflow as tf
import zfit
import math
import hist as hist
import dill as pickle

class poly58(zfit.pdf.ZPDF):
    """
    Class:
      Czarnecki et al parameterization for DIO theory spectrum
    
    Methods:
      unnormalized_pdf(self, x):
        defines parameters and zfit PDF
    """
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
                        'theo_exp' : None,
                        'poly58' : {'a5'     : (8.97879e-17,    0,     1e-16),
                                    'a6'     : (1.17169e-17,   0,     1e-16),
                                    'a7'     : (-1.06599e-19, -1e-18, 0),
                                    'a8'     : (8.14251e-20,   0,     1e-19)},
                        'Gauss'  : {'mu'     : (100,           95,    115),
                                    'sigma'  : (0.5,           1e-3,  1e3)},
                        'uniform' : {}
                        }

default_norms = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 200, 'RPC' : 1} #FIXME - should we make these relative?


def MomModel(obs_mom, params_tot, process, model, pardict, treat_params, fit_range, constraints, dio_efficiency = None,
    dio_resolution = None):
    """
    Builds momentum fit model

    Parameters
    ----------
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
    params = default_model_params[model]
    # If any parameters are specified in mom_components.py, override
    if pardict is not None:
        if params is None:
            params = pardict
        else:
            for par in pardict.keys(): 
                if par in params.keys() : params[par] = pardict[par]
    zpars = {}

    if isinstance(params,dict) and params:
        lineshape = params.pop('lineshape',None) # lineshape X resolution fits
        info = params.pop('info',None)
        
        for p in params.keys():
            if treat_params == 'constrain':
                zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][0]+5*params[p][1], params[p][0]+5*params[p][2],step_size=0.0001)
                params_tot.append(zpars[p])
                constraints.append(zfit.constraint.GaussianConstraint(zpars[p],observation=params[p][0],uncertainty=max(abs(params[p][1]),abs(params[p][2]))))
            elif treat_params == 'fix':
                zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][0]-0.005, params[p][0]+0.005, floating=False)
            elif treat_params == 'simul':
                zpars[p] = zfit.ComposedParameter(p+'_'+process, lambda x : 1*x, params=params[p])
            elif treat_params == 'param':
                zpars[p] = params[p]
                params_tot.append(params[p])
                constraints.append(zfit.constraint.GaussianConstraint(zpars[p],observation=float(params[p].value()),uncertainty=max((float(params[p].upper)-float(params[p].value())),(float(params[p].value())-float(params[p].lower)))/5.))
            else:
                if treat_params != 'float':
                    print(f"Supported values for treat_params are 'float', 'fix', 'constrain', 'param', or 'simul'. You are using {treat_params} -- will be treated as 'float'")
                zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][1], params[p][2])
                params_tot.append(zpars[p])

    if model == 'dscb': # Crystalball function with asymmetric tails
        PDF = zfit.pdf.DoubleCB(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], alphal=zpars['alphaL'], nl=zpars['nL'], alphar=zpars['alphaR'], nr=zpars['nR'], extended=N)
        
    elif model == 'gcb': # Fully asymmetric Crystalball function
        PDF = zfit.pdf.GeneralizedCB(obs=obs_mom, mu=zpars['mu'], sigmal=zpars['sigmaL'], sigmar=zpars['sigmaR'], alphal=zpars['alphaL'], nl=zpars['nL'], alphar=zpars['alphaR'], nr=zpars['nR'], extended=N)

    elif model == 'kde': # Kernel Density Estimator
        data = zfit.Data.from_numpy(array=params, obs=obs_mom)
        PDF = zfit.pdf.KDE1DimGrid(data, num_grid_points=256, binning_method='linear', extended=N, bandwidth='adaptive_zfit')
        
    elif model == 'theo_exp':
        if lineshape is None or info is None:
            raise Exception("ERROR: 'theo_exp' model can only be used if 'lineshape' and 'info' are defined in parameters dict")
        
        # Get lineshape
        prob,edges = lineshape
            
        # Get relevant spaces for convolution
        # Want final obs_conv to be obs_mom
        obs_kern = []
        obs_func = []
        bound_all = 0
        for ipdf, name in enumerate(info.keys()):
            bound = 1 if name == 'res' else 10
            bound_all += bound
            obs_func.append(zfit.Space('x',float(obs_mom.v1.lower)-bound_all,float(obs_mom.v1.upper)+bound_all))
            obs_kern.append(zfit.Space('x',-bound,bound))
        obs_func.reverse()
        obs_kern.reverse()
        names = reversed(list(info.keys()))

        # Iteratively do convolution
        from helper import doConv
        pdf_conv = (prob,edges)
        for name, obs_f, obs_k in zip(names,obs_func,obs_kern):
            pdf_conv = doConv(pdf_conv, obs_f, obs_k, name, info[name], zpars)
        PDF = zfit.pdf.TruncatedPDF(pdf_conv,limits=obs_mom,obs=obs_mom,extended=N)

    elif model == 'poly58':
        PDF = poly58(obs=obs_mom, a5=zpars['a5'], a6=zpars['a6'], a7=zpars['a7'], a8=zpars['a8'], extended=N)

    elif model == 'uniform':
        PDF = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N)
        
    elif model == 'Gauss':
        PDF = zfit.pdf.Gauss(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], extended=N)
        
    else:
        raise Exception(f"ERROR: model {model} not defined!")
    
    return PDF, N

def _load_pdf(file_path_or_pdf):
    if isinstance(file_path_or_pdf, zfit.pdf.ZPDF):
        return file_path_or_pdf  
    else:
        with open(file_path_or_pdf, "rb") as f:
            return pickle.load(f)
        
