# Define custom PDF for momentum

import numpy as np
import tensorflow as tf
import zfit

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
                                    'alphal' : (0.422,         0,     10),
                                    'nl'     : (25.1,          0,     100),
                                    'alphar' : (2.227,         0,     100),
                                    'nr'     : (5.954,         0,     100)},
                        'poly58' : {'a5'     : (8.6434e-17,    0,     1e-16),
                                    'a6'     : (1.16874e-17,   0,     1e-16),
                                    'a7'     : (-1.87828e-19, -1e-18, 0),
                                    'a8'     : (9.16327e-20,   0,     1e-18)},
                        'Gauss'  : {'mu'     : (100,           95,    115),
                                    'sigma'  : (0.5,           1e-3,  1e3)},
                        'uniform' : {}
                        }

default_norms = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 200, 'RPC' : 23}

def MomModel(obs_mom, params_tot, process, model, pardict, fit_range):

    if pardict is not None and 'N' in pardict:
        N = zfit.Parameter('N_'+process, pardict['N'][0], pardict['N'][1], pardict['N'][2])
    elif process in list(default_norms.keys()):
        N = zfit.Parameter('N_'+process, default_norms[process], 0, 1e6)
    else:
        N = zfit.Parameter('N_'+process, 10,                     0, 1e6)

    # Start with default parameters for model
    params = default_model_params[model]

    # If any parameters are specified in components, override
    if pardict is not None:
        for par in pardict: 
            if par in default_model_params: params[par] = pardict[par]                

    zpars = {'N' : N}
    for p in params.keys():
        zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][1], params[p][2])
        
    if model == 'dscb': # Double-Sided Crystalball function
        params_tot.extend(list(zpars.values()))
        PDF = zfit.pdf.DoubleCB(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], alphal=zpars['alphal'], nl=zpars['nl'], alphar=zpars['alphar'], nr=zpars['nr'], extended=N)
    
    elif model == 'poly58':
        params_tot.append(N)
        PDF = poly58(obs=obs_mom, a5=zpars['a5'], a6=zpars['a6'], a7=zpars['a7'], a8=zpars['a8'], extended=N)

    elif model == 'uniform':
        params_tot.append(N)
        PDF = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N)

    elif model == 'Gauss':
        params_tot.extend(list(zpars.values()))
        PDF = zfit.pdf.Gauss(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], extended=N)

    else:
        raise Exception(f"ERROR: model {model} not defined!")

    return PDF, N
  
