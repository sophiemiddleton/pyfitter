# Custom_PDF
# Define custom PDF

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


def MomModel(obs_mom, params, process, model, fit_range):

    # Normalization depends on process, not model
    norm_defaults = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 200, 'RPC' : 23}

    if process in list(norm_defaults.keys()):
        N = zfit.Parameter('N_'+process, norm_defaults[process], 0, 1e6)
    else:
        N = zfit.Parameter('N_'+process, 10,                     0, 1e6)

    if model == 'dscb': # Double-Sided Crystalball function
        mu     = zfit.Parameter('mu_'    +process, 104,   103,  107)
        sigma  = zfit.Parameter('sigma_' +process, 0.5,   0.08, 2.0)
        alphal = zfit.Parameter('alphal_'+process, 0.422, 0,    10)
        nl     = zfit.Parameter('nl_'    +process, 25.1,  0,    100)
        alphar = zfit.Parameter('alphar_'+process, 2.227, 0,    100)
        nr     = zfit.Parameter('nr_'    +process, 5.954, 0,    100)
        params.extend([mu, sigma, alphal, nl, alphar, nr, N])
        PDF = zfit.pdf.DoubleCB(obs=obs_mom, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, extended=N)
    
    elif model == 'poly58':
        a5 = zfit.Parameter('a5_'+process,  8.6434e-17,   0,     1e-16)
        a6 = zfit.Parameter('a6_'+process,  1.16874e-17,  0,     1e-16)
        a7 = zfit.Parameter('a7_'+process, -1.87828e-19, -1e-18, 0)
        a8 = zfit.Parameter('a8_'+process,  9.16327e-20,  0,     1e-18)    
        params.append(N)
        PDF = poly58(obs=obs_mom, a5=a5, a6=a6, a7=a7, a8=a8, extended=N)

    elif model == 'uniform':
        PDF = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N)
        params.append(N)

    elif model == 'Gauss':
        mu    = zfit.Parameter('mu_'   +process, 100, 95,   115)
        sigma = zfit.Parameter('sigma_'+process, 0.5, 1e-3, 1e3)
        params.extend([mu, sigma, N])
        PDF = zfit.pdf.Gauss(obs=obs_mom, mu=mu, sigma=sigma, extended=N)

    else:
        raise Exception("ERROR: RPC Model not Defined!")

    return PDF, N
  
