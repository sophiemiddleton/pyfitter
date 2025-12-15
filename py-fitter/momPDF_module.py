# momPDF_module: # Define custom PDF for momentum

import numpy as np
import tensorflow as tf
import zfit
import math
import hist as hist
import dill as pickle

# DIO models
m_mu = 105.194 # mass of the muon [MeV]

class poly58(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = ['a5', 'a6', 'a7', 'a8']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        a5, a6, a7, a8 = self.params['a5'], self.params['a6'], self.params['a7'], self.params['a8']
        m_Al = 25133 # mass of the Aluminum atom [MeV]
        delta = tf.nn.relu(m_mu - x - x**2 / (2 * m_Al))
        return a5 * delta**5 + a6 * delta**6 + a7 * delta**7 + a8 * delta**8

class DIO_custom_model_2025(zfit.pdf.ZPDF):
    _N_OBS = 1
    _PARAMS = ['DIO_endpoint', 'beta', 'degree_shift']

    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        endpoint = self.params['DIO_endpoint']
        beta = self.params['beta']
        degree_shift = self.params['degree_shift']

        delta_E = (endpoint - x)
        is_valid = delta_E > 0
        safe_delta_E = tf.where(is_valid, delta_E, 1.0)
        log_delta_E_over_mu = tf.math.log(safe_delta_E/m_mu)

        power = 5.0 + degree_shift
        poly_term = beta * tf.square(log_delta_E_over_mu)
        pdf_val = tf.pow(safe_delta_E, power) * tf.exp(poly_term)

        return tf.where(is_valid, pdf_val, 0.0)

default_model_params = {
    'dscb'   : {'mu': (104, 103, 107), 'sigma': (0.5, 0.08, 2.0), 'alphaL': (0.422, 0, 10), 
                'nL': (25.1, 0, 100), 'alphaR': (2.227, 0, 100), 'nR': (5.954, 0, 100)},
    'Gauss'  : {'mu': (100, 95, 115), 'sigma': (0.5, 1e-3, 1e3)},
    'uniform': {}
}

default_norms = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 200, 'RPC' : 1}

def MomModel(obs_mom, params_tot, process, model, pardict, treat_params, fit_range, constraints, 
             dio_efficiency=None, dio_resolution=None, advanced_config=None, use_advanced=False):
    """
    Builds momentum fit model with a toggle for Advanced Configuration
    """
    
    # Handle the Normalization (N)
    if isinstance(pardict, dict) and 'N' in pardict:
        N = zfit.Parameter('N_'+process, pardict['N'][0], pardict['N'][1], pardict['N'][2])
    elif process in default_norms:
        N = zfit.Parameter('N_'+process, default_norms[process], 0, 1e6)
    else:
        N = zfit.Parameter('N_'+process, 10, 0, 1e6)
    params_tot.append(N)

    # Branching Logic: Advanced Model (Theo_Exp) vs Simple Model (DSCB/Gauss)
    if use_advanced and advanced_config:
        # --- ADVANCED PATH ---
        adv_model = advanced_config.get('pdf_theo')
        adv_treat = advanced_config.get('treat_params_adv')
        
        if adv_model == 'theo_exp':
            # Use the structured pars pre-built in mom_components.py
            theo_exp_pars = advanced_config['fitpars_in_formatted']
            
            # Extract formatted parts for the doConv logic
            prob, edges = theo_exp_pars['lineshape']
            info = theo_exp_pars['info']
            
            # Map parameters based on treatment (simul, param, etc.)
            zpars = {}
            for comp in ['res', 'loss']:
                comp_pars = theo_exp_pars[comp].get_params()
                for p, val in comp_pars.items():
                    if adv_treat == 'simul':
                        zpars[p] = zfit.ComposedParameter(p+'_'+process, lambda x: 1*x, params=val)
                    else:
                        zpars[p] = val # Use existing param objects
                        if val not in params_tot: params_tot.append(val)

            from helper import doConv, make_HistogramPDF
            obs_gen = zfit.Space('x', fit_range)
            obs_res = zfit.Space('x', -10, 10) # Standard resolution window
            
            true_pdf_slice = (prob, edges)
            pdf_conv = doConv(true_pdf_slice, obs_gen, obs_res, process, info, zpars)
            pdf_conv = zfit.pdf.TruncatedPDF(pdf_conv,limits=obs_mom,obs=obs_mom,extended=N)
            PDF.set_yield(N)
            return PDF, N

    # --- SIMPLE PATH  ---
    params = default_model_params.get(model, {}).copy()
    if pardict is not None:
        for par in pardict.keys(): 
            if par in params: params[par] = pardict[par]
    
    zpars = {}
    for p in params.keys():
        p_name = p + '_' + process
        if treat_params == 'constrain':
            zpars[p] = zfit.Parameter(p_name, params[p][0], params[p][0]+5*params[p][1], params[p][0]+5*params[p][2], step_size=0.0001)
            params_tot.append(zpars[p])
            constraints.append(zfit.constraint.GaussianConstraint(zpars[p], observation=params[p][0], uncertainty=max(abs(params[p][1]), abs(params[p][2]))))
        elif treat_params == 'fix':
            zpars[p] = zfit.Parameter(p_name, params[p][0], floating=False)
        else:
            zpars[p] = zfit.Parameter(p_name, params[p][0], params[p][1], params[p][2])
            params_tot.append(zpars[p])

    if model == 'dscb':
        PDF = zfit.pdf.DoubleCB(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], alphal=zpars['alphaL'], nl=zpars['nL'], alphar=zpars['alphaR'], nr=zpars['nR'], extended=N)
    elif model == 'Gauss':
        PDF = zfit.pdf.Gauss(obs=obs_mom, mu=zpars['mu'], sigma=zpars['sigma'], extended=N)
    elif model == 'uniform':
        PDF = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_mom, extended=N)
    elif model == 'DIO_custom_model_2025':
        PDF = DIO_custom_model_2025(obs=obs_mom, DIO_endpoint=zpars.get('endpoint', 104.97), beta=zpars.get('beta', -0.002), degree_shift=zpars.get('degree_shift', 0), extended=N)
    else:
        raise ValueError(f"Model {model} not recognized in Simple Path")

    return PDF, N

