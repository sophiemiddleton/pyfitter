# momPDF_module: # Define custom PDF for momentum

import numpy as np
import tensorflow as tf
import zfit
import math
import hist as hist
import dill as pickle

# DIO models
m_mu = 105.194 # mass of the muon [MeV]

# Shared time parameters cache for MomTimeModel
_shared_time_params = {}

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
    'poly2' : { 'c1' : (0.47, 0.46, 0.48),'c2' : (0.011,0.0018,0.0202)},
    'dscb'   : {'mu': (104, 103, 107), 'sigma': (0.5, 0.08, 2.0), 'alphaL': (0.422, 0, 10), 
                'nL': (25.1, 0, 100), 'alphaR': (2.227, 0, 100), 'nR': (5.954, 0, 100)},
    'Gauss'  : {'mu'     : (100,           95,    115),'sigma'  : (0.5,           1e-3,  1e3)},
    'uniform': {}
}

default_norms = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 5000, 'RPC' : 20}

def MomModel(obs_mom, params_tot, process, model, pardict, treat_params, fit_range, constraints, advanced_config=None, use_advanced=False):
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
            pdf_conv = zfit.pdf.TruncatedPDF(pdf_conv, limits=obs_mom, obs=obs_mom, extended=N)
            try:
                pdf_conv.set_yield(N)
            except Exception:
                pass
            return pdf_conv, N

    # --- SIMPLE PATH  ---
    params = default_model_params.get(model, {}).copy()
    # Normalize pardict keys: allow keys with or without process suffix
    if pardict is not None:
        for par, val in pardict.items():
            if par in params:
                params[par] = val
            else:
                # if pardict uses name with process suffix (e.g., c1_Cosmic), map to base name
                suffix = f"_{process}"
                if par.endswith(suffix):
                    base = par[:-len(suffix)]
                    if base in params:
                        params[base] = val
                        continue
                # also accept keys with process prefix (unlikely) or exact match otherwise store as-is
                params[par] = val
    
    zpars = {}
    for p in params.keys():
        p_name = p + '_' + process
        if treat_params == 'constrain':
            zpars[p] = zfit.Parameter(p_name, params[p][0], params[p][0]+5*params[p][1], params[p][0]+5*params[p][2], step_size=0.0001)
            params_tot.append(zpars[p])
            constraints.append(zfit.constraint.GaussianConstraint(zpars[p], observation=params[p][0], uncertainty=max(abs(params[p][1]), abs(params[p][2]))))
        elif treat_params == 'fix':
            zpars[p] = zfit.Parameter(p_name, params[p][0], floating=False)
            try:
                params_tot.append(zpars[p])
            except Exception:
                pass
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
    elif model == 'poly2': # Cheb. poly order 2
        # Accept either zfit.Parameter in zpars or numeric defaults; ensure unique names per process
        def _get_coeff(name, default=0.0):
            val = zpars.get(name)
            pname = f"{name}_{process}"
            if isinstance(val, zfit.Parameter):
                return val
            try:
                num = float(val)
                p = zfit.Parameter(pname, num, floating=False)
                try:
                    params_tot.append(p)
                except Exception:
                    pass
                return p
            except Exception:
                p = zfit.Parameter(pname, float(default), floating=False)
                try:
                    params_tot.append(p)
                except Exception:
                    pass
                return p
        
        # use base coefficient names ('c1','c2') and allow pardict to have supplied values
        c1 = _get_coeff('c1', default=0.0)
        c2 = _get_coeff('c2', default=0.0)
        coeffs = [c1, c2]
        print("COSMIC COEEFS", c1,c2)
        # Create a Chebyshev polynomial PDF
        PDF = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N)
    elif model == 'poly5': # Cheb. poly order 5
        def _get_coeff(name, default=0.0):
            val = zpars.get(name)
            pname = f"{name}_{process}"
            if isinstance(val, zfit.Parameter):
                return val
            try:
                num = float(val)
                p = zfit.Parameter(pname, num, floating=False)
                try:
                    params_tot.append(p)
                except Exception:
                    pass
                return p
            except Exception:
                p = zfit.Parameter(pname, float(default), floating=False)
                try:
                    params_tot.append(p)
                except Exception:
                    pass
                return p

        c1 = _get_coeff('c1', default=0.0)
        c2 = _get_coeff('c2', default=0.0)
        c3 = _get_coeff('c3', default=0.0)
        c4 = _get_coeff('c4', default=0.0)
        c5 = _get_coeff('c5', default=0.0)
        coeffs = [c1, c2, c3, c4, c5]

        # Create a Chebyshev polynomial PDF
        PDF = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N)
    else:
        raise ValueError(f"Model {model} not recognized in Simple Path")

    return PDF, N


def MomTimeModel(obs_mom, obs_time, mom_params_tot, time_params_tot, process, mom_model, time_model, pardict, treat_params, fit_range, constraints):
    """
    Combined momentum × time PDF. Momentum part is built by `MomModel` (extended with yield N).
    Time part uses fixed/shared decay rates:
      - DIO and CE: exponential with exponent -1/864 (shared)
      - RPC: exponential with exponent -1/26
      - Cosmic: uniform between 400 and 1695 ns

    Returns (pdf_2d, N, mom_pdf, time_pdf)
    """
    mom_pdf, N = MomModel(obs_mom, mom_params_tot, process, mom_model, pardict, treat_params, fit_range, constraints)

    # Ensure shared/fixed decay parameters exist
    if 'decay_shared_CE_DIO' not in _shared_time_params:
        _shared_time_params['decay_shared_CE_DIO'] = zfit.Parameter('decay_shared_CE_DIO', -1.0/864.0, floating=False)
    if 'decay_rpc' not in _shared_time_params:
        _shared_time_params['decay_rpc'] = zfit.Parameter('decay_rpc', -1.0/26.0, floating=False)

    # Select time PDF for each process
    if process in ('DIO', 'CE'):
        lam = _shared_time_params['decay_shared_CE_DIO']
        time_pdf = zfit.pdf.Exponential(lam, obs=obs_time)
    elif process == 'RPC':
        lam = _shared_time_params['decay_rpc']
        time_pdf = zfit.pdf.Exponential(lam, obs=obs_time)
    elif process == 'Cosmic':
        time_pdf = zfit.pdf.Uniform(low=400.0, high=1695.0, obs=obs_time)
    else:
        # Fallback: try to use TimeModel if available
        try:
            time_pdf, N_time = TimeModel(obs_time, time_params_tot, process, time_model, pardict, fit_range)
        except Exception:
            time_pdf = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_time)

    # Append fixed params to the time_params_tot list if not already present
    try:
        if process in ('DIO', 'CE'):
            if _shared_time_params['decay_shared_CE_DIO'] not in time_params_tot:
                time_params_tot.append(_shared_time_params['decay_shared_CE_DIO'])
        elif process == 'RPC':
            if _shared_time_params['decay_rpc'] not in time_params_tot:
                time_params_tot.append(_shared_time_params['decay_rpc'])
    except Exception:
        pass

    obs_2d = obs_mom * obs_time
    pdf_2d = zfit.pdf.ProductPDF([mom_pdf, time_pdf], obs=obs_2d)
    try:
        pdf_2d.set_yield(N)
    except Exception:
        # Fall back to wrapping in a TruncatedPDF with extended yield
        try:
            pdf_2d = zfit.pdf.TruncatedPDF(pdf_2d, limits=obs_2d, obs=obs_2d, extended=N)
        except Exception:
            pass

    return pdf_2d, N, mom_pdf, time_pdf

