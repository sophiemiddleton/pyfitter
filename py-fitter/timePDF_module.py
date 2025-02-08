import numpy as np
import tensorflow as tf
import zfit

default_norms = {'CE' : 600, 'DIO' : 55000, 'Cosmic' : 200, 'RPC' : 23}


def TimeModel(obs_mom, params_tot, process, model, pardict, fit_range):
  
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
    if model == "muon_exp":
      decay_rate = zfit.Parameter('decay_rate', -1/864, -1/10, -1/1000)
      N_exp = zfit.Parameter('N_exp', 6000, 0, 1e5)
      exp_t = zfit.pdf.Exponential(decay_rate, obs=obs_time, extended=N_exp)
    if model == "pion_exp":
      decay_rate_RPC = zfit.Parameter('decay_rate_rpc', -1/864, -1/10, -1/1000)
      N_RPC = zfit.Parameter('N_rpc', 0, 0, 1e4)
      exp_t_RPC = zfit.pdf.Exponential(decay_rate_RPC, obs=obs_time, extended=N_rpc)
    if model == "uniform"
      N_cosmic = zfit.Parameter('N_cosmic', 0, 0, 1e4)
      cosmic_t = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_time, extended=N_cosmic)
