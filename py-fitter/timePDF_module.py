import numpy as np
import tensorflow as tf
import zfit

default_model_params = {'muexp'   : {'decay_rate_mu'     : (-0.001157,-0.0015, -0.001)},
                        'piexp'   : {'decay_rate_pi'     : (-0.03846, -0.04, -0.01)},
                        'cosmicexp'   : {'decay_rate_cosmic'     : (-0.037, -0.04, -0.03)},
                        'uniform' : {}
                        }
                        
default_norms = {'Cosmic' : 35, 'RPC' : 39, 'Muon' : 55600} #FIXME - should we make these relative?


def TimeModel(obs_time, params_tot, process, model, pardict, fit_range):
    """
    Builds time fit model

    Parameters
    ----------
      obs_time = zfit parameter for reco momentum
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

    # If any parameters are specified in components, override
    if pardict is not None:
        for par in pardict: 
            if par in default_model_params: params[par] = pardict[par]                

    zpars = {'N' : N}
    for p in params.keys():
      zpars[p] = zfit.Parameter(p+'_'+process, params[p][0], params[p][1], params[p][2])
    if model == "muexp":
      params_tot.append(N)
      PDF = zfit.pdf.Exponential(zpars['decay_rate_mu'], obs=obs_time, extended=N)
    elif model == "piexp":
      params_tot.append(N)
      PDF = zfit.pdf.Exponential(zpars['decay_rate_pi'], obs=obs_time, extended=N)
    elif model == "cosmicexp":
      params_tot.append(N)
      PDF = zfit.pdf.Exponential(zpars['decay_rate_cosmic'], obs=obs_time, extended=N)
    elif model == "uniform":
      params_tot.append(N)
      PDF = zfit.pdf.Uniform(low=fit_range[0], high=fit_range[1], obs=obs_time, extended=N)
    else:
        raise Exception(f"ERROR: model {model} not defined!")

    return PDF, N
