
# mom_components.py: Component Definitions and Theoretical Functions

import numpy as np
import math
import zfit

#from fit_config import *
from theo_components import LeadingLog, binned_spectrum_CeLL
from res_components import res_components
from helper import *

#  --- initiate advance DIO ------#
EFFICIENCY_PATH = '../common/efficiency.pkl'
DIO_THEORY_PATH = '/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v10_34_00/Offline/ConditionsService/data/czarnecki_szafron_Al_2016.tbl'

# --- initiate advance CE ------#

lineshapes_in_CE_BASE = {
    'theo': binned_spectrum_CeLL(binwidth=0.1),
    'eff': EFFICIENCY_PATH
}

#------------ For a simultaneous fit ------------------#
# Load the data required for the resolution/loss likelihood terms (for simultaneous fit)
dict_flat = pkl.load(open('../common/skimmed_flat_mom_v2.pkl','rb'))

# Pass the tuple (gen, mc) to simul_source so get_nll() can unpack it
flat_res = res_components(
    params = '../common/fitpars_flat_res_entrance_gcb.pkl', 
    res_type = 'res', 
    pdf = 'gcb',
    simul_source = (dict_flat['entrance']['gen'], dict_flat['entrance']['mc'])
)

flat_loss = res_components(
    params = '../common/fitpars_flat_loss_entrance_landau_unbinned.pkl',
    res_type = 'loss', 
    pdf = 'landau',
    simul_source = (dict_flat['entrance']['mc'], dict_flat['entrance']['reco'])
)

#------------ For a fixed input fit ------------------#
# No simul_source provided -> get_nll() will be skipped by fit_module
flat_res_fixed  = res_components(params = '../common/fitpars_flat_res_entrance_gcb.pkl', 
                           res_type = 'res',  pdf = 'gcb')
flat_loss_fixed = res_components(
    params = '../common/fitpars_flat_loss_entrance_landau_unbinned.pkl',
    res_type = 'loss', 
    pdf = 'landau')
    
fitpars_in = {
    'res': {'params': flat_res.get_params()}, 
    'loss': {'params': flat_loss.get_params()}
}
########################################################

# Generate structured pars and merge convolution objects
theo_exp_pars = gen_theo_exp(fitpars_in, lineshapes_in_CE_BASE) 
theo_exp_pars.update({
    'res': flat_res,
    'loss': flat_loss,
    'info': {
        'p_bins': flat_res.fitpars['info']['p_bins'],
        'res': None,
        'loss': None,
    }
})




# --- Component Dictionary ---
mom_components = {

    'Cosmic' : {'pdf' : 'uniform', # Default: no assumptions of res+eff+loss yet
                'pars' : None,
                'treat_params' : 'float',
                'startCode' : [None],
                'genCode' : [44,38],
                'lineColor' : 'm',
                'lineStyle' : '-.',
                'catColor' : 'violet',
                'advanced_pars': None
                }, # No advanced model for Cosmics yet

    'CE': {
        'pdf': 'dscb',  # Default: simple double sided crystal ball
        'pars': {
            'mu': (104, 103, 107), 'sigma': (0.5, 0.08, 2.0),
            'alphaL': (0.422, 0, 10), 'nL': (25.1, 0, 100),
            'alphaR': (2.227, 0, 100), 'nR': (5.954, 0, 100)
        },
        'treat_params': 'float',
        'startCode': [168],
        'genCode': [None],
        'lineColor': 'b',
        'lineStyle': '--',
        'catColor': 'lightskyblue',
        'advanced_pars': {
            'pdf_theo': 'theo_exp',
            'treat_params_adv': 'param',
            'fitpars_in_formatted': theo_exp_pars,
            'nll_sources': [flat_res, flat_loss] # List for iteration
        }
    },
    'DIO': { # Decay in Orbit Background From Target
        'pdf': 'DIO_custom_model_2025', # <-- Default: custom theory model
        'pars': {'N': (55000, 0, 1e6), },
        'treat_params': 'fix',
        'startCode': [166, 170],
        'genCode': [None],
        'lineColor': 'g',
        'lineStyle': ':',
        'catColor': 'lightgreen',
        'advanced_pars': None
    },
    
    'RPC': { # Radiative Pion Background (Internal + External)
        'pdf': 'Gauss', # <-- Default: assumes res+eff+loss already
        'pars': {'mu': (100, 98, 102), 'sigma': (11, 5, 25)},
        'treat_params': 'float',
        'startCode': [178, 179],
        'genCode': [None],
        'lineColor': 'darkorange',
        'lineStyle': (0, (3, 5, 1, 5)),
        'catColor': 'orange',
        'advanced_pars': None # No advanced model for RPC
    }
}
