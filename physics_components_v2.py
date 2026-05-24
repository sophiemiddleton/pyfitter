# physics_components.py
# Consolidated physics component definitions and constants
# Pure dictionary-based config file (no implementation details)

# ============================================================================
# PHYSICS CONSTANTS (from theo_components.py)
# ============================================================================


import numpy as np
import math
import zfit
import pickle as pkl

from custom_models import LeadingLog, binned_spectrum_CeLL, res_components
from helper import gen_theo_exp

# Efficiency from flat e- at target (can be used for all "from target" processes
EFFICIENCY_PATH = 'common/efficiency.pkl'

# --- initiate advance CE ------#

# Line shape comes from theory spectrum and efficiency function product
lineshapes_in_CE_BASE = {
    'theo': binned_spectrum_CeLL(binwidth=0.1),
    'eff': EFFICIENCY_PATH
}

#------------ For a simultaneous fit including resolution (res) and loss (loss) taken from flat e- samples ------------------#
# Load the data required for the resolution/loss likelihood terms (for simultaneous fit)
dict_flat = pkl.load(open('common/skimmed_flat_mom_v2.pkl','rb'))

# Pass the tuple (gen, mc) to simul_source so get_nll() can unpack it
flat_res = res_components(
    params = 'common/fitpars_flat_res_entrance_gcb.pkl', 
    res_type = 'res', 
    pdf = 'gcb',
    simul_source = (dict_flat['entrance']['gen'], dict_flat['entrance']['mc'])
)

flat_loss = res_components(
    params = 'common/fitpars_flat_loss_entrance_landau_unbinned.pkl',
    res_type = 'loss', 
    pdf = 'landau',
    simul_source = (dict_flat['entrance']['mc'], dict_flat['entrance']['reco'])
)
# -------------------------------------------------#

#------------ For a fixed input fit fit including resolution (res) and loss (loss) taken from flat e- samples ------------------#
# No simul_source provided -> get_nll() will be skipped by fit_module
flat_res_fixed  = res_components(params = 'common/fitpars_flat_res_entrance_gcb.pkl', 
                           res_type = 'res',  pdf = 'gcb')
flat_loss_fixed = res_components(
    params = 'common/fitpars_flat_loss_entrance_landau_unbinned.pkl',
    res_type = 'loss', 
    pdf = 'landau')
    
fitpars_in = {
    'res': {'params': flat_res.get_params()}, 
    'loss': {'params': flat_loss.get_params()}
}
# -------------------------------------------------#

# Generate structured pars and merge convolution objects
theo_exp_pars = gen_theo_exp(fitpars_in, lineshapes_in_CE_BASE) 
theo_exp_pars.update({
    'res': flat_res,
    'loss': flat_loss,
    'info': {
        'pdf': 'gcb',  # Resolution PDF type
        'p_bins': flat_res.fitpars['info']['p_bins'],
        'res': None,
        'loss': None,
    }
})

# --------- end CE -----------------------------------#

#  --- initiate advanced DIO ------#

DIO_THEORY_PATH = '/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v10_34_00/Offline/ConditionsService/data/czarnecki_szafron_Al_2016.tbl'

# Generate the DIO Advanced Parameters
lineshapes_in_DIO = {
    'theo': DIO_THEORY_PATH,
    'eff' : EFFICIENCY_PATH
}

# This generates the binned lineshape combined with the resolution/loss parameters
theo_exp_pars_DIO = gen_theo_exp(fitpars_in, lineshapes_in_DIO)

# Add the required convolution info (sharing the p_bins and objects from CE)
theo_exp_pars_DIO.update({
    'res': flat_res,
    'loss': flat_loss,
    'info': {
        'pdf': 'gcb',  # Resolution PDF type
        'p_bins': flat_res.fitpars['info']['p_bins'],
        'res': None,
        'loss': None,
    }
})

# ============================================================================
# MOMENTUM COMPONENTS
# ============================================================================

mom_components = {
    'CE': {
        'pdf': 'dscb',  # Double-sided crystal ball
        'pars': {
            'mu': (104, 103, 107),
            'sigma': (0.5, 0.1, 2.0),
            'alphaL': (0.422, 0, 10),
            'nL': (25.1, 0, 100),
            'alphaR': (2.227, 0, 100),
            'nR': (5.954, 0, 100)
        },
        'treat_params': 'float',
        'startCode': [168],
        'genCode': [None],
        'lineColor': 'b',
        'lineStyle': '--',
        'catColor': '#ffff00',
        'advanced_pars': {
            'pdf_theo': 'theo_exp',
            'treat_params_adv': 'param',
            'fitpars_in_formatted': theo_exp_pars,
            'nll_sources': [flat_res, flat_loss]
        }
    },
    'Cosmic': {
        'pdf': 'poly2',
        'pars': {
            'c1': (0.219, 0.197, 0.241),
            'c2': (-0.108803, -0.130803, -0.086803)
        },
        'treat_params': 'constrain',
        'startCode': [None],
        'genCode': [44, 38],
        'lineColor': 'm',
        'lineStyle': '-.',
        'catColor': '#1f77b4',
    },

    'DIO': { # Decay in Orbit Background From Target
        'pdf': 'poly58',
        'pars': {'N_DIO': (2000, 1000, 30000)},
        'treat_params': 'fix',
        'startCode': [166, 170],
        'genCode': [None],
        'lineColor': 'g',
        'lineStyle': ':',
        'catColor': 'lightgreen',
        'advanced_pars': {
          'pdf_theo': 'theo_exp',
          'treat_params_adv': 'simul', # Use 'simul' to share parameters with CE
          'fitpars_in_formatted': theo_exp_pars_DIO,
          'nll_sources': None 
        }
    },
}


# ============================================================================
# TIME COMPONENTS
# ============================================================================

time_components = {
    'Cosmic': {
        'pdf': 'uniform',
        'pars': None,
        'startCode': [None],
        'genCode': [44, 38],
        'lineColor': 'm',
        'lineStyle': '-.',
        'catColor': 'violet',
    },
    'Muon': {
        'pdf': 'muexp',
        'pars': {'decay_rate_mu': (-0.001157, -0.0015, -0.001)},
        'startCode': [168, 166, 170],
        'genCode': [None],
        'lineColor': 'b',
        'lineStyle': '--',
        'catColor': 'lightskyblue',
    },
    'RPC': {
        'pdf': 'piexp',
        'pars': {'decay_rate_pi': (-0.03846, -0.04, -0.01)},
        'startCode': [178, 179],
        'genCode': [None],
        'lineColor': 'black',
        'lineStyle': (0, (3, 5, 1, 5)),
        'catColor': 'black',
    }
}
