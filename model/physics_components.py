# Pure dictionary-based config file (no implementation details)

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

# for convolution
from helper import make_HistogramPDF
import zfit

# Create the DIO theory PDF from the existing lineshape
prob_dio, edges_dio = theo_exp_pars_DIO['lineshape']
obs_x = zfit.Space('x', limits=(95, 115))
dio_theory_pdf = make_HistogramPDF(prob_dio, edges_dio)(obs=obs_x)

# ============================================================================
# MOMENTUM COMPONENTS
# ============================================================================

mom_components = {
    'CE': {
        'pdf': 'dscb',  # Double-sided crystal ball
        'pars': {
            'mu': (104.339, 104.3296, 104.3484),
            'sigma': (0.286437,0.280637,0.292237),
            'alphaL': (0.561202 ,0.540202,0.582202),
            'nL': (2.39481,2.32181,2.46781),
            'alphaR': (2.73596,2.58596,2.88596),
            'nR': (2.84393,2.16393,3.52393)
        },
        'norm': (0, -1e4, 1e4),  # (value, lower_bound, upper_bound) - allows negative for BG-only fits
        'treat_params': 'float',        
        'fixed_params': ['mu', 'sigma', 'alphaL', 'nL','alphaR','nR'],
        'startCode': [168],
        'genCode': [None],
        'lineColor': 'b',
        'lineStyle': '--',
    },

    'Cosmic': {
        'pdf': 'poly2',
        'pars': {
            'c1': (0.219, 0.197, 0.241),
            'c2': (-0.108803, -0.130803, -0.086803)
        },
        'norm': (5000, 0.0, 1e6),  # (value, lower_bound, upper_bound)
        'treat_params': 'constrain',
        'startCode': [None],
        'genCode': [44, 38],
        'lineColor': 'm',
        'lineStyle': '-.',
    },
    'RPC': {
        'pdf': 'poly2',
        'pars': {
            'c1': (-0.54, -0.5462, -0.5338),
            'c2': (-0.1792, -0.2474, -0.111)
        },
        'norm': (24, 0.0, 1e6),  # (value, lower_bound, upper_bound)
        'treat_params': 'constrain',
        'startCode': [178, 179],
        'genCode': [None],
        'lineColor': 'black',
        'lineStyle': '-'
    },
    'DIO': { # Decay in Orbit Background From Target
        'pdf': 'poly58',
        'pars': {'N_DIO': (2000, 1000, 10000)},
        'norm': (55000, 0.0, 1e6),  # (value, lower_bound, upper_bound)
        'treat_params': 'float',        
        #'fixed_params': ['a5', 'a6', 'a7', 'a8'],  # Fix spectrum shape, let N_DIO float
        'startCode': [166, 170],
        'genCode': [None],
        'lineColor': 'g',
        'lineStyle': ':'
        
    },
}


# ============================================================================
# TIME COMPONENTS
# ============================================================================

time_components = {
    'Cosmic': {
        'pdf': 'uniform',
        'pars': None,
        'norm': (35, 0.0, 1e6),  # (value, lower_bound, upper_bound)
        'startCode': [None],
        'genCode': [44, 38],
        'lineColor': 'm',
        'lineStyle': '-.',
    },
    'Muon': {
        'pdf': 'muexp',
        'pars': {'decay_rate_mu': (-0.001157, -0.0015, -0.001)},
        'norm': (55600, 0.0, 1e6),  # (value, lower_bound, upper_bound)
        'startCode': [168, 166, 170],
        'genCode': [None],
        'lineColor': 'b',
        'lineStyle': '--',
    },
    'RPC': {
        'pdf': 'piexp',
        'pars': {'decay_rate_pi': (-0.03846, -0.04, -0.01)},
        'norm': (39, 0.0, 1e6),  # (value, lower_bound, upper_bound)
        'startCode': [178, 179],
        'genCode': [None],
        'lineColor': 'black',
        'lineStyle': (0, (3, 5, 1, 5)),
    }
}
