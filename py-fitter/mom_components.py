# mom_components.py: Component Definitions and Theoretical Functions

import numpy as np
import math
import zfit

from theo_components import LeadingLog, binned_spectrum_CeLL
from res_components import res_components
from helper import *

# Efficiency from flat e- at target (can be used for all "from target" processes
EFFICIENCY_PATH = '../common/efficiency.pkl'

# --- initiate advance CE ------#

# Line shape comes from theory spectrum and efficiency function product
lineshapes_in_CE_BASE = {
    'theo': binned_spectrum_CeLL(binwidth=0.1),
    'eff': EFFICIENCY_PATH
}

#------------ For a simultaneous fit including resolution (res) and loss (loss) taken from flat e- samples ------------------#
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
# -------------------------------------------------#

#------------ For a fixed input fit fit including resolution (res) and loss (loss) taken from flat e- samples ------------------#
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
# -------------------------------------------------#

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
        'p_bins': flat_res.fitpars['info']['p_bins'],
        'res': None,
        'loss': None,
    }
})


# --- Component Dictionary ---
mom_components = {

    # Cosmics assumes eff+res+loss included from control region (Off spill)
    'Cosmic' : {'pdf' : 'poly2',
                'pars' : { 'c1' : (0.223,0.009, 0.437),
                           'c2' : (-0.063,-0.0841,-0.0419)},
                'treat_params' : 'constrain',
                'startCode' : [None],
                'genCode' : [44,38],
                'lineColor' : 'm',
                'lineStyle' : '-.',
                'catColor' : 'violet',
                'advanced_pars': None},

    'DIO': { # Decay in Orbit Background From Target
        'pdf': 'poly58', # Default: to what is in our generator
        'pars': {'a5'     : (8.97879e-17,    1e-17,     1e-16),
                                    'a6'     : (1.17169e-17,   1e-18,     1e-16),
                                    'a7'     : (-1.06599e-19, -1e-18, -1e-19),
                                    'a8'     : (8.14251e-20,   1e-20,     1e-19)},
        'treat_params': 'constrian',
        'startCode': [166, 170],
        'genCode': [None],
        'lineColor': 'g',
        'lineStyle': ':',
        'catColor': 'lightgreen',
        'advanced_pars': None}
    }
    

"""
    # Radiative Pion Background (Internal + External) 
    'RPC'    : {'pdf' : 'Gauss', # Default: assumes res+eff+loss already included from e+ control region
                'pars' : {'mu'    : (100.26, 100.0,100.5),
                          'sigma' : (11.96, 11.5,12.0),
                          'decay_rate_pi'    : (-0.03846,  -0.04, -0.01)},
                'treat_params' : 'float',
                'startCode' : [178,179],
                'genCode' : [None],
                'lineColor' : 'black',
                'lineStyle' : (0, (3, 5, 1, 5)),
                'catColor' : 'black',
                'advanced_pars': None } # No advanced model for RPC
    
}
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
    'advanced_pars': None
},
"""