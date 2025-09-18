# Define fitting and plotting options for all components to include in fit
# pdf                          : model type from momPDF_module.py
# startCode, genCode, catColor : with --categorize, events with this set of startCode and genCode will be colored catColor in the data-fit plot
#                              : order of components gives stacking order (bottom to top)
# lineColor, lineStyle         : line color and style for the PDF component in the data-fit plot (see matplotlib documentation for options)

mom_components = {
    'Cosmic' : {'pdf' : 'uniform',
                'pars' : None,
                'treat_params' : 'float',
                'startCode' : [None],
                'genCode' : [44,38],
                'lineColor' : 'm',
                'lineStyle' : '-.',
                'catColor' : 'violet'},
    
   # 'RPC'    : {'pdf' : 'Gauss',
   #             'pars' : {'mu'    : (100, 98,   102),
   #                       'sigma' : (11, 5, 25)},
   #             'treat_params' : 'float',
   #             'startCode' : [178,179],
   #             'genCode' : [None],
   #             'lineColor' : 'darkorange',
   #             'lineStyle' : (0, (3, 5, 1, 5)),
   #             'catColor' : 'orange'},
    
    'CE'     : {'pdf' : 'gcb',
                'pars' : {'mu'     : (104,   103,  107),
                          'sigmal' : (0.5,   0.08, 2.0),
                          'sigmar' : (0.5,   0.08, 2.0),
                          'alphal' : (0.422, 0,    10),
                          'nl'     : (25.1,  0,    100),
                          'alphar' : (2.227, 0,    100),
                          'nr'     : (5.954, 0,    100)},
                'treat_params' : 'float',
                'startCode' : [168],
                'genCode' : [None],
                'lineColor' : 'b',
                'lineStyle' : '--',
                'catColor' : 'lightskyblue'},

    'DIO'    : {'pdf' : 'poly58',
                'pars' : {'N' : (55000, 0, 1e6)},
                'treat_params' : 'fix',
                'startCode' : [166,170],
                'genCode' : [None],
                'lineColor' : 'g',
                'lineStyle' : ':',
                'catColor' : 'lightgreen'}
}

import pickle as pkl

# DCB signal, load parameters from fit to signal sample and fix to values or use constraints
#fitpars_DCB = pkl.load(open('../common/fitpars_CE_DCB.pkl','rb'))
#mom_components['CE']['pdf'] = 'dscb'
#mom_components['CE']['pars'] = fitpars_DCB['best'][0]
#mom_components['CE']['treat_params'] = 'constrain'
#mom_components['CE']['treat_params'] = 'fix'

# GCB signal, load parameters from fit to signal sample and fix to values or use constraints
#fitpars_GCB = pkl.load(open('../common/fitpars_CE_GCB.pkl','rb'))
#mom_components['CE']['pdf'] = 'gcb'
#mom_components['CE']['pars'] = fitpars_GCB['best'][0]
#mom_components['CE']['treat_params'] = 'constrain'
#mom_components['CE']['treat_params'] = 'fix'

# KDE signal, load array of momenta from .pkl to generate KDE
# Momenta are from signal standalone sample, after passing same selection as here
# Stored as dict[reco='best','perfect'][sid=0,1,2]
#mom_components['CE']['pdf'] = 'kde'
#data_dict = pkl.load(open('../common/fitpars_CE_KDE.pkl','rb'))
#mom_components['CE']['pars'] = data_dict['best'][0]
#mom_components['CE']['treat_params'] = 'fix'

################################################################################################
# Below are the different components (sometimes interdependent) for 'theo_exp' parametrization #
# Generically, this is a theo component (which may be a product of several lineshapes)         #
# convolved with an experimental component (which may be a convolution of several sources)     #
# Current default is theory lineshape * efficiency for 'theo', loss X resolution for 'exp'     #
################################################################################################

# First component (here CE) to use 'theo_exp' PDF
from theo_components import binned_spectrum_CeLL
from res_components import res_components
lineshapes_in_CE  = {'theo' : binned_spectrum_CeLL(),
                     'eff'  : '../common/efficiency.pkl'}

# If parametrization for resolution, loss is already known, use this formulation
flat_res  = res_components(params = '../common/fitpars_flat_res_entrance_gcb.pkl',             res_type = 'res',  pdf = 'gcb')
flat_loss = res_components(params = '../common/fitpars_flat_loss_entrance_landau_unbinned.pkl',res_type = 'loss', pdf = 'landau')

# If resolution, loss will be fit simultaneously in independent source, use this formulation
#dict_flat = pkl.load(open('/exp/mu2e/data/users/sdittmer/SignalShape/skimmed_flat_mom_v2.pkl','rb'))
#flat_res  = res_components(p_bins = [95.,97.,99.,101.,103.,105.], simul_source = (dict_flat['entrance']['gen'],dict_flat['entrance']['mc']),  res_type = 'res',  pdf = 'gcb')
#flat_loss = res_components(params = [95.,105.],                   simul_source = (dict_flat['entrance']['mc'], dict_flat['entrance']['reco']),res_type = 'loss', pdf = 'landau')

fitpars_in = {'res'  : flat_res.get_params(), 'loss' : flat_loss.get_params()}

from helper import gen_theo_exp
mom_components['CE']['pdf'] = 'theo_exp'
mom_components['CE']['pars'] = gen_theo_exp(fitpars_in,lineshapes_in_CE)
mom_components['CE']['treat_params'] = 'param'     # If this is the first time parameters are being defined / passed in
#mom_components['CE']['treat_params'] = 'simul'     # If parameters are already defined (simultaneous fit, or not first process to use them)
#mom_components['CE']['nll'] = [flat_res,flat_loss] # If parameters are being fit from independent measurement, provide sources here

# Second component (here DIO) to use 'theo_exp' PDF
# Reusing inputs for efficiency, loss, resolution
lineshapes_in_DIO = {'theo' : '/cvmfs/mu2e.opensciencegrid.org/Musings/Offline/v10_34_00/Offline/ConditionsService/data/czarnecki_szafron_Al_2016.tbl',
                     'eff'  : '../common/efficiency.pkl'}

mom_components['DIO']['pdf'] = 'theo_exp'
mom_components['DIO']['pars'] = gen_theo_exp(fitpars_in,lineshapes_in_DIO)
mom_components['DIO']['treat_params'] = 'simul'

