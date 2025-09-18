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
    
   
    
    'CE'     : {'pdf' : 'dscb',
                'pars' : {'mu'     : (104,           103,   107),
                                    'sigma'  : (0.5,           0.08,  2.0),
                                    'alphaL' : (0.422,         0,     10),
                                    'nL'     : (25.1,          0,     100),
                                    'alphaR' : (2.227,         0,     100),
                                    'nR'     : (5.954,         0,     100)},
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
                'catColor' : 'lightgreen'},
   'RPC'    : {'pdf' : 'Gauss',
                'pars' : {'mu'    : (100, 98,   102),
                          'sigma' : (11, 5, 25)},
                'treat_params' : 'float',
                'startCode' : [178,179],
                'genCode' : [None],
                'lineColor' : 'darkorange',
                'lineStyle' : (0, (3, 5, 1, 5)),
                'catColor' : 'orange'}
}

import pickle as pkl

def generate_res_X_lineshape(fitpars_in, lineshape_in) -> dict:
    if isinstance(fitpars_in,str):
        pardict = dict(pkl.load(open(fitpars_in,'rb')))
    else:
        pardict = fitpars_in
    # Lineshape may be .txt file with p, pdf values or .pkl with array of momenta
    try:
        lineshape = pkl.load(open(lineshape_in,'rb'))
        print('Loaded lineshape as pickle!')
    except:
        lineshape = []
        with open(lineshape_in,'r') as f:
            for line in f.readlines():
                parts = line.split()
                lineshape.append[parts[0],parts[1]]
        print('Loaded lineshape as .txt file!')
    pardict['lineshape'] = lineshape
    return pardict

# DCB signal, load parameters from fit to signal sample and fix to values or use constraints
#fitpars_DCB = pkl.load(open('/exp/mu2e/data/users/sdittmer/SignalShape/fitpars_DCB.pkl','rb'))
#mom_components['CE']['pdf'] = 'dscb'
#mom_components['CE']['pars'] = fitpars_DCB['best'][0]
#mom_components['CE']['treat_params'] = 'constrain'
#mom_components['CE']['treat_params'] = 'fix'

# GCB signal, load parameters from fit to signal sample and fix to values or use constraints
#fitpars_GCB = pkl.load(open('/exp/mu2e/data/users/sdittmer/SignalShape/fitpars_GCB.pkl','rb'))
#mom_components['CE']['pdf'] = 'gcb'
#mom_components['CE']['pars'] = fitpars_GCB['best'][0]
#mom_components['CE']['treat_params'] = 'constrain'
#mom_components['CE']['treat_params'] = 'fix'

# KDE signal, load array of momenta from .pkl to generate KDE
# Momenta are from signal standalone sample, after passing same selection as here
# Stored as dict[reco='best','perfect'][sid=0,1,2]
#mom_components['CE']['pdf'] = 'kde'
#data_dict = pkl.load(open('/exp/mu2e/data/users/sdittmer/SignalShape/fitpars_KDE.pkl','rb'))
#mom_components['CE']['pars'] = data_dict['best'][0]
#mom_components['CE']['treat_params'] = 'fix'

# Gen momentum X loss+resolution (GCB), fix resolution parameters to fitted values or constrain within uncertainties
#mom_components['CE']['pdf'] = 'gcb_gen_res'
#mom_components['CE']['pars'] = generate_res_X_lineshape('/exp/mu2e/data/users/sdittmer/SignalShape/fitpars_flat_gen_entrance_gcb.pkl','/exp/mu2e/data/users/sdittmer/SignalShape/sig_p_gen_entrance_best.pkl')
#mom_components['CE']['treat_params'] = 'fix'
#mom_components['CE']['treat_params'] = 'constrain'

# Momentum at plane X resolution (GCB), fix resolution parameters to fitted values or constrain within uncertainties
#mom_components['CE']['pdf'] = 'gcb_mc_res'
#mom_components['CE']['pars'] = generate_res_X_lineshape('/exp/mu2e/data/users/sdittmer/SignalShape/fitpars_flat_mc_entrance_gcb.pkl','/exp/mu2e/data/users/sdittmer/SignalShape/sig_p_mc_entrance_best.pkl')
#mom_components['CE']['treat_params'] = 'fix'
#mom_components['CE']['treat_params'] = 'constrain'

# Gen momentum X loss+resolution, simultaneous fit
#from res_components import res_components
#flat_res = res_components([95., 97., 99., 101., 103., 105.], '/exp/mu2e/data/users/sdittmer/SignalShape/skimmed_flat_mom.pkl', 'gen')
#mom_components['CE']['pdf'] = 'gcb_gen_res'
#mom_components['CE']['nll'] = flat_res
#mom_components['CE']['pars'] = generate_res_X_lineshape(flat_res.params(),'/exp/mu2e/data/users/sdittmer/SignalShape/sig_p_gen_entrance_best.pkl')
#mom_components['CE']['treat_params'] = 'simul'

# Momentum at plane X resolution, simultaneous fit
#from res_components import res_components
#flat_res = res_components([95., 97., 99., 101., 103., 105.], '/exp/mu2e/data/users/sdittmer/SignalShape/skimmed_flat_mom.pkl', 'mc')
#mom_components['CE']['pdf'] = 'gcb_mc_res'
#mom_components['CE']['nll'] = flat_res
#mom_components['CE']['pars'] = generate_res_X_lineshape(flat_res.params(),'/exp/mu2e/data/users/sdittmer/SignalShape/sig_p_mc_entrance_best.pkl')
#mom_components['CE']['treat_params'] = 'simul'

# If you want to use the same resolution with DIO, you should be able to do something like
#mom_components['DIO']['pdf'] = 'gcb_mc_res'
#mom_components['DIO']['pars'] = generate_res_X_lineshape(flat_res.params(),<DIO lineshape .pkl goes here>)
#mom_components['DIO']['treat_params'] = 'simul'
