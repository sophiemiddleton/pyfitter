# Define fitting and plotting options for all components to include in fit
# pdf                          : model type from momPDF_module.py
# startCode, genCode, catColor : with --categorize, events with this set of startCode and genCode will be colored catColor in the data-fit plot
#                              : order of components gives stacking order (bottom to top)
# lineColor, lineStyle         : line color and style for the PDF component in the data-fit plot (see matplotlib documentation for options)

time_components = {
    'Cosmic' : {'pdf' : 'uniform',
                'pars' : None,
                'startCode' : [None],
                'genCode' : [44,38],
                'lineColor' : 'm',
                'lineStyle' : '-.',
                'catColor' : 'violet'},
    
    
    
    'Muon'     : {'pdf' : 'muexp',
                'pars' : {'decay_rate_mu'    : (-0.001157, -0.0015, -0.001)},
                'startCode' : [168,166,170],
                'genCode' : [None],
                'lineColor' : 'b',
                'lineStyle' : '--',
                'catColor' : 'lightskyblue'},
                  
                
      'RPC'    : {'pdf' : 'piexp',
                'pars' : {'decay_rate_pi'    : (-0.03846,  -0.04, -0.01)},
                'startCode' : [178,179],
                'genCode' : [None],
                'lineColor' : 'black',
                'lineStyle' : (0, (3, 5, 1, 5)),
                'catColor' : 'black'}
}
