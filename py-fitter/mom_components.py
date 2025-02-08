# Define fitting and plotting options for all components to include in fit
# pdf                          : model type from momPDF_module.py
# startCode, genCode, catColor : with --categorize, events with this set of startCode and genCode will be colored catColor in the data-fit plot
#                              : order of components gives stacking order (bottom to top)
# lineColor, lineStyle         : line color and style for the PDF component in the data-fit plot (see matplotlib documentation for options)

components = {
    'Cosmic' : {'pdf' : 'uniform',
                'pars' : None,
                'startCode' : [None],
                'genCode' : [44],
                'lineColor' : 'm',
                'lineStyle' : '-.',
                'catColor' : 'violet'},
    
    'RPC'    : {'pdf' : 'Gauss',
                'pars' : {'mu'    : (100, 95,   115),
                          'sigma' : (0.5, 1e-3, 1e3)},
                'startCode' : [178,179],
                'genCode' : [None],
                'lineColor' : 'darkorange',
                'lineStyle' : (0, (3, 5, 1, 5)),
                'catColor' : 'orange'},
    
    'CE'     : {'pdf' : 'dscb',
                'pars' : {'mu'     : (104,   103,  107),
                          'sigma'  : (0.5,   0.08, 2.0),
                          'alphal' : (0.422, 0,    10),
                          'nl'     : (25.1,  0,    100),
                          'alphar' : (2.227, 0,    100),
                          'nr'     : (5.954, 0,    100)},
                'startCode' : [168],
                'genCode' : [None],
                'lineColor' : 'b',
                'lineStyle' : '--',
                'catColor' : 'lightskyblue'},

    'DIO'    : {'pdf' : 'poly58',
                'pars' : {'N' : (55000, 0, 1e6)},
                'startCode' : [166,170],
                'genCode' : [None],
                'lineColor' : 'g',
                'lineStyle' : ':',
                'catColor' : 'lightgreen'}
}
