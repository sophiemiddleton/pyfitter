# file holds a list of systematic uncertainties and their expected scale
"""
Here I list some of the major sources of mostly uncertainties.
Each background is characterized as follows:

* 'type' : 
      - a 'shift' is a straight +/- Value in units of MeV/c
      - a 'frac' means its a percentage on the chosen component
      - a 'shape' means that this is a shape uncertainty and the noted parameter in the shape has a 1sigma error of the quoted value
* 'sim' :
      - True means that  this is currently from simualtion, several of these can be measured using early data
* 'process'
      - describes the physics process that uncertainty relates to, can be 'all'
* 'component'
      - where to apply this uncertainty
        - 'mom' i.e. this has specific impact on the momentum spectrum e.g. a shift in scale
        - 'both' this could effect the yield of events in both momentum and time
"""
sysunc_components = {
    ##### General #######
    'Abs_Mom_Scale' : {
                'type' : 'shift',
                'sim' : True
                'process': 'all',
                'component' : ['mom'],
                'value' : [0.1, 0.1]# MeV - SU2020 discussed for DIO, resulting in large uncertainty on cut and count [plus, minus] allowing for assymetric values
                }
    'Mom_Res' : {?}
           
      ###### DIO #########
      'DIO_Theory' : {
                'type' : 'frac',
                'sim' : True
                'process' : 'DIO'
                'component' : ['mom','time'],
                'value' : [0.025, 0.025]
                }
                
       ###### RPC ########
       'RPC_rate' : {
                'type' : 'frac',
                'sim' : True,
                'process' : 'RPC'
                'component' : ['mom','time'],
                'value' : [0.093, 0.093] # from use of magneisum = 9.3%
       
       }
       'pion_rate' : {
                'type' : 'frac',
                'sim' : True,
                'process' : 'RPC'
                'component' : ['mom','time'],
                'value' : [0.27, 0.09] # -27% to +9 % from G4 studies
       
       }
       'internalconv_rate' : {
                'type' : 'frac', # means that 0.025=2.5% not necessarily 2.5 MeV
                'sim' : True, # means that we expect to have a better value from a data driven value
                'process' : 'RPC',
                'component' : ['mom','time'],
                'value' : [0.0045, 0.0045] # -27% to +9 % from G4 studies
       
       }
       # What's missing? OOT RPC, extinction uncertainty etc. 
       ##### Cosmics #######   
       'CRV_eff' : {
                'type' : 'frac',
                'sim' : True,
                'process' : 'Cosmics',
                'component' : ['mom','time'],
                'value' : [0.04, 0.04] # from CRV studies detailed in SU2020
       
       }
       'generator' : {
                'type' : 'frac',
                'sim' : True,
                'process' : 'Cosmics',
                'component' : ['mom','time'],
                'value' : [0.20,0.20] # from comparisons of generators
       
       }
       # What's missing? CRV aginig
      }
