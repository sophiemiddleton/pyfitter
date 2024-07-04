# Make some plots with features from MC

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from cycler import cycler

# setup a cycler to get a different default color and linestyle
custom_cycler = (cycler(color=list('bgm')) +
                 cycler(linestyle=['--', ':', '-.']))


def plot_feature(data, feature, n_bins=100, plot_range=None):

    fig, ax = plt.subplots(1,1)
    ax.hist(data[feature], bins=n_bins, range=plot_range, label=feature)

    ax.set_xlabel(feature)
    ax.set_ylabel('# of events')
    ax.legend()

def plot_MC(data, feature, n_bins=100, plot_range=None): # FIXME - won't work for MDC2024
    MC_count = count_MC(data)

    data_plot = []
    name_plot = []
    for (index_gen, name_gen, N_gen) in MC_count:
        data_gen = data[data['demmcsim','gen'] == index_gen]
        data_plot.append(ak.to_numpy(data_gen[feature]))
        name_plot.append(name_gen)

    data_np = ak.to_numpy(data[feature])
    data_hist, data_binedge = np.histogram(data_np, bins=n_bins, range=plot_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig,ax = plt.subplots(1,1)
    ax.hist(data_plot, bins=n_bins, range=plot_range, histtype='stepfilled', stacked=True, label=name_plot)

    ax.set_xlabel(feature)
    ax.set_ylabel('# of events')
    ax.legend()


def count_MC(data_MC):
    """MC gen code can be found in Offline/MCDataProducts/inc/GenId.hh"""

    # MC gen code can be found in Offline/MCDataProducts/inc/genId.hh
    gen_code = ['unknown', 'particleGun', 'CeEndpoint',
                'cosmicToy', 'cosmicDYB', 'cosmic', 'obsolete1', #6
                'dioTail', 'obsolete2', 'obsolete3', 'obsolete4', 'ExternalRPC', #11
                'muonCapture', 'muonDecayInFlight', 'ejectedProtonGun', #14
                'piEplusNuGun', 'primaryProtonGun', 'fromG4BLFile', 'ePlusfromStoppedPi', #18
                'ejectedNeutronGun', 'ejectedPhotonGun', 'nuclearCaptureGun', 'InternalRPC', #22
                'extMonFNALGun', 'fromStepPointMCs', 'stoppedMuonGun', 'PiCaptureCombined', #26
                'MARS', 'StoppedParticleReactionGun', 'bremElectronGun', 'muonicXRayGun', #30
                'fromSimParticleStartPoint', 'fromSimParticleCompact', 'StoppedParticleG4Gun', #33
                'CaloCalib', 'InFlightParticleSampler', 'muplusDecayGun', 'StoppedMuonXRayGammaRayGun', #37
                'cosmicCRY', 'pbarFlat', 'fromAscii', 'ExternalRMC', 'InternalRMC', 'CeLeadingLog', 'cosmicCORSIKA', #44
                'MuCapProtonGenTool', 'MuCapDeuteronGenTool', 'DIOGenTool', 'MuCapNeutronGenTool', #48
                'MuCapPhotonGenTool', 'MuCapGammaRayGenTool', 'CeLeadingLogGenTool', 'MuplusMichelGenTool', #52
                'gammaPairProduction', #53
                'lastEnum' #54
                ]

    # MC process code can be found in Offline/MCDataProducts/inc/ProcessCode
    proc_code = [
          'unknown',                'AlphaInelastic',          'annihil',             'AntiLambdaInelastic', # 3
          'AntiNeutronInelastic',   'AntiOmegaMinusInelastic', 'AntiProtonInelastic', 'AntiSigmaMinusInelastic', # 7
          'AntiSigmaPlusInelastic', 'AntiXiMinusInelastic',    'AntiXiZeroInelastic', 'CHIPSNuclearCaptureAtRest', # 11
          'compt',                  'conv',                    'Decay',               'DeuteronInelastic', # 15
          'eBrem',                  'eIoni',                   'ElectroNuclear',      'hBrems', # 19
          'hElastic',               'hIoni',                   'hPairProd',           'ionIoni',  # 23
          'KaonMinusInelastic',     'KaonPlusInelastic',       'KaonZeroLInelastic',  'KaonZeroSInelastic', # 27
          'LambdaInelastic',        'msc',                     'muBrems',             'muIoni',  # 31
          'muMinusCaptureAtRest',   'muMsc',                   'muPairProd',          'nCapture',  # 35
          'NeutronInelastic',       'nFission',                'nKiller',             'OmegaMinusInelastic', # 39
          'phot',                   'PhotonInelastic',         'PionMinusInelastic',  'PionPlusInelastic', # 43
          'PositronNuclear',        'ProtonInelastic',         'SigmaMinusInelastic', 'SigmaPlusInelastic', # 47
          'StepLimiter',            'Transportation',          'TritonInelastic',     'XiMinusInelastic', # 51
          'XiZeroInelastic',        'mu2eLowEKine',            'mu2eKillerVolume',    'mu2eMaxSteps',  # 55
          'mu2ePrimary',            'mu2eSpecialCutsProcess',  'hadElastic',          'CoulombScat', # 59
          'nuclearStopping',        'mu2eMaxGlobalTime',       'TNuclearCapture',     'muMinusAtomicCapture', # 63
          'MuAtomDecay',            'Rayl',                    'ionInelastic',        'He3Inelastic', # 67
          'alphaInelastic',         'AntiHe3InelasticProcess', 'AntiAlphaInelasticProcess', 'AntiDeuteronInelastic', # 71
          'dInelastic',             'tInelastic',              'RadioactiveDecay',    'CHIPS_Inelastic', # 75
          'NotSpecified',           'hFritiofCaptureAtRest',   'hBertiniCaptureAtRest', 'AntiTritonInelasticProcess', # 79
          'anti_He3Inelastic',      'anti_alphaInelastic',     'anti_deuteronInelastic', 'anti_lambdaInelastic', # 83
          'anti_neutronInelastic',  'anti_omega_MinusInelastic', 'anti_protonInelastic', 'anti_sigma_PlusInelastic',  # 87
          'anti_sigma_MinusInelastic', 'anti_tritonInelastic', 'anti_xi_MinusInelastic', 'anti_xi0Inelastic',  # 91
          'kaon_PlusInelastic',     'kaon_MinusInelastic',     'kaon0LInelastic',     'kaon0SInelastic', # 95
          'lambdaInelastic',        'neutronInelastic',        'omega_MinusInelastic', 'pi_PlusInelastic',  # 99
          'pi_MinusInelastic',      'protonInelastic',         'sigma_PlusInelastic', 'sigma_MinusInelastic', # 103
          'sigma0Inelastic',        'xi_MinusInelastic',       'xi0Inelastic',        'positronNuclear', # 107
          'electronNuclear',        'photonNuclear',           'antilambdaInelastic', 'DecayWithSpin', # 111
          'ionElastic',             'EMCascade',               'DIO',                 'NuclearCapture', # 115
          'muonNuclear',            'GammaToMuPair',           'AnnihiToMuPair',      'ee2hadr', # 119
          'G4MinEkineCuts',         'G4MaxTimeCuts',           'OpAbsorption',        'OpBoundary', # 123
          'Scintillation',          'inelastic',               'G4ErrorEnergyLoss',   'G4ErrorStepLengthLimit', # 127
          'G4ErrorMagFieldLimit',   'ePairProd',               'mu2eFieldPropagator', 'mu2eRecorderProcess',  # 131
          'mu2eProtonInelastic',    'RadioactiveDecayBase',    'B_PlusInelastic',     'B_MinusInelastic', # 135
          'B0Inelastic',            'Bc_PlusInelastic',        'Bc_MinusInelastic',   'Bs0Inelastic', # 139
          'D_PlusInelastic',        'D_MinusInelastic',        'D0Inelastic',         'Ds_PlusInelastic', # 143
          'Ds_MinusInelastic',      'anti_B0Inelastic',        'anti_Bs0Inelastic',   'anti_D0Inelastic', # 147
          'anti_lambda_bInelastic', 'anti_lambda_c_PlusInelastic', 'anti_omega_b_MinusInelastic', 'anti_omega_c0Inelastic', # 151
          'anti_xi_b_MinusInelastic', 'anti_xi_b0Inelastic',   'anti_xi_c_PlusInelastic', 'anti_xi_c0Inelastic', # 155
          'lambda_bInelastic',      'lambda_c_PlusInelastic',  'omega_b_MinusInelastic', 'omega_c0Inelastic', # 159
          'xi_b_MinusInelastic',    'xi_b0Inelastic',          'xi_c_PlusInelastic',  'xi_c0Inelastic', # 163
          # stopped-muon physics processes, specific to Mu2e
          'truncated',       'mu2eMuonCaptureAtRest',  'mu2eMuonDecayAtRest',       'mu2eCeMinusEndpoint', # 167
          'mu2eCeMinusLeadingLog',   'mu2eCePlusEndpoint',  'mu2eDIOLeadingLog', 'mu2eInternalRMC',  # 171
          'mu2eExternalRMC',         'mu2eFlateMinus',      'mu2eFlatePlus', 'mu2eFlatPhoton', # 175
          'mu2eCePlusLeadingLog', 'mu2ePionCaptureAtRest', 'mu2eExternalRPC', 'mu2eInternalRPC', # 179
          'mu2eCaloCalib', 'mu2ePienu', 'mu2eunused7', 'mu2eunused8', # 183
          'uninitialized', 'NoProcess', 'GammaGeneralProc', # 186
          'mu2eGammaConversion', 'Radioactivation', 'nCaptureHP', 'nFissionHP', # 190
          'lastEnum'
          ]

    array_gen = ak.flatten(data_MC['demmcsim','gen',...,:1], axis=None)
    gen_count = []
    print('gen code counts:')
    for index_gen, name_gen in enumerate(gen_code):
        N_gen = ak.num(array_gen[array_gen == index_gen], axis=0)
        if N_gen != 0:
            gen_count.append(( name_gen, N_gen.item()))


    array_proc = ak.flatten(data_MC['demmcsim','startCode',...,:1], axis=None)
    proc_count = []
    print('process code counts:')
    for index_proc, name_proc in enumerate(proc_code):
        N_proc = ak.num(array_proc[array_proc == index_proc], axis=0)
        if N_proc != 0:
            proc_count.append(( name_proc, N_proc.item()))

    MC_count = gen_count + proc_count

    return MC_count


def plot_MC_comparison(MC_count, result):
    """Compare number of each particles to MC data"""

    # Extract the data from MC count
    # Use switch case syntax only available in python 3.10+
    for (index_gen, name_gen, N_gen) in MC_count:
        if name_gen == 'CeLeadingLog':
            MC_count_CE = N_gen
        elif name_gen == 'dioTail':
            MC_count_DIO = N_gen
        elif name_gen == 'cosmicCRY':
            MC_count_cosmic = N_gen

    MC_count_total = MC_count_CE + MC_count_DIO + MC_count_cosmic

    # Extract the data from fit result
    count_CE = result.params['N_CE']['value']
    count_CE_err = np.abs([[result.params['N_CE']['errors']['lower']], [result.params['N_CE']['errors']['upper']]])
    count_DIO = result.params['N_DIO']['value']
    count_DIO_err = np.abs([[result.params['N_DIO']['errors']['lower']], [result.params['N_DIO']['errors']['upper']]])
    count_cosmic = result.params['N_cosmic']['value']
    count_cosmic_err = np.abs([[result.params['N_cosmic']['errors']['lower']], [result.params['N_cosmic']['errors']['upper']]])

    count_total = count_CE + count_DIO + count_cosmic
    count_total_err = np.sqrt(count_CE_err**2 + count_DIO_err**2 + count_cosmic_err**2)

    # plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    ax1.plot('CE', MC_count_CE, '*b', markersize=10)
    ax1.errorbar('CE', count_CE, yerr=count_CE_err, markersize=10, color='None', marker='+', markerfacecolor='blue', ecolor='blue', capsize=3)
    ax2.plot('DIO', MC_count_DIO, '*g', markersize=10)
    ax2.errorbar('DIO', count_DIO, yerr=count_DIO_err, markersize=10, color='None', marker='+', markerfacecolor='green', ecolor='green', capsize=3)
    ax3.plot('Cosmic', MC_count_cosmic, '*m', markersize=10)
    ax3.errorbar('Cosmic', count_cosmic, yerr=count_cosmic_err, markersize=10, color='None', marker='+', markerfacecolor='magenta', ecolor='magenta', capsize=3)
    ax4.plot('Total', MC_count_total, '*r', markersize=10)
    ax4.errorbar('Total', count_total, yerr=count_total_err, markersize=10, color='None', marker='+', markerfacecolor='red', ecolor='red', capsize=3)
