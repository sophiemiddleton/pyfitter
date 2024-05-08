# Plot class
# Make some plots and histogram on the data

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


def plot_cut(data, data_cuts):
    n_bins = 100
    plot_range = (60, 150)

    fig, ax = plt.subplots(1,1)
    ax.hist(data['deent','mom'], bins=n_bins, range=plot_range, color='blue', label='data (before cut)')
    ax.hist(data_cuts['deent','mom'], bins=n_bins, range=plot_range, color='red', label='data(after cut)')

    ax.set_xlabel('Momentum [MeV]')
    ax.set_ylabel('# of events')
    ax.legend()


def plot_MC(data, feature, n_bins=100, plot_range=None):
    MC_count = count_MC(data)

    data_plot = []
    name_plot = []
    for (index_gen, name_gen, N_gen) in MC_count:
        data_gen = data[data['demc','gen'] == index_gen]
        data_plot.append(ak.to_numpy(data_gen[feature]))
        name_plot.append(name_gen)

    #data_CE = data[data['demc','gen'] == 43]
    #data_DIO = data[data['demc','gen'] == 7]
    #data_Cosmic = data[data['demc','gen'] == 38]

    data_np = ak.to_numpy(data[feature])
    #data_CE_np = ak.to_numpy(data_CE['deent','mom'])
    #data_DIO_np = ak.to_numpy(data_DIO['deent','mom'])
    #data_Cosmic_np = ak.to_numpy(data_Cosmic['deent','mom'])

    data_hist, data_binedge = np.histogram(data_np, bins=n_bins, range=plot_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig,ax = plt.subplots(1,1)
    ax.hist(data_plot, bins=n_bins, range=plot_range, histtype='stepfilled', stacked=True, label=name_plot)

    ax.set_xlabel(feature)
    ax.set_ylabel('# of events')
    ax.legend()


def count_MC(data_MC):
    """MC gen code can be found in Offline/MCDataProducts/inc/GenId.hh"""

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

    MC_count = []
    for index_gen, name_gen in enumerate(gen_code):
        N_gen = ak.num(data_MC[data_MC['demc','gen'] == index_gen], axis=0)
        if N_gen != 0:
            MC_count.append((index_gen, name_gen, N_gen.item()))

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


def plot_fit(data, fit_range, list_pdfs):
    n_bins = 100
    mom_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])

    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='stepfilled', alpha=0.1)
    ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    #set custom cycler
    ax1.set_prop_cycle(custom_cycler)
    combine_plot = np.zeros(len(mom_plot))
    for name, pdfs, N_pdfs in list_pdfs:
        pdf_plot = (pdfs.pdf(mom_plot) * N_pdfs * scale).numpy()
        combine_plot += pdf_plot
        ax1.plot(mom_plot, pdf_plot, label=name)
    #ce_plot = (ce.pdf(mom_plot) * N_CE * scale).numpy()
    #dio_plot = (dio.pdf(mom_plot) * N_DIO * scale).numpy()
    #cosmic_plot = (cosmic.pdf(mom_plot) * N_Cosmic * scale).numpy()
    #combine_plot = ce_plot + dio_plot + cosmic_plot

    #ax1.plot(mom_plot, ce_plot, '--', color='blue', label='CE')
    #ax1.plot(mom_plot, dio_plot, ':', color='green', label='DIO')
    #ax1.plot(mom_plot, cosmic_plot, '-.', color='orange', label='Cosmic')
    ax1.plot(mom_plot, combine_plot, '-r', label='Total')

    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range) #FIXME change range to variables
    ax1.set_ylim([1e-1, 1e3])
    ax1.set_xlabel('Momentum [MeV]')
    ax1.set_ylabel('# of events')
    ax1.legend()

    ax2.errorbar(mom_plot, np.abs(combine_plot - data_hist), yerr=np.sqrt(data_hist), color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)

    ax2.grid(True)
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Momentum [MeV]')
    ax2.set_ylabel('residual')
