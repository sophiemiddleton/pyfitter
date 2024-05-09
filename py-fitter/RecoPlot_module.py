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
