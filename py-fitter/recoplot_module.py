# module holds functions associated with plotted reco features
import numpy as np
import awkward as ak
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mom_components import mom_components
from time_components import time_components

def plotmom_fit(data, fit_range, list_pdfs, cat=None):
    """ plot the final plot with fit and data overlay, plus a residual plot """
    n_bins = 50
    mom_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])

    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])

    if cat is not None:
        colors = ['lightgrey']+[idict['catColor'] for idict in mom_components.values()]
        hists,_,_ = ax1.hist([data[cat==icat] for icat in range(len(mom_components)+1)], color=colors, bins=n_bins, range=fit_range, histtype='bar',stacked=True)
        print('Other  :',np.sum(hists[0]))
        for iproc, proc in enumerate(mom_components.keys()):
            print(proc.ljust(10)+':',np.sum(hists[iproc+1])-np.sum(hists[iproc]))
    else:
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    combine_plot = np.zeros(len(mom_plot))
    for name, pdfs, N_pdfs in list_pdfs:
        pdf_plot = (pdfs.pdf(mom_plot) * N_pdfs * scale).numpy()
        combine_plot += pdf_plot
        ax1.plot(mom_plot, pdf_plot, label=name, color=mom_components[name]['lineColor'],linestyle=mom_components[name]['lineStyle'])

    ax1.plot(mom_plot, combine_plot, '-r', label='Total')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1e-1, max(data_hist)])
    ax1.set_xlabel('Reconstructed Momentum [MeV/c]')
    ax1.set_ylabel('# of events per bin')
    ax1.legend()
    err = np.sqrt((np.sqrt(data_hist))*(np.sqrt(data_hist)) + (np.sqrt(combine_plot))* (np.sqrt(combine_plot)))/np.sqrt(data_hist) #FIXME - throws error if /0
    ax2.errorbar(mom_plot, np.abs(combine_plot - data_hist)/np.sqrt(data_hist), yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3) #FIXME error if data_hist empty

    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
    ax2.set_ylabel('Normalized Residual')

def plot_time_fit(data, fit_range, list_pdfs, cat=None):
    """ plot the final plot with fit and data overlay, plus a residual plot """
    n_bins = 50
    time_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])

    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    if cat is not None:
        colors = ['lightgrey']+[idict['catColor'] for idict in components.values()]
        hists,_,_ = ax1.hist([data[cat==icat] for icat in range(len(components)+1)], color=colors, bins=n_bins, range=fit_range, histtype='bar',stacked=True)
        print('Other  :',np.sum(hists[0]))
        for iproc, proc in enumerate(components.keys()):
            print(proc.ljust(10)+':',np.sum(hists[iproc+1])-np.sum(hists[iproc]))
    else:
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')

    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    combine_plot = np.zeros(len(time_plot))
    for name, pdfs, N_pdfs in list_pdfs:
        pdf_plot = (pdfs.pdf(time_plot) * N_pdfs * scale).numpy()
        combine_plot += pdf_plot
        ax1.plot(time_plot, pdf_plot, label=name)

    ax1.plot(time_plot, combine_plot, '-r', label='Total')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([10, max(data_hist)])
    ax1.set_xlabel('Reconstructed Time [ns]')
    ax1.set_ylabel('# of events per bin')
    ax1.legend()
    err = np.sqrt((np.sqrt(data_hist))*(np.sqrt(data_hist)) + (np.sqrt(combine_plot))* (np.sqrt(combine_plot)))/np.sqrt(data_hist) #FIXME - throws error if /0
    ax2.errorbar(time_plot, np.abs(combine_plot - data_hist)/np.sqrt(data_hist), yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3) #FIXME error if data_hist empty

    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Time [ns]')
    ax2.set_ylabel('Normalized Residual')
