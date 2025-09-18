# module holds functions associated with plotted reco features
import numpy as np
import awkward as ak
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mom_components import mom_components
from time_components import time_components

def plotmom_fit(data, track_cats, fit_range, list_pdfs, cat=None):
    """
    Configures and draws the 1D histogram of momentum, with the combined fit and residuals plot underneath


    Parameters
    ----------
    data : numpy array (with cuts applied)
        your data array post-processing
    track_cats:
        array of cat numbers, corresponds to order in the mom_component
    fit_range : [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    list_pdfs: (proc,pdfs[proc],norms[proc])
        process, pdf and normalization associated with that process (one per physics process)
    cat: bool
        show the MC truth processes on the histogram

    """
    n_bins = 50
    mom_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data = data[~np.isnan(data)] 
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    
    # run catagorization and plot
    if cat is None:
      print("[py-fitter/recoplot_module/plotmom_fit] ❌ cat option is {cat}, will not include MC truth")
    if cat is not None:
        colors = ['lightgrey']+[idict['catColor'] for idict in mom_components.values()]
        cat_list = []
        data_list = []
        print(len(track_cats),len(data))
        for i in range(len(mom_components)+1):
          cat_list.append([])
          data_list.append([])
        for j in range(len(track_cats)):
          #check there are some non-DIO at this point
          if data[j] != None and data[j] < fit_range[1] and data[j] > fit_range[0] :
            data_list[track_cats[j]].append(data[j])
            cat_list[track_cats[j]].append(track_cats[j])
        for i in range(len(data_list)):

          hists,_,_ = ax1.hist(data_list[i], color=colors[i], bins=n_bins, range=fit_range, histtype='bar',stacked=True)
        print('[py-fitter/recoplot_module/plotmom_fit] ✅ Printing MC Truth ')
        print('Other  :',data_list[0])
        for iproc, proc in enumerate(mom_components.keys()):
            print(iproc, proc.ljust(10)+':', len(data_list[iproc+1]))
    else:
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    # make plot
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
    err = []
    dev = []
    
    # add error bars to residual plot
    for i, ent in enumerate(data_hist):
      if ent != 0:
        err.append(np.sqrt((np.sqrt(data_hist[i]))*(np.sqrt(data_hist[i])) + (np.sqrt(combine_plot[i]))* (np.sqrt(combine_plot[i])))/np.sqrt(data_hist[i]))
        dev.append(np.abs(combine_plot[i] - data_hist[i])/np.sqrt(data_hist[i]))
      else:
        err.append(0)
        dev.append(0)
    if len(data_hist) == 0:
       print('[py-fitter/recoplot_module/plotmom_fit] ⚠️ WARNING! histogram empty')

    ax2.errorbar(mom_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Momentum [MeV/c]')
    ax2.set_ylabel('Normalized Residual')

def plot_time_fit(data, track_cats, fit_range, list_pdfs, cat=None):
    """
    Configures and draws the 1D histogram of time, with the combined fit and residuals plot underneath

    Parameters
    ----------
    data : numpy array (with cuts applied)
        your data array post-processing
    fit_range : [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    list_pdfs: (proc,pdfs[proc],norms[proc])
        process, pdf and normalization associated with that process (one per physics process)
    cat: bool
        show the MC truth processes on the histogram

    """
    
    n_bins = 50
    time_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data = data[~np.isnan(data)] 
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    
    # run catagorization and plot
    if cat is None:
      print("[py-fitter/recoplot_module/plottime_fit] ❌ cat option is {cat}, will not include MC truth")
    if cat is not None:
        colors = ['lightgrey']+[idict['catColor'] for idict in time_components.values()]
        cat_list = []
        data_list = []
        print(len(track_cats),len(data))
        for i in range(len(time_components)+1):
          cat_list.append([])
          data_list.append([])
        for j in range(len(track_cats)):
          #check there are some non-DIO at this point
          if data[j] != None and data[j] < fit_range[1] and data[j] > fit_range[0] :
            data_list[track_cats[j]].append(data[j])
            cat_list[track_cats[j]].append(track_cats[j])
        for i in range(len(data_list)):

          hists,_,_ = ax1.hist(data_list[i], color=colors[i], bins=n_bins, range=fit_range, histtype='bar',stacked=True)
        print('[py-fitter/recoplot_module/plottime_fit] ✅ Printing MC Truth ')
        print('Other  :',data_list[0])
        for iproc, proc in enumerate(time_components.keys()):
            print(iproc, proc.ljust(10)+':', len(data_list[iproc+1]))
    else:
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)

    # make plot
    combine_plot = np.zeros(len(time_plot))
    for name, pdfs, N_pdfs in list_pdfs:
        pdf_plot = (pdfs.pdf(time_plot) * N_pdfs * scale).numpy()
        combine_plot += pdf_plot
        ax1.plot(time_plot, pdf_plot, label=name, color=time_components[name]['lineColor'],linestyle=time_components[name]['lineStyle'])

    ax1.plot(time_plot, combine_plot, '-r', label='Total')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1e-1, max(data_hist)])
    ax1.set_xlabel('Reconstructed Time [ns]')
    ax1.set_ylabel('# of events per bin')
    ax1.legend()
    err = []
    dev = []
    
    # add error bars to residual plot
    for i, ent in enumerate(data_hist):
      if ent != 0:
        err.append(np.sqrt((np.sqrt(data_hist[i]))*(np.sqrt(data_hist[i])) + (np.sqrt(combine_plot[i]))* (np.sqrt(combine_plot[i])))/np.sqrt(data_hist[i]))
        dev.append(np.abs(combine_plot[i] - data_hist[i])/np.sqrt(data_hist[i]))
      else:
        err.append(0)
        dev.append(0)
    if len(data_hist) == 0:
       print('[py-fitter/recoplot_module/plottime_fit] ⚠️ WARNING! histogram empty')

    ax2.errorbar(time_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed time [ns]')
    ax2.set_ylabel('Normalized Residual')

