# module holds functions associated with plotted reco features
import numpy as np
import awkward as ak
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mom_components import mom_components
from time_components import time_components
from pyutils.pylogger import Logger

# Module logger
try:
  logger = Logger(print_prefix='[recoplot_module] ', verbosity=2)
except Exception:
  logger = None


def plot_variable(val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], density=False):
  """
  Plots distributions of the given parameter (val), splitting by process code

  Args:
      val : list of values e.g. rmax
      val_label : text formated value name e.g. "rmax"
      lo : plot range lower bound
      hi : plot range upper bound
      cut_lo : lower cut choice
      cut_hi : upper cut choice
      mc_counts : list of process codes

  Returns:
      plots saved as pdfs
      
  Useage:
    columns = ["legend title"]
    plot_variable(rmax, "rmax", "rmax",300,750, [450,450],[680,680], mc_count,columns)
  """
  sets = []
  cols = ['magenta','orange','orange','black','cyan','grey','green','blue']
  labs = ['cosmic','irpc','erpc','irmc','ermc','ipa dio','dio', 'signal']
  styles = ['bar','step','step']
  lines=["","-","--"]
  alphas = [0.2,1,1]
  for i, val in enumerate(val_overlay):
    
    val = ak.drop_none(val)
    val_signal = ak.mask(val, mc_count == 168)
    val_signal = np.array(ak.flatten(val_signal, axis=None))
    val_cosmics = ak.mask(val, mc_count == -1)
    val_cosmics = np.array(ak.flatten(val_cosmics, axis=None))
    val_dio = ak.mask(val, mc_count == 166)
    val_dio = np.array(ak.flatten(val_dio, axis=None))
    val_erpc = ak.mask(val, mc_count == 178)
    val_erpc = np.array(ak.flatten(val_erpc,axis=None))
    val_irpc = ak.mask(val, mc_count == 179)
    val_irpc = np.array(ak.flatten(val_irpc,axis=None))
    val_ermc = ak.mask(val, mc_count == 171)
    val_ermc = np.array(ak.flatten(val_ermc,axis=None))
    val_irmc = ak.mask(val, mc_count == 172)
    val_irmc = np.array(ak.flatten(val_irmc,axis=None))
    val_ipa = ak.mask(val, mc_count == 0)
    val_ipa = np.array(ak.flatten(val_ipa,axis=None))
    plt.yscale('log')
    sets.append([val_cosmics,val_irpc,val_erpc,val_irmc,val_ermc,val_ipa, val_dio, val_signal])
  for i in range(0,len(sets)):
    dummy_handle = plt.plot([], marker="",color='white', label=columns[i])
    n,bins,patch = plt.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=50, histtype=styles[i], linestyle=lines[i],alpha=alphas[i], stacked=True, density=density)
  plt.xlabel(str(val_label))
  # draw cuts
  plt.plot(cut_lo, [0,1000], 'k--')
  plt.plot(cut_hi, [0,1000], 'k--')
  
  plt.legend(ncol=len(columns),loc='upper center')

  plt.savefig(str(filenames)+"_selection.pdf")
  plt.show()

def plotmom_fit(mom_mag,mc_count, fit_range, list_pdfs, cat=None):
    """
    Configures and draws the 1D histogram of momentum, with the combined fit and residuals plot underneath


    Parameters
    ----------
    data : numpy array (with cuts applied)
        your data array post-processing
    mc_count:
        array of cat numbers, corresponds to order in the mom_component
    fit_range : [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    list_pdfs: (proc,pdfs[proc],norms[proc])
        process, pdf and normalization associated with that process (one per physics process)
    cat: bool
        show the MC truth processes on the histogram

    """
    mom_mag_skim = ak.nan_to_none(mom_mag)
    mom_mag_skim = ak.drop_none(mom_mag_skim)
    data = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
    n_bins = 50
    mom_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data = data[~np.isnan(data)] 
    
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    
    # run catagorization and plot
    if cat is None:
      if logger:
        logger.log('cat option is None; will not include MC truth', 'info')
      else:
        print("[py-fitter/recoplot_module/plotmom_fit] ❌ cat option is {cat}, will not include MC truth")
    
    if cat is not None:
        mom_mag = ak.drop_none(mom_mag)

        data_signal = ak.mask(mom_mag, mc_count == 168)
        data_signal = np.array(ak.flatten(data_signal, axis=None))
        data_signal = [x for x in data_signal if 95 <= x <= 115]

        data_cosmics = ak.mask(mom_mag, mc_count == -1)
        data_cosmics = np.array(ak.flatten(data_cosmics, axis=None))
        data_cosmics = [x for x in data_cosmics if 95 <= x <= 115]
        
        data_dio = ak.mask(mom_mag, mc_count == 166)
        data_dio = np.array(ak.flatten(data_dio, axis=None))
        data_dio = [x for x in data_dio if 95 <= x <= 115]
        
        data_erpc = ak.mask(mom_mag, mc_count == 178)
        data_erpc = np.array(ak.flatten(data_erpc,axis=None))
        data_erpc = [x for x in data_erpc if 95 <= x <= 115]
        
        data_irpc = ak.mask(mom_mag, mc_count == 179)
        data_irpc = np.array(ak.flatten(data_irpc,axis=None))
        data_irpc = [x for x in data_irpc if 95 <= x <= 115]
        
        data_ipa = ak.mask(mom_mag, mc_count == 0)
        data_ipa = np.array(ak.flatten(data_ipa,axis=None))
        data_ipa = [x for x in data_ipa if 95 <= x <= 115]
        
        datasets = [data_cosmics,data_irpc,data_erpc,data_ipa, data_dio, data_signal]
        colors = ['violet','darkorange','grey','yellow','lightgreen','lightskyblue']
        labs_true = ['Cosmic','iRPC','eRPC','IPA DIO','Trgt DIO', ' CE']
        datasets_filled = []
        colors_filled = []
        labels_filled = []
        for i, dat in enumerate(datasets):
          if len(dat) !=0:
            datasets_filled.append(dat)
            colors_filled.append(colors[i])
            labels_filled.append(labs_true[i])
        dummy_handle1 = ax1.plot([], marker="",color='white', label="Reco. MC")
        n,bins,patch = ax1.hist(datasets_filled,range=(fit_range[0],fit_range[1]), color=colors_filled, label=labels_filled, bins=50, histtype="bar", stacked=True)

        
        """
        for iproc, proc in enumerate(mom_components.keys()):
            print(iproc, proc.ljust(10)+':', len(sets[iproc+1]))
        """
        if logger:
          logger.log('======= True Events in Fit Range =======', 'info')
          logger.log(f'N Cosmics {len(data_cosmics)}', 'info')
          logger.log(f'N iRPC {len(data_irpc)}', 'info')
          logger.log(f'N eRPC {len(data_erpc)}', 'info')
          logger.log(f'N IPA {len(data_ipa)}', 'info')
          logger.log(f'N DIO {len(data_dio)}', 'info')
          logger.log(f'N CELL {len(data_signal)}', 'info')
        else:
          print("======= True Events in Fit Range =======")
          print("N Cosmics", len(data_cosmics))
          print("N iRPC", len(data_irpc))
          print("N eRPC", len(data_erpc))
          print("N IPA", len(data_ipa))
          print("N DIO", len(data_dio))
          print("N CELL", len(data_signal))
    else:
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    dummy_handle3 = ax1.plot([], marker="+",color='black', label="Mock Data")
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)
    
    # make plot
    combine_plot = np.zeros(len(mom_plot))
    labs_fit = []
    dummy_handle2 = ax1.plot([], marker="",color='white', label="Fit Components")
    for name, pdfs, N_pdfs in list_pdfs:
        pdf_plot = (pdfs.pdf(mom_plot) * N_pdfs * scale).numpy()
        combine_plot += pdf_plot
        labs_fit.append(name)
        ax1.plot(mom_plot, pdf_plot, label=name, color=mom_components[name]['lineColor'],linestyle=mom_components[name]['lineStyle'])

    ax1.plot(mom_plot, combine_plot, '-r', label='Total')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1e-1, max(data_hist)])
    ax1.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=16)
    ax1.set_ylabel('# of events per bin', fontsize=16)
    leg = ax1.legend(fontsize='large')
    legend_texts = leg.get_texts()
    if len(legend_texts) > 0:
      legend_texts[0].set_weight('bold')
    if len(legend_texts) > 7:
      legend_texts[7].set_weight('bold')
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
       if logger:
         logger.log('histogram empty', 'info')
       else:
         print('[py-fitter/recoplot_module/plotmom_fit] ⚠️ WARNING! histogram empty')

    ax2.errorbar(mom_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=16)
    ax2.set_ylabel('Normalized Residual', fontsize=16)
    
 
def plotmom_fit_old(data, track_cats, fit_range, list_pdfs, cat=None):
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
      if logger:
        logger.log('cat option is None; will not include MC truth', 'info')
      else:
        print("[py-fitter/recoplot_module/plotmom_fit] ❌ cat option is {cat}, will not include MC truth")
    
    if cat is not None:
        colors = ['lightgrey']+[idict['catColor'] for idict in mom_components.values()]
        cat_list = []
        data_list = []
        if logger:
          logger.log(f'len(track_cats)={len(track_cats)}, len(data)={len(data)}', 'max')
        else:
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
        if logger:
          logger.log('Printing MC Truth', 'info')
          logger.log(f'Other : {data_list[0]}', 'max')
        else:
          print('[py-fitter/recoplot_module/plotmom_fit] ✅ Printing MC Truth ')
          print('Other  :',data_list[0])
        for iproc, proc in enumerate(mom_components.keys()):
          if logger:
            logger.log(f'{iproc} {proc.ljust(10)}: {len(data_list[iproc+1])}', 'max')
          else:
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
    ax1.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=16)
    ax1.set_ylabel('# of events per bin', fontsize=16)
    ax1.legend(fontsize='large')
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
       if logger:
         logger.log('histogram empty', 'info')
       else:
         print('[py-fitter/recoplot_module/plotmom_fit] ⚠️ WARNING! histogram empty')

    ax2.errorbar(mom_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=16)
    ax2.set_ylabel('Normalized Residual', fontsize=16)

def plottime_fit(time,mc_count, fit_range, list_pdfs, cat=None):
    """
    Configures and draws the 1D histogram of time, with the combined fit and residuals plot underneath


    Parameters
    ----------
    data : numpy array (with cuts applied)
        your data array post-processing
    mc_count:
        array of cat numbers, corresponds to order in the time_component
    fit_range : [float, float]
        min and max of fit ranges for each dimension (args in the main function)
    list_pdfs: (proc,pdfs[proc],norms[proc])
        process, pdf and normalization associated with that process (one per physics process)
    cat: bool
        show the MC truth processes on the histogram

    """
    time_skim = ak.nan_to_none(time)
    time_skim = ak.drop_none(time_skim)
    data = ak.to_numpy(ak.flatten(time_skim, axis=None))
    n_bins = 50
    time_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data = data[~np.isnan(data)] 
    
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    
    # run catagorization and plot
    if logger:
      logger.log(f'Cat {cat}', 'max')
    else:
        print("Cat",cat)
    if cat is None:
      if logger:
        logger.log('cat option is None; will not include MC truth', 'info')
      else:
        print("[py-fitter/recoplot_module/plottime_fit] ❌ cat option is {cat}, will not include MC truth")
    
    if cat is not None:
        # Validate mc_count is an array aligned with `time`; if not, skip categorization
        valid_mc = True
        try:
            mc_len = len(mc_count)
            time_len = len(ak.flatten(time, axis=None))
            if mc_len != time_len:
                valid_mc = False
        except Exception:
            valid_mc = False

        if not valid_mc:
            if logger:
                logger.log('mc_count not valid or length mismatch; skipping MC categorization', 'info')
            else:
                print('[py-fitter/recoplot_module/plottime_fit] ⚠️ mc_count invalid; skipping MC categorization')
            cat = None

    if cat is not None:
        time = ak.drop_none(time)
        if logger:
          logger.log('filling list', 'max')
        else:
          print("filling list")
        data_signal = ak.mask(time, mc_count == 168)
        data_signal = np.array(ak.flatten(data_signal, axis=None))
        #data_signal = [x for x in data_signal if 475 <= x <= 1650]

        data_cosmics = ak.mask(time, mc_count == -1)
        data_cosmics = np.array(ak.flatten(data_cosmics, axis=None))
        #data_cosmics = [x for x in data_cosmics if 475 <= x <= 1650]
        
        data_dio = ak.mask(time, mc_count == 166)
        data_dio = np.array(ak.flatten(data_dio, axis=None))
        #data_dio = [x for x in data_dio if 475 <= x <= 1650]
        
        data_erpc = ak.mask(time, mc_count == 178)
        data_erpc = np.array(ak.flatten(data_erpc,axis=None))
        #data_erpc = [x for x in data_erpc if 475 <= x <= 1650]
        
        data_irpc = ak.mask(time, mc_count == 179)
        data_irpc = np.array(ak.flatten(data_irpc,axis=None))
        #data_irpc = [x for x in data_irpc if 475 <= x <= 1650]
        
        data_ermc = ak.mask(time, mc_count == 172)
        data_ermc = np.array(ak.flatten(data_ermc,axis=None))
        #data_erpc = [x for x in data_erpc if 475 <= x <= 1650]
        
        data_irmc = ak.mask(time, mc_count == 171)
        data_irmc = np.array(ak.flatten(data_irmc,axis=None))
        
        data_ipa = ak.mask(time, mc_count == 0)
        data_ipa = np.array(ak.flatten(data_ipa,axis=None))
        #data_ipa = [x for x in data_ipa if 475 <= x <= 1650]
        
        datasets = [data_cosmics,data_irpc,data_erpc,data_ipa, data_irmc, data_ermc, data_dio, data_signal]
        colors = ['violet','darkorange','grey','yellow','magenta','cyan','lightgreen','lightskyblue']
        labs_true = ['Cosmic','iRPC','eRPC','IPA DIO',"iRMC","eRMC",'Trgt DIO', ' CE']
        datasets_filled = []
        colors_filled = []
        labels_filled = []
        for i, dat in enumerate(datasets):

          if len(dat) !=0:
            datasets_filled.append(dat)
            colors_filled.append(colors[i])
            labels_filled.append(labs_true[i])
          if logger:
            logger.log(f'len(colors_filled)={len(colors_filled)}', 'max')
          else:
            print(len(colors_filled))
        dummy_handle1 = ax1.plot([], marker="",color='white', label="Reco. MC")

        n,bins,patch = ax1.hist(datasets_filled,range=(fit_range[0],fit_range[1]), color=colors_filled, label=labels_filled, bins=50, histtype="bar", stacked=True)

        
        """
        for iproc, proc in enumerate(mom_components.keys()):
            print(iproc, proc.ljust(10)+':', len(sets[iproc+1]))
        """
        if logger:
          logger.log('======= True Events in Fit Range =======', 'info')
          logger.log(f'N Cosmics {len(data_cosmics)}', 'info')
          logger.log(f'N iRPC {len(data_irpc)}', 'info')
          logger.log(f'N eRPC {len(data_erpc)}', 'info')
          logger.log(f'N IPA {len(data_ipa)}', 'info')
          logger.log(f'N DIO {len(data_dio)}', 'info')
          logger.log(f'N CELL {len(data_signal)}', 'info')
        else:
          print("======= True Events in Fit Range =======")
          print("N Cosmics", len(data_cosmics))
          print("N iRPC", len(data_irpc))
          print("N eRPC", len(data_erpc))
          print("N IPA", len(data_ipa))
          print("N DIO", len(data_dio))
          print("N CELL", len(data_signal))
    else:
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    dummy_handle3 = ax1.plot([], marker="+",color='black', label="Mock Data")
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)
    
    # make plot
    combine_plot = np.zeros(len(time_plot))
    labs_fit = []
    dummy_handle2 = ax1.plot([], marker="",color='white', label="Fit Components")
    for name, pdfs, N_pdfs in list_pdfs:
        pdf_plot = (pdfs.pdf(time_plot) * N_pdfs * scale).numpy()
        combine_plot += pdf_plot
        labs_fit.append(name)
        style = time_components.get(name, {})
        color = style.get('lineColor', 'k')
        linestyle = style.get('lineStyle', '-')
        ax1.plot(time_plot, pdf_plot, label=name, color=color, linestyle=linestyle)

    ax1.plot(time_plot, combine_plot, '-r', label='Total')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1e-1, max(data_hist)])
    ax1.set_xlabel('Reconstructed Time [ns]', fontsize=16)
    ax1.set_ylabel('# of events per bin', fontsize=16)
    leg = ax1.legend(fontsize='large')
    legend_texts = leg.get_texts()
    if len(legend_texts) > 0:
      legend_texts[0].set_weight('bold')
    if len(legend_texts) > 7:
      legend_texts[7].set_weight('bold')
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
    ax2.set_xlabel('Reconstructed Time [ns]', fontsize=16)
    ax2.set_ylabel('Normalized Residual', fontsize=16)
    
def plot_time_fit_old(data, track_cats, fit_range, list_pdfs, cat=None):
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
        if logger:
          logger.log('Printing MC Truth', 'info')
        else:
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
    ax1.set_xlabel('Reconstructed Time [ns]', fontsize=16)
    ax1.set_ylabel('# of events per bin', fontsize=16)
    ax1.legend(fontsize='large')
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
    ax2.set_xlabel('Reconstructed time [ns]', fontsize=16)
    ax2.set_ylabel('Normalized Residual', fontsize=16)

