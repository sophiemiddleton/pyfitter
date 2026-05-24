# module holds functions associated with plotted reco features
import numpy as np
import awkward as ak
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from physics_components import mom_components, time_components
from pyutils.pylogger import Logger
from data_prep import DataPreparationManager
from config import GLOBAL_VERBOSITY

# Module logger
logger = Logger(print_prefix='[plot_module] ', verbosity=GLOBAL_VERBOSITY)


def get_leg_handles_labels(ax):
  try:
    handles, labels = ax.get_legend_handles_labels()
    # filter out empty or matplotlib-internal labels
    new_handles = []
    new_labels = []
    for h, l in zip(handles, labels):
      if l and (not str(l).startswith('_')):
        new_handles.append(h)
        new_labels.append(l)
    if new_labels:
      return new_handles, new_labels
    # fallback: inspect artists for labelled items
    new_handles = []
    new_labels = []
    for a in ax.get_children():
      if hasattr(a, 'get_label'):
        lab = a.get_label()
        if lab and (not str(lab).startswith('_')):
          new_handles.append(a)
          new_labels.append(lab)
    return new_handles, new_labels
  except Exception:
    return [], []



def plot_variable(val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], density=False):
  """
  Plots distributions of the given parameter (val), splitting by process code.

  try:
      handles, labels = get_leg_handles_labels(ax1)
    ncol = max(1, min(4, (len(labels) + 1) // 2))
    if labels:
      fig.subplots_adjust(right=0.68)
      fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.73, 0.5), ncol=1, fontsize=10, frameon=False)
      leg = None
    else:
      leg = ax1.legend(fontsize='large')
  except Exception:
    leg = ax1.legend(fontsize='large')

  Returns:
      saves a PDF named `<filenames>_selection.pdf`
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
    # create dummy handle for legend columns
    dummy_handle = plt.plot([], marker="",color='white', label=columns[i])
    n,bins,patch = plt.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=25, histtype=styles[i], linestyle=lines[i],alpha=alphas[i], stacked=True, density=density)
  plt.xlabel(str(val_label))
  # draw cuts
  plt.plot(cut_lo, [0,1000], 'k--')
  plt.plot(cut_hi, [0,1000], 'k--')

  # place legend in reserved figure space above the axes to avoid overlaying data
  try:
    fig = plt.gcf()
    ncol = max(1, min(6, len(columns)))
    fig.subplots_adjust(top=0.80)
    handles, labels = get_leg_handles_labels(plt.gca())
    if labels:
      # reserve space on right for legend and place legend inside reserved margin
      fig.subplots_adjust(right=0.66)
      leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.68, 0.5), ncol=1, fontsize=10, frameon=True)
      if leg is not None:
        leg.set_frame_on(True)
        for txt in leg.get_texts():
          txt.set_color('black')
          txt.set_alpha(1.0)
    else:
      leg = plt.legend(ncol=1, loc='center left', bbox_to_anchor=(0.68, 0.5))
  except Exception:
    plt.legend(ncol=len(columns), loc='upper center')

  plt.savefig(str(filenames)+"_selection.pdf", bbox_inches='tight')
  plt.close()

def plotmom_fit(mom_mag,mc_count, fit_range, list_pdfs, plot_truth=None):
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
    plot_truth : bool
        show the MC truth processes on the histogram

    """
    data = DataPreparationManager.get_numpy_array(mom_mag, remove_nans=True)
    n_bins = 25
    mom_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data = data[~np.isnan(data)] 
    
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    
    # run categorization and plot
    if plot_truth is None:
      logger.log('plot_truth option is None; will not include MC truth', 'info')
    
    if plot_truth is not None:
      mom_mag = ak.drop_none(mom_mag)
      # use fit_range bounds for selection instead of hard-coded values
      lo, hi = float(fit_range[0]), float(fit_range[1])

      data_signal = ak.mask(mom_mag, mc_count == 168)
      data_signal = np.array(ak.flatten(data_signal, axis=None))
      data_signal = [x for x in data_signal if (not np.isnan(x)) and (lo <= x <= hi)]

      data_cosmics = ak.mask(mom_mag, mc_count == -1)
      data_cosmics = np.array(ak.flatten(data_cosmics, axis=None))
      data_cosmics = [x for x in data_cosmics if (not np.isnan(x)) and (lo <= x <= hi)]
        
      data_dio = ak.mask(mom_mag, mc_count == 166)
      data_dio = np.array(ak.flatten(data_dio, axis=None))
      data_dio = [x for x in data_dio if (not np.isnan(x)) and (lo <= x <= hi)]

      data_erpc = ak.mask(mom_mag, mc_count == 178)
      data_erpc = np.array(ak.flatten(data_erpc,axis=None))
      data_erpc = [x for x in data_erpc if (not np.isnan(x)) and (lo <= x <= hi)]

      data_irpc = ak.mask(mom_mag, mc_count == 179)
      data_irpc = np.array(ak.flatten(data_irpc,axis=None))
      data_irpc = [x for x in data_irpc if (not np.isnan(x)) and (lo <= x <= hi)]

      data_ipa = ak.mask(mom_mag, mc_count == 0)
      data_ipa = np.array(ak.flatten(data_ipa,axis=None))
      data_ipa = [x for x in data_ipa if (not np.isnan(x)) and (lo <= x <= hi)]

      datasets = [data_cosmics,data_irpc,data_erpc,data_ipa, data_dio, data_signal]
      colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#e377c2', '#ffff00']
      labs_true = ['Cosmic','int. RPC','ext. RPC','IPA Decays','DIO', 'Signal']
  
      datasets_filled = []
      colors_filled = []
      labels_filled = []
      for i, dat in enumerate(datasets):
        if len(dat) != 0:
          datasets_filled.append(dat)
          colors_filled.append(colors[i])
          labels_filled.append(labs_true[i])
      dummy_handle1 = ax1.plot([], marker="", color='white', label="Reco. MC")
      n,bins,patch = ax1.hist(datasets_filled, range=(fit_range[0],fit_range[1]), color=colors_filled, label=labels_filled, bins=25, histtype="bar", stacked=True, edgecolor='black', linewidth=0.8,)

        
      """
      for iproc, proc in enumerate(mom_components.keys()):
          logger.log(f'{iproc} {proc.ljust(10)}: {len(sets[iproc+1])}', 'debug')
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
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    dummy_handle3 = ax1.plot([], marker="+",color='black', label="Mock Data")
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)
    
    # make plot
    combine_plot = np.zeros(len(mom_plot))
    labs_fit = []
    #dummy_handle2 = ax1.plot([], marker="",color='white', label="Fit Components")
    for name, pdfs, N_pdfs in list_pdfs:
      pdf_plot = (pdfs.pdf(mom_plot) * N_pdfs * scale).numpy()
      if logger:
        logger.log(f'{name}: pdf min={np.min(pdf_plot)}, max={np.max(pdf_plot)}, has_nan={np.any(np.isnan(pdf_plot))}, N_pdfs={N_pdfs}', 'info')
      
      # Only add to combine_plot if component has valid (non-NaN) values
      if not np.all(np.isnan(pdf_plot)):
        # Replace any NaN values with 0 before adding
        pdf_plot_clean = np.nan_to_num(pdf_plot, nan=0.0)
        combine_plot += pdf_plot_clean
      else:
        # Component is all NaN, skip it in combine_plot
        pdf_plot_clean = np.zeros_like(pdf_plot)
      
      labs_fit.append(name)
      style = mom_components.get(name, mom_components.get(name.strip(), {}))
      color = style.get('lineColor', 'k')
      ls = style.get('lineStyle', '-')
      ax1.plot(mom_plot, pdf_plot_clean, label=name, color=color, linestyle=ls)

    if logger:
      logger.log(f'combine_plot min: {np.min(combine_plot)}, max: {np.max(combine_plot)}', 'info')
    ax1.plot(mom_plot, combine_plot, '-r', label='Total', linewidth=2, zorder=100)
    if logger:
      logger.log('Red Total line plotted', 'info')
    #ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1.0,0.2*max(data_hist) + max(data_hist)])
    #ax1.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=12)
    ax1.set_ylabel('# of events per bin', fontsize=10)
    # build explicit proxy legend entries so text appears in reserved white space
    try:
      from matplotlib.lines import Line2D
      from matplotlib.patches import Patch
      handles0, labels0 = get_leg_handles_labels(ax1)
     
      proxy_handles = []
      proxy_labels = []
      # MC truth stacked components (if any)
      if 'labels_filled' in locals() and len(labels_filled) > 0:
        for col, lab in zip(colors_filled, labels_filled):
          proxy_handles.append(Patch(facecolor=col, edgecolor='black'))
          proxy_labels.append(lab)
      # Fit components (per-process lines) — ensure these appear in the legend
      if 'labs_fit' in locals() and len(labs_fit) > 0:
        for nm in labs_fit:
          style = mom_components.get(nm, mom_components.get(nm.strip(), {}))
          color = style.get('lineColor', 'k')
          ls = style.get('lineStyle', '-')
          proxy_handles.append(Line2D([0], [0], color=color, linestyle=ls))
          proxy_labels.append(nm)
      # Always add Mock Data and Total lines (outside inner try-except for robustness)
      proxy_handles.append(Line2D([0], [0], marker='+', color='black', linestyle=''))
      proxy_labels.append('Mock Data')
      proxy_handles.append(Line2D([0], [0], color='red', linestyle='-'))
      proxy_labels.append('Total')
      ncol = 1
      fig.subplots_adjust(right=0.66)
      
      leg = fig.legend(proxy_handles, proxy_labels, loc='center left', bbox_to_anchor=(0.68, 0.5), ncol=ncol, fontsize=10, frameon=True)
      if leg is not None:
        leg.set_frame_on(True)
        for txt in leg.get_texts():
          txt.set_color('black')
          txt.set_alpha(1.0)
    except Exception as e:
      if logger:
        logger.log(f'Legend creation failed: {e}; falling back to ax1.legend', 'warning')
      try:
        leg = ax1.legend(fontsize=10)
      except Exception:
        leg = None
    # Make only the 'Reco. MC' legend entries bold; others normal
    legend_texts = leg.get_texts() if leg is not None else []
    for txt in legend_texts:
      t = txt.get_text()
      if t in ('Reco. MC'):
        txt.set_weight('bold')
      else:
        txt.set_weight('normal')
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
       logger.log('histogram empty', 'warning')

    ax2.errorbar(mom_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    # add horizontal line at zero for residuals
    try:
      ax2.axhline(0, color='red', linewidth=1)
    except Exception:
      pass
    #ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=10)
    ax2.set_ylabel('Normalized Residual', fontsize=10)
    # yield comparison plot (expected vs fitted) for momentum
    try:
      plot_yield_comparison(mom_mag, mc_count, list_pdfs, filename_prefix='yield_compare_mom')
    except Exception:
      logger.log('yield comparison (mom) failed', 'warning')
    # note: plot_yield_comparison belongs with momentum and time plotting callers
    

def plottime_fit(time,mc_count, fit_range, list_pdfs, plot_truth=None):
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
    plot_truth : bool
        show the MC truth processes on the histogram

    """
    data = DataPreparationManager.get_numpy_array(time, remove_nans=True)
    n_bins = 25
    time_plot = np.linspace(fit_range[0], fit_range[1], n_bins)
    scale = 1 / n_bins * (fit_range[1] - fit_range[0])
    data = data[~np.isnan(data)] 
    
    data_hist, data_binedge = np.histogram(data, bins=n_bins, range=fit_range)
    data_bincenter = 0.5 * (data_binedge[1:] + data_binedge[:-1])

    fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
    
    # run categorization and plot
    logger.log(f'Plot_truth {plot_truth}', 'debug')
    if plot_truth is None:
      logger.log('plot_truth option is None; will not include MC truth', 'info')
    
    if plot_truth is not None:
        # Align mc_count and time explicitly. Prefer keeping awkward arrays with matching per-event counts.
        try:
          # compute per-event counts for time and mc_count (if mc_count is awkward)
          counts_time = ak.num(ak.drop_none(time), axis=1)
          counts_mc = ak.num(mc_count, axis=1)
          counts_time_np = ak.to_numpy(counts_time)
          counts_mc_np = ak.to_numpy(counts_mc)
          if counts_time_np.shape == counts_mc_np.shape and np.array_equal(counts_time_np, counts_mc_np):
            # shapes align; keep `time` and `mc_count` as awkward arrays
            pass
          else:
            # fallback to flattened comparison and explicit broadcast-or-error
            time_flat = ak.to_numpy(ak.flatten(ak.drop_none(time), axis=None))
            try:
              mc_arr = np.asarray(mc_count).flatten()
            except Exception:
              mc_arr = np.array([])

            if len(mc_arr) == len(time_flat):
              mc_count = mc_arr
              time = time_flat
            else:
              if mc_arr.size == 1 and time_flat.size > 0:
                logger.log(f'mc_count length 1; broadcasting value {mc_arr[0]} to length {len(time_flat)}', 'info')
                mc_count = np.full(len(time_flat), mc_arr[0], dtype=mc_arr.dtype)
                time = time_flat
              else:
                msg = (
                  f'plot_module.plottime_fit: mc_count/time length mismatch ({len(mc_arr)} vs {len(time_flat)}). '
                  'Do not silently truncate — provide track-aligned `mc_count` (one entry per flattened time) or adjust selection.'
                )
                logger.log(msg, 'error')
                raise ValueError(msg)
        except Exception:
          # If mc_count is not awkward or counts computation failed, fallback to flattened logic
          time_flat = ak.to_numpy(ak.flatten(ak.drop_none(time), axis=None))
          try:
            mc_arr = np.asarray(mc_count).flatten()
          except Exception:
            mc_arr = np.array([])

          if len(mc_arr) == len(time_flat):
            mc_count = mc_arr
            time = time_flat
          else:
            if mc_arr.size == 1 and time_flat.size > 0:
              logger.log(f'mc_count length 1; broadcasting value {mc_arr[0]} to length {len(time_flat)}', 'info')
              mc_count = np.full(len(time_flat), mc_arr[0], dtype=mc_arr.dtype)
              time = time_flat
            else:
              msg = (
                f'plot_module.plottime_fit: mc_count/time length mismatch ({len(mc_arr)} vs {len(time_flat)}). '
                'Do not silently truncate — provide track-aligned `mc_count` (one entry per flattened time) or adjust selection.'
              )
              logger.log(msg, 'error')
              raise ValueError(msg)
    if cat is not None:
        time = ak.drop_none(time)
        logger.log('filling list', 'debug')
        # use fit_range bounds for time selection
        lo, hi = float(fit_range[0]), float(fit_range[1])
        data_signal = ak.mask(time, mc_count == 168)
        data_signal = np.array(ak.flatten(data_signal, axis=None))
        data_signal = [x for x in data_signal if (not np.isnan(x)) and (lo <= x <= hi)]

        data_cosmics = ak.mask(time, mc_count == -1)
        data_cosmics = np.array(ak.flatten(data_cosmics, axis=None))
        data_cosmics = [x for x in data_cosmics if (not np.isnan(x)) and (lo <= x <= hi)]
        
        data_dio = ak.mask(time, mc_count == 166)
        data_dio = np.array(ak.flatten(data_dio, axis=None))
        data_dio = [x for x in data_dio if (not np.isnan(x)) and (lo <= x <= hi)]
        
        data_erpc = ak.mask(time, mc_count == 178)
        data_erpc = np.array(ak.flatten(data_erpc,axis=None))
        data_erpc = [x for x in data_erpc if (not np.isnan(x)) and (lo <= x <= hi)]
        
        data_irpc = ak.mask(time, mc_count == 179)
        data_irpc = np.array(ak.flatten(data_irpc,axis=None))
        data_irpc = [x for x in data_irpc if (not np.isnan(x)) and (lo <= x <= hi)]
        
        data_ermc = ak.mask(time, mc_count == 172)
        data_ermc = np.array(ak.flatten(data_ermc,axis=None))
        data_ermc = [x for x in data_ermc if (not np.isnan(x)) and (lo <= x <= hi)]
        
        data_irmc = ak.mask(time, mc_count == 171)
        data_irmc = np.array(ak.flatten(data_irmc,axis=None))
        data_irmc = [x for x in data_irmc if (not np.isnan(x)) and (lo <= x <= hi)]
        
        data_ipa = ak.mask(time, mc_count == 0)
        data_ipa = np.array(ak.flatten(data_ipa,axis=None))
        data_ipa = [x for x in data_ipa if (not np.isnan(x)) and (lo <= x <= hi)]
        
        datasets = [data_cosmics,data_irpc,data_erpc,data_ipa, data_dio, data_signal]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#e377c2', '#ffff00']
        labs_true = ['Cosmic','int. RPC','ext. RPC','IPA Decays','DIO', 'Signal']
        datasets_filled = []
        colors_filled = []
        labels_filled = []
        for i, dat in enumerate(datasets):

          if len(dat) !=0:
            datasets_filled.append(dat)
            colors_filled.append(colors[i])
            labels_filled.append(labs_true[i])
          logger.log(f'len(colors_filled)={len(colors_filled)}', 'debug')
        dummy_handle1 = ax1.plot([], marker="",color='white', label="Reco. MC")

        n,bins,patch = ax1.hist(datasets_filled,range=(fit_range[0],fit_range[1]), color=colors_filled, label=labels_filled, bins=25, edgecolor='black', linewidth=0.8,histtype="bar", stacked=True)

        
        """
        for iproc, proc in enumerate(mom_components.keys()):
            logger.log(f'{iproc} {proc.ljust(10)}: {len(sets[iproc+1])}', 'debug')
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
        ax1.hist(data, color='black', bins=n_bins, range=fit_range, histtype='step')
    dummy_handle3 = ax1.plot([], marker="+",color='black', label="Mock Data")
    ax1.errorbar(data_bincenter, data_hist, yerr=np.sqrt(data_hist), color='None', ecolor='black', capsize=3)
    
    # make plot
    combine_plot = np.zeros(len(time_plot))
    labs_fit = []
    #dummy_handle2 = ax1.plot([], marker="",color='white', label="Fit Components")

    # combine DIO and CE into a single 'Stopped muon' curve for the fit
    stopped_plot = np.zeros(len(time_plot))
    stopped_names = ('DIO', 'CE')
    for name, pdfs, N_pdfs in list_pdfs:
      pdf_plot = (pdfs.pdf(time_plot) * N_pdfs * scale).numpy()
      
      # Only add to combine_plot if component has valid (non-NaN) values
      if not np.all(np.isnan(pdf_plot)):
        # Replace any NaN values with 0 before adding
        pdf_plot_clean = np.nan_to_num(pdf_plot, nan=0.0)
        combine_plot += pdf_plot_clean
        if name in stopped_names:
          stopped_plot += pdf_plot_clean
      else:
        # Component is all NaN, skip it in combine_plot
        pdf_plot_clean = np.zeros_like(pdf_plot)
        
      if name in stopped_names:
        continue
      # plot other components individually
      labs_fit.append(name)
      style = time_components.get(name, {})
      color = style.get('lineColor', 'k')
      linestyle = style.get('lineStyle', '-')
      ax1.plot(time_plot, pdf_plot_clean, label=name, color=color, linestyle=linestyle)

    # plot combined stopped muon curve (DIO + CE) if present
    if np.any(stopped_plot):
      # choose a representative style (use DIO if available, else CE)
      stopped_style = time_components.get('DIO', time_components.get('CE', {}))
      stopped_color = stopped_style.get('lineColor', 'cyan')
      stopped_ls = stopped_style.get('lineStyle', '--')
      ax1.plot(time_plot, stopped_plot, label='Stopped muon', color=stopped_color, linestyle=stopped_ls)

    ax1.plot(time_plot, combine_plot, '-r', label='Total')
    #ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1.0, max(data_hist) + 0.2*max(data_hist)])
    #ax1.set_xlabel('Reconstructed Time [ns]', fontsize=12)
    ax1.set_ylabel('# of events per bin', fontsize=10)
    # build explicit proxy legend entries for time plot so text appears
    try:
      from matplotlib.lines import Line2D
      from matplotlib.patches import Patch
      proxy_handles = []
      proxy_labels = []
      if 'labels_filled' in locals() and len(labels_filled) > 0:
        for col, lab in zip(colors_filled, labels_filled):
          proxy_handles.append(Patch(facecolor=col, edgecolor='black'))
          proxy_labels.append(lab)
      # add per-process fit component proxies
      if 'labs_fit' in locals() and len(labs_fit) > 0:
        for nm in labs_fit:
          style = time_components.get(nm, {})
          color = style.get('lineColor', 'k')
          ls = style.get('lineStyle', '-')
          proxy_handles.append(Line2D([0], [0], color=color, linestyle=ls))
          proxy_labels.append(nm)
      # add stopped-muon proxy (DIO+CE) if present
      try:
        if 'stopped_plot' in locals() and np.any(stopped_plot):
          stopped_style = time_components.get('DIO', time_components.get('CE', {}))
          stopped_color = stopped_style.get('lineColor', 'cyan')
          stopped_ls = stopped_style.get('lineStyle', '--')
          proxy_handles.append(Line2D([0], [0], color=stopped_color, linestyle=stopped_ls))
          proxy_labels.append('Stopped muon')
      except Exception:
        pass
      proxy_handles.append(Line2D([0], [0], marker='+', color='black', linestyle=''))
      proxy_labels.append('Mock Data')

      proxy_handles.append(Line2D([0], [0], color='red'))
      proxy_labels.append('Total')
      ncol = 1
      fig.subplots_adjust(right=0.66)
      leg = fig.legend(proxy_handles, proxy_labels, loc='center left', bbox_to_anchor=(0.68, 0.5), ncol=ncol, fontsize=10, frameon=True)
      if leg is not None:
        leg.set_frame_on(True)
        for txt in leg.get_texts():
          txt.set_color('black')
          txt.set_alpha(1.0)
    except Exception:
      try:
        leg = ax1.legend(fontsize=10)
      except Exception:
        leg = None
    # Make only the 'Reco. MC' and 'Fit Components' legend entries bold; others normal
    legend_texts = leg.get_texts() if leg is not None else []
    for txt in legend_texts:
      t = txt.get_text()
      if t in ('Reco. MC'):
        txt.set_weight('bold')
      else:
        txt.set_weight('normal')
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
       logger.log('histogram empty', 'warning')

    ax2.errorbar(time_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    # add horizontal line at zero for residuals
    try:
      ax2.axhline(0, color='red', linewidth=1)
    except Exception:
      pass
    #ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Time [ns]', fontsize=10)
    ax2.set_ylabel('Normalized Residual', fontsize=10)
    # yield comparison plot (expected vs fitted) for momentum
    try:
      plot_yield_comparison(mom_mag, mc_count, list_pdfs, filename_prefix='yield_compare_mom')
    except Exception:
      logger.log('yield comparison (mom) failed', 'warning')
    # note: plot_yield_comparison belongs with momentum and time plotting callers


def plot_yield_comparison(data, mc_count, list_pdfs, filename_prefix='yield_compare'):
    """Plot fitted yields vs expected (true) counts from `mc_count`.

    - `data` is the array to align masks with (time or momentum).
    - `list_pdfs` is list of (proc, pdf, N) as used elsewhere.
    The function saves a PNG named `{filename_prefix}_{timestamp}.png`.
    """
    import time as _time
    # mapping fit process names to mc_count codes (collection where needed)
    proc_code_map = {
        'Cosmic': [-1],
        'RPC': [179, 178],
        'iRPC': [179],
        'eRPC': [178],
        'IPA': [0],
        'DIO': [166],
        'CE': [168]
    }

    def _extract_value(N):
        try:
            return float(N.numpy())
        except Exception:
            try:
                return float(N.value())
            except Exception:
                try:
                    return float(N)
                except Exception:
                    return 0.0

    def _count_for_codes(data_arr, mc_count_arr, codes):
        total = 0
        for c in codes:
            try:
                masked = ak.mask(data_arr, mc_count_arr == c)
                masked = ak.to_numpy(ak.flatten(masked, axis=None))
                if len(masked) > 0:
                    total += int(np.count_nonzero(~np.isnan(masked)))
            except Exception:
                pass
        return total

    # Build arrays for plotting
    procs = []
    fitted = []
    expected = []
    for proc, pdf_obj, N_par in list_pdfs:
        procs.append(proc)
        fitted.append(_extract_value(N_par))
        codes = proc_code_map.get(proc, None)
        if codes is None:
            # Try direct name-match fallback (single code)
            try:
                code = int(proc)
                codes = [code]
            except Exception:
                codes = []
        expected.append(_count_for_codes(data, mc_count, codes))

    # Always print yield comparison
    print("================== Fitted vs True Yields =======================")
    for p, f, e in zip(procs, fitted, expected):
        print(f"  {p:12s}:  Fitted={f:8.1f}  True={e:8d}")
    print("="*60)

    # plot side-by-side bars
    x = np.arange(len(procs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(procs)*1.2), 4))
    rects1 = ax.bar(x - width/2, expected, width, label='Expected')
    rects2 = ax.bar(x + width/2, fitted, width, label='Fitted')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')

    ax.set_title('Expected vs Fitted Yields')
    ax.set_xticks(x)
    ax.set_xticklabels(procs, rotation=45, ha='right')
    try:
      handles, labels = get_leg_handles_labels(ax)
      ncol = max(1, min(4, (len(labels) + 1) // 2))
      if labels:
        fig.subplots_adjust(right=0.66)
        leg = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.68, 0.5), ncol=1, fontsize=10, frameon=True)
        if leg is not None:
          leg.set_frame_on(True)
          for txt in leg.get_texts():
            txt.set_color('black')
            txt.set_alpha(1.0)
      else:
        leg = ax.legend(fontsize=10)
    except Exception:
      leg = ax.legend(fontsize=10)

    # add error bars: expected ~ Poisson sqrt(N); fitted use sqrt(N) as an approximate uncertainty
    expected_errs = [np.sqrt(e) if e >= 0 else 0.0 for e in expected]
    fitted_errs = []
    for fv in fitted:
      try:
        fv_f = float(fv)
        fitted_errs.append(np.sqrt(abs(fv_f)))
      except Exception:
        fitted_errs.append(0.0)

    ax.errorbar(x - width/2, expected, yerr=expected_errs, fmt='none', ecolor='black', capsize=4)
    ax.errorbar(x + width/2, fitted, yerr=fitted_errs, fmt='none', ecolor='black', capsize=4)

    ts = int(_time.time())
    fname = f"{filename_prefix}_{ts}.png"
    try:
        plt.tight_layout()
        plt.savefig(fname)
        logger.log(f"Saved yield comparison to {fname}", 'info')
    except Exception as e:
        logger.log(f"Failed to save yield comparison: {e}", 'error')
    plt.close(fig)


def bin_by_bin_mom_confusion(mom_mag, mc_count, list_pdfs, fit_range, bin_width=0.5, filename_prefix='mom_confusion'):
    """Compare true vs fitted relative yields per momentum bin.

    - `mom_mag`: awkward array of reconstructed momenta
    - `mc_count`: array of process codes aligned with `mom_mag`
    - `list_pdfs`: list of (proc, mom_pdf, N) as used elsewhere
    - `fit_range`: [low, high]
    - `bin_width`: width of momentum bins in MeV
    Saves `{filename_prefix}_{timestamp}.png` and returns a dict with arrays.
    """
    import time as _time
    lo, hi = float(fit_range[0]), float(fit_range[1])
    edges = np.arange(lo, hi + 1e-6, bin_width)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)

    proc_code_map = {
        'Cosmic': [-1],
        'RPC': [179, 178],
        'iRPC': [179],
        'eRPC': [178],
        'IPA': [0],
        'DIO': [166],
        'CE': [168]
    }

    mom_flat = ak.to_numpy(ak.flatten(ak.drop_none(mom_mag), axis=None))
    # coerce mc_count to a flat numpy array, prefer awkward flatten if available
    try:
      mc_arr = ak.to_numpy(ak.flatten(ak.drop_none(mc_count), axis=None))
    except Exception:
      try:
        mc_arr = np.asarray(mc_count).flatten()
      except Exception:
        mc_arr = np.array([])

    # align lengths by trimming the longer array to preserve per-event pairing
    if len(mc_arr) != len(mom_flat):
      nmin = min(len(mc_arr), len(mom_flat))
      logger.log(f'mc_count/mom_mag length mismatch in bin_by_bin_mom_confusion ({len(mc_arr)} vs {len(mom_flat)}); trimming to {nmin}', 'info')
      mc_arr = mc_arr[:nmin]
      mom_flat = mom_flat[:nmin]

    procs = [p for p, _, _ in list_pdfs]
    true_counts = {p: np.zeros(len(centers), dtype=float) for p in procs}
    for i in range(len(centers)):
        lo_edge, hi_edge = edges[i], edges[i+1]
        in_bin = (mom_flat >= lo_edge) & (mom_flat < hi_edge)
        for p in procs:
            codes = proc_code_map.get(p, [])
            if len(mc_arr) == 0:
                cnt = 0
            else:
                mask = np.zeros_like(in_bin, dtype=bool)
                for c in codes:
                    mask = mask | (mc_arr == c)
                cnt = int(np.sum(in_bin & mask))
            true_counts[p][i] = cnt

    fitted_counts = {p: np.zeros(len(centers), dtype=float) for p in procs}
    for p, pdf_obj, N_par in list_pdfs:
        try:
            N_val = float(N_par.numpy()) if hasattr(N_par, 'numpy') else float(N_par)
        except Exception:
            try:
                N_val = float(N_par.value())
            except Exception:
                N_val = 0.0
        try:
            dens = pdf_obj.pdf(centers).numpy()
        except Exception:
            try:
                dens = np.array(pdf_obj.pdf(centers))
            except Exception:
                dens = np.zeros(len(centers))
        fitted_counts[p] = dens * N_val * bw

    # build matrices for vectorized fraction computation (clean NaNs/Infs)
    true_mat = np.vstack([np.nan_to_num(true_counts[p], nan=0.0, posinf=0.0, neginf=0.0).astype(float) for p in procs])
    fit_mat = np.vstack([np.nan_to_num(fitted_counts[p], nan=0.0, posinf=0.0, neginf=0.0).astype(float) for p in procs])
    # column-wise totals
    ttot_vec = np.sum(true_mat, axis=0)
    ftot_vec = np.sum(fit_mat, axis=0)
    # compute fractions with safe division; when total==0 keep zeros
    true_frac = np.zeros_like(true_mat, dtype=float)
    fit_frac = np.zeros_like(fit_mat, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
      np.divide(true_mat, ttot_vec, out=true_frac, where=ttot_vec > 0)
      np.divide(fit_mat, ftot_vec, out=fit_frac, where=ftot_vec > 0)
    # sanitize numerical issues: replace NaN/inf with 0, clip to [0,1], zero tiny values
    true_frac = np.nan_to_num(true_frac, nan=0.0, posinf=0.0, neginf=0.0)
    fit_frac = np.nan_to_num(fit_frac, nan=0.0, posinf=0.0, neginf=0.0)
    true_frac[~np.isfinite(true_frac)] = 0.0
    fit_frac[~np.isfinite(fit_frac)] = 0.0
    true_frac[np.abs(true_frac) < 1e-12] = 0.0
    fit_frac[np.abs(fit_frac) < 1e-12] = 0.0
    true_frac = np.clip(true_frac, 0.0, 1.0)
    fit_frac = np.clip(fit_frac, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    # compute totals for clearer labeling
    # use nan-safe sums to avoid NaN -> int conversion errors
    totals_true = {p: float(np.nansum(true_counts[p])) for p in procs}
    totals_fit = {p: float(np.nansum(fitted_counts[p])) for p in procs}
    max_val = 0.0
    for ip, p in enumerate(procs):
      ttrue = int(np.nan_to_num(totals_true[p], nan=0.0))
      tfit = int(np.nan_to_num(np.round(totals_fit[p]), nan=0.0))
      ytrue = true_frac[ip]
      yfit = fit_frac[ip]
      if len(ytrue) > 0:
        max_val = max(max_val, np.nanmax(ytrue))
      if len(yfit) > 0:
        max_val = max(max_val, np.nanmax(yfit))
      # plot true (filled circle) and fit (hollow x) with larger markers and z-order to ensure visibility
      ax.plot(centers, ytrue, marker='o', linestyle='-', label=f"{p} true (N={ttrue})", markersize=6, markeredgewidth=1.2, zorder=2)
      ax.plot(centers, yfit, marker='x', linestyle='--', label=f"{p} fit (N={tfit})", markersize=7, markeredgewidth=1.5, markerfacecolor='none', zorder=3)
    # ensure some headroom for markers
    ymax = max(0.05, max_val * 1.15)
    ax.set_xlabel('Reconstructed Momentum [MeV/c] (bin center)')
    ax.set_ylabel('Relative fraction in bin')
    ax.set_title('Bin-by-bin true vs fitted relative yields (momentum)')
    ax.set_ylim(0.0, ymax)
    try:
      handles0, labels0 = get_leg_handles_labels(ax)

      ncol = 1
      if labels0:
        fig.subplots_adjust(right=0.66)

        leg = fig.legend(handles0, labels0, loc='center left', bbox_to_anchor=(0.68, 0.5), ncol=ncol, fontsize=10, frameon=False)
      else:
        leg = ax.legend(fontsize=10)
    except Exception:
      leg = ax.legend(fontsize=10)
    ax.grid(True, linestyle=':')
    ts = int(_time.time())
    fname = f"{filename_prefix}_{ts}.png"
    try:
      plt.tight_layout()
      plt.savefig(fname)
      logger.log(f'Saved momentum confusion plot to {fname}', 'info')
    except Exception as e:
      logger.log(f'Failed to save momentum confusion plot: {e}', 'error')
    plt.close(fig)
    # Print summary totals to terminal for quick inspection
    logger.log('Momentum bin-by-bin confusion summary:', 'info')
    for p in procs:
      ttrue = int(np.nan_to_num(totals_true[p], nan=0.0))
      tfit = int(np.nan_to_num(np.round(totals_fit[p]), nan=0.0))
      logger.log(f"  {p}: true N = {ttrue}, fitted N = {tfit}", 'info')

    # Print per-bin arrays for each process: true counts, fitted counts, and relative fractions
    try:
      logger.log('Bin centers (MeV):', 'debug')
      logger.log(np.array2string(centers, precision=2, separator=', '), 'debug')
      for p in procs:
        tcounts = np.nan_to_num(true_counts[p], nan=0.0)
        fcounts = np.nan_to_num(fitted_counts[p], nan=0.0)
        tfr = np.nan_to_num(true_frac[procs.index(p)], nan=0.0)
        ffr = np.nan_to_num(fit_frac[procs.index(p)], nan=0.0)
        logger.log(f"Process {p}:", 'debug')
        logger.log('  True counts per bin:   ' + np.array2string(tcounts, precision=3, separator=', '), 'debug')
        logger.log('  Fitted counts per bin: ' + np.array2string(fcounts, precision=3, separator=', '), 'debug')
        logger.log('  True frac per bin:     ' + np.array2string(tfr, precision=3, separator=', '), 'debug')
        logger.log('  Fit frac per bin:      ' + np.array2string(ffr, precision=3, separator=', '), 'debug')
    except Exception:
      pass

    return {'bins': centers, 'true_counts': true_counts, 'fitted_counts': fitted_counts, 'true_frac': true_frac, 'fit_frac': fit_frac, 'totals_true': totals_true, 'totals_fit': totals_fit}


def bin_by_bin_time_confusion(time_arr, mc_count, list_pdfs, fit_range, bin_width=25.0, filename_prefix='time_confusion'):
    """Compare true vs fitted relative yields per time bin.

    - `time_arr`: awkward array of reconstructed times
    - `mc_count`: array of process codes aligned with `time_arr`
    - `list_pdfs`: list of (proc, time_pdf, N) as used elsewhere
    - `fit_range`: [low, high]
    - `bin_width`: width of time bins in ns
    Saves `{filename_prefix}_{timestamp}.png` and returns a dict with arrays.
    """
    import time as _time
    lo, hi = float(fit_range[0]), float(fit_range[1])
    edges = np.arange(lo, hi + 1e-6, bin_width)
    if edges[-1] < hi:
        edges = np.append(edges, hi)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = np.diff(edges)

    proc_code_map = {
        'Cosmic': [-1],
        'RPC': [179, 178],
        'iRPC': [179],
        'eRPC': [178],
        'IPA': [0],
        'DIO': [166],
        'CE': [168]
    }

    time_flat = ak.to_numpy(ak.flatten(ak.drop_none(time_arr), axis=None))
    try:
        mc_arr = ak.to_numpy(ak.flatten(ak.drop_none(mc_count), axis=None))
    except Exception:
        try:
            mc_arr = np.asarray(mc_count).flatten()
        except Exception:
            mc_arr = np.array([])

    # align lengths by trimming the longer array to preserve per-event pairing
    if len(mc_arr) != len(time_flat):
        nmin = min(len(mc_arr), len(time_flat))
        logger.log(f'mc_count/time length mismatch in bin_by_bin_time_confusion ({len(mc_arr)} vs {len(time_flat)}); trimming to {nmin}', 'info')
        mc_arr = mc_arr[:nmin]
        time_flat = time_flat[:nmin]

    procs = [p for p, _, _ in list_pdfs]
    true_counts = {p: np.zeros(len(centers), dtype=float) for p in procs}
    for i in range(len(centers)):
        lo_edge, hi_edge = edges[i], edges[i+1]
        in_bin = (time_flat >= lo_edge) & (time_flat < hi_edge)
        for p in procs:
            codes = proc_code_map.get(p, [])
            if len(mc_arr) == 0:
                cnt = 0
            else:
                mask = np.zeros_like(in_bin, dtype=bool)
                for c in codes:
                    mask = mask | (mc_arr == c)
                cnt = int(np.sum(in_bin & mask))
            true_counts[p][i] = cnt

    fitted_counts = {p: np.zeros(len(centers), dtype=float) for p in procs}
    for p, pdf_obj, N_par in list_pdfs:
        try:
            N_val = float(N_par.numpy()) if hasattr(N_par, 'numpy') else float(N_par)
        except Exception:
            try:
                N_val = float(N_par.value())
            except Exception:
                N_val = 0.0
        try:
            dens = pdf_obj.pdf(centers).numpy()
        except Exception:
            try:
                dens = np.array(pdf_obj.pdf(centers))
            except Exception:
                dens = np.zeros(len(centers))
        fitted_counts[p] = dens * N_val * bw

    true_mat = np.vstack([np.nan_to_num(true_counts[p], nan=0.0, posinf=0.0, neginf=0.0).astype(float) for p in procs])
    fit_mat = np.vstack([np.nan_to_num(fitted_counts[p], nan=0.0, posinf=0.0, neginf=0.0).astype(float) for p in procs])
    ttot_vec = np.sum(true_mat, axis=0)
    ftot_vec = np.sum(fit_mat, axis=0)
    true_frac = np.zeros_like(true_mat, dtype=float)
    fit_frac = np.zeros_like(fit_mat, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
      np.divide(true_mat, ttot_vec, out=true_frac, where=ttot_vec > 0)
      np.divide(fit_mat, ftot_vec, out=fit_frac, where=ftot_vec > 0)
    true_frac = np.nan_to_num(true_frac, nan=0.0, posinf=0.0, neginf=0.0)
    fit_frac = np.nan_to_num(fit_frac, nan=0.0, posinf=0.0, neginf=0.0)
    true_frac[~np.isfinite(true_frac)] = 0.0
    fit_frac[~np.isfinite(fit_frac)] = 0.0
    true_frac[np.abs(true_frac) < 1e-12] = 0.0
    fit_frac[np.abs(fit_frac) < 1e-12] = 0.0
    true_frac = np.clip(true_frac, 0.0, 1.0)
    fit_frac = np.clip(fit_frac, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    totals_true = {p: float(np.nansum(true_counts[p])) for p in procs}
    totals_fit = {p: float(np.nansum(fitted_counts[p])) for p in procs}
    max_val = 0.0
    for ip, p in enumerate(procs):
      ttrue = int(np.nan_to_num(totals_true[p], nan=0.0))
      tfit = int(np.nan_to_num(np.round(totals_fit[p]), nan=0.0))
      ytrue = true_frac[ip]
      yfit = fit_frac[ip]
      if len(ytrue) > 0:
        max_val = max(max_val, np.nanmax(ytrue))
      if len(yfit) > 0:
        max_val = max(max_val, np.nanmax(yfit))
      ax.plot(centers, ytrue, marker='o', linestyle='-', label=f"{p} true (N={ttrue})", markersize=6, markeredgewidth=1.2, zorder=2)
      ax.plot(centers, yfit, marker='x', linestyle='--', label=f"{p} fit (N={tfit})", markersize=7, markeredgewidth=1.5, markerfacecolor='none', zorder=3)
    ymax = max(0.05, max_val * 1.15)
    ax.set_xlabel('Reconstructed Time [ns] (bin center)')
    ax.set_ylabel('Relative fraction in bin')
    ax.set_title('Bin-by-bin true vs fitted relative yields (time)')
    ax.set_ylim(0.0, ymax)
    try:
      handles0, labels0 = get_leg_handles_labels(ax)
      ncol = 1
      if labels0:
        fig.subplots_adjust(right=0.66)
        leg = fig.legend(handles0, labels0, loc='center left', bbox_to_anchor=(0.68, 0.5), ncol=ncol, fontsize=10, frameon=False)
      else:
        leg = ax.legend(fontsize=10)
    except Exception:
      leg = ax.legend(fontsize=10)
    ax.grid(True, linestyle=':')
    ts = int(_time.time())
    fname = f"{filename_prefix}_{ts}.png"
    try:
      plt.tight_layout()
      plt.savefig(fname)
      logger.log(f'Saved time confusion plot to {fname}', 'info')
    except Exception as e:
      logger.log(f'Failed to save time confusion plot: {e}', 'error')
    plt.close(fig)

    logger.log('Time bin-by-bin confusion summary:', 'info')
    for p in procs:
      ttrue = int(np.nan_to_num(totals_true[p], nan=0.0))
      tfit = int(np.nan_to_num(np.round(totals_fit[p]), nan=0.0))
      logger.log(f"  {p}: true N = {ttrue}, fitted N = {tfit}", 'info')

    try:
      logger.log('Bin centers (ns):', 'debug')
      logger.log(np.array2string(centers, precision=2, separator=', '), 'debug')
      for p in procs:
        tcounts = np.nan_to_num(true_counts[p], nan=0.0)
        fcounts = np.nan_to_num(fitted_counts[p], nan=0.0)
        tfr = np.nan_to_num(true_frac[procs.index(p)], nan=0.0)
        ffr = np.nan_to_num(fit_frac[procs.index(p)], nan=0.0)
        logger.log(f"Process {p}:", 'debug')
        logger.log('  True counts per bin:   ' + np.array2string(tcounts, precision=3, separator=', '), 'debug')
        logger.log('  Fitted counts per bin: ' + np.array2string(fcounts, precision=3, separator=', '), 'debug')
        logger.log('  True frac per bin:     ' + np.array2string(tfr, precision=3, separator=', '), 'debug')
        logger.log('  Fit frac per bin:      ' + np.array2string(ffr, precision=3, separator=', '), 'debug')
    except Exception:
      pass

    return {'bins': centers, 'true_counts': true_counts, 'fitted_counts': fitted_counts, 'true_frac': true_frac, 'fit_frac': fit_frac, 'totals_true': totals_true, 'totals_fit': totals_fit}

def plot_nll_scan(pars, loss, minimizer, mom_mag, count_particle_types, result, fit_range, verbose=0):
    """
    Perform and plot an NLL scan over signal yield range.

    Parameters
    ----------
    pars : list
        List of fit parameters (first one assumed to be signal yield)
    loss : zfit loss
        Loss function for minimization
    minimizer : zfit minimizer
        Minimizer object to use
    mom_mag : awkward array
        Momentum magnitude data
    count_particle_types : awkward array
        Particle type codes for filtering signal
    result : zfit result
        Fit result containing best-fit value
    fit_range : tuple
        (low, high) bounds for fit range
    verbose : int, optional
        Verbosity level (default: 0)
    """
    import time as _time

    if verbose > 0:
        logger.log('Starting NLL scan...', 'info')
    
    best_nll = result.fmin
    logger.log(f"Best fit nsig: {result.params[pars[0]]['value']:.2f}", 'info')
    logger.log(f"Minimum NLL: {best_nll:.2f}", 'info')

    scan_range = np.linspace(0, float(pars[0].value()) + float(pars[0].value())*0.5, 41)
    nll_values = []

    # Loop over the scan range for the signal yield
    for n in scan_range:
        with pars[0].set_value(n):
            pars[0].floating = False
            
            minimizer.minimize(loss)
            nll_values.append(loss.value())  
            pars[0].floating = True

    logger.log('Scan complete', 'info')

    # Find true number:
    data_signal = ak.mask(mom_mag, count_particle_types == 168)
    data_signal = np.array(ak.flatten(data_signal, axis=None))
    
    delta_nll = np.array(nll_values) - best_nll
    fig, ax = plt.subplots()
    ax.plot(scan_range, delta_nll)
    true_signal = len(data_signal)
    ax.axvline(true_signal, color='red', linestyle='--', label=f'True $N_{{sig}}$: {true_signal:.1f}')
    ax.legend()
    ax.text(true_signal + 5, 4, f'True $N_{{sig}} = {true_signal:.1f}$',
            verticalalignment='top', horizontalalignment='left', color='red')

    ax.set_xlabel('$N_{sig}$')
    ax.set_ylabel(r'$-2\Delta \ln(L)$')
    ax.set_title('NLL Scan for $N_{sig}$')
    ax.grid(True)
    ts = int(_time.time())
    fname_nll = f"fit_mom_nll_{ts}.png"
    try:
        plt.savefig(fname_nll)
        logger.log(f"Saved NLL scan to {fname_nll}", "info")
    except Exception as e:
        logger.log(f"Failed to save NLL scan: {e}", "error")
    plt.close()
