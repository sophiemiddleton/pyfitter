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
    # Make only the 'Reco. MC' and 'Fit Components' legend entries bold; others normal
    legend_texts = leg.get_texts()
    for txt in legend_texts:
      t = txt.get_text()
      if t in ('Reco. MC', 'Fit Components'):
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
    # yield comparison plot (expected vs fitted) for momentum
    try:
      plot_yield_comparison(mom_mag, mc_count, list_pdfs, filename_prefix='yield_compare_mom')
    except Exception:
      if logger:
        logger.log('yield comparison (mom) failed', 'info')
      else:
        print('yield comparison (mom) failed')
    # note: plot_yield_comparison belongs with momentum and time plotting callers
    

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
        # Align mc_count and time by trimming the longer array to the shorter
        time_flat = ak.to_numpy(ak.flatten(ak.drop_none(time), axis=None))
        try:
            mc_arr = np.asarray(mc_count).flatten()
        except Exception:
            mc_arr = np.array([])

        if len(mc_arr) != len(time_flat):
            nmin = min(len(mc_arr), len(time_flat))
            if logger:
                logger.log(f'mc_count/time length mismatch ({len(mc_arr)} vs {len(time_flat)}); trimming to {nmin}', 'info')
            else:
                print(f'[recoplot_module] mc_count/time length mismatch ({len(mc_arr)} vs {len(time_flat)}); trimming to {nmin}')
            mc_arr = mc_arr[:nmin]
            time_flat = time_flat[:nmin]

        # replace originals so existing ak.mask(...) calls work with aligned numpy/awkward data
        mc_count = mc_arr
        time = time_flat
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

    # combine DIO and CE into a single 'Stopped muon' curve for the fit
    stopped_plot = np.zeros(len(time_plot))
    stopped_names = ('DIO', 'CE')
    for name, pdfs, N_pdfs in list_pdfs:
      pdf_plot = (pdfs.pdf(time_plot) * N_pdfs * scale).numpy()
      combine_plot += pdf_plot
      if name in stopped_names:
        stopped_plot += pdf_plot
        continue
      # plot other components individually
      labs_fit.append(name)
      style = time_components.get(name, {})
      color = style.get('lineColor', 'k')
      linestyle = style.get('lineStyle', '-')
      ax1.plot(time_plot, pdf_plot, label=name, color=color, linestyle=linestyle)

    # plot combined stopped muon curve (DIO + CE) if present
    if np.any(stopped_plot):
      # choose a representative style (use DIO if available, else CE)
      stopped_style = time_components.get('DIO', time_components.get('CE', {}))
      stopped_color = stopped_style.get('lineColor', 'cyan')
      stopped_ls = stopped_style.get('lineStyle', '--')
      ax1.plot(time_plot, stopped_plot, label='Stopped muon', color=stopped_color, linestyle=stopped_ls)

    ax1.plot(time_plot, combine_plot, '-r', label='Total')
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.set_xlim(fit_range)
    ax1.set_ylim([1e-1, max(data_hist)])
    ax1.set_xlabel('Reconstructed Time [ns]', fontsize=16)
    ax1.set_ylabel('# of events per bin', fontsize=16)
    leg = ax1.legend(fontsize='large')
    # Make only the 'Reco. MC' and 'Fit Components' legend entries bold; others normal
    legend_texts = leg.get_texts()
    for txt in legend_texts:
      t = txt.get_text()
      if t in ('Reco. MC', 'Fit Components'):
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
       print('[py-fitter/recoplot_module/plottime_fit] ⚠️ WARNING! histogram empty')

    ax2.errorbar(time_plot, dev , yerr=err, color='None', marker='+', markerfacecolor='black', ecolor='black', capsize=3)
    ax2.grid(True)
    ax2.yaxis.set_ticks(np.arange(-5, 5,2))
    ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax2.set_xlim(fit_range)
    ax2.set_xlabel('Reconstructed Time [ns]', fontsize=16)
    ax2.set_ylabel('Normalized Residual', fontsize=16)
    # yield comparison plot (expected vs fitted) for momentum
    try:
      plot_yield_comparison(mom_mag, mc_count, list_pdfs, filename_prefix='yield_compare_mom')
    except Exception:
      if logger:
        logger.log('yield comparison (mom) failed', 'info')
      else:
        print('yield comparison (mom) failed')
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

    # plot side-by-side bars
    x = np.arange(len(procs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(procs)*1.2), 4))
    rects1 = ax.bar(x - width/2, expected, width, label='Expected')
    rects2 = ax.bar(x + width/2, fitted, width, label='Fitted')
    ax.set_ylabel('Counts')
    ax.set_title('Expected vs Fitted Yields')
    ax.set_xticks(x)
    ax.set_xticklabels(procs, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)

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
        if logger:
            logger.log(f"Saved yield comparison to {fname}", 'info')
        else:
            print(f"Saved yield comparison to {fname}")
    except Exception as e:
        if logger:
            logger.log(f"Failed to save yield comparison: {e}", 'error')
        else:
            print(f"Failed to save yield comparison: {e}")
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
    try:
        mc_arr = np.asarray(mc_count).flatten()
    except Exception:
        mc_arr = np.array([])

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
    ax.legend(ncol=2, fontsize='small')
    ax.grid(True, linestyle=':')
    ts = int(_time.time())
    fname = f"{filename_prefix}_{ts}.png"
    try:
      plt.tight_layout()
      plt.savefig(fname)
      if logger:
        logger.log(f'Saved momentum confusion plot to {fname}', 'info')
      else:
        print(f'Saved momentum confusion plot to {fname}')
    except Exception as e:
      if logger:
        logger.log(f'Failed to save momentum confusion plot: {e}', 'error')
      else:
        print(f'Failed to save momentum confusion plot: {e}')
    plt.close(fig)
    # Print summary totals to terminal for quick inspection
    print('\nMomentum bin-by-bin confusion summary:')
    for p in procs:
      ttrue = int(np.nan_to_num(totals_true[p], nan=0.0))
      tfit = int(np.nan_to_num(np.round(totals_fit[p]), nan=0.0))
      print(f"  {p}: true N = {ttrue}, fitted N = {tfit}")

    # Print per-bin arrays for each process: true counts, fitted counts, and relative fractions
    try:
      print('\nBin centers (MeV):')
      print(np.array2string(centers, precision=2, separator=', '))
      for p in procs:
        tcounts = np.nan_to_num(true_counts[p], nan=0.0)
        fcounts = np.nan_to_num(fitted_counts[p], nan=0.0)
        tfr = np.nan_to_num(true_frac[procs.index(p)], nan=0.0)
        ffr = np.nan_to_num(fit_frac[procs.index(p)], nan=0.0)
        print(f"\nProcess {p}:")
        print('  True counts per bin:   ', np.array2string(tcounts, precision=3, separator=', '))
        print('  Fitted counts per bin: ', np.array2string(fcounts, precision=3, separator=', '))
        print('  True frac per bin:     ', np.array2string(tfr, precision=3, separator=', '))
        print('  Fit frac per bin:      ', np.array2string(ffr, precision=3, separator=', '))
    except Exception:
      pass

    return {'bins': centers, 'true_counts': true_counts, 'fitted_counts': fitted_counts, 'true_frac': true_frac, 'fit_frac': fit_frac, 'totals_true': totals_true, 'totals_fit': totals_fit}
