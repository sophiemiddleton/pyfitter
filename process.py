import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse
import csv
import traceback
import os
import matplotlib.pyplot as _plt
import numpy as _np
from control_region import ControlRegion
import pickle
from pyutils.pylogger import Logger
from pathlib import Path
from config import GLOBAL_VERBOSITY

# Module-level logger
module_logger = Logger(print_prefix='[process] ', verbosity=GLOBAL_VERBOSITY)


from fit_module import *
from results_module import ResultsClass
from analyze import Analyze
from data_prep import DataPreparationManager, safe_float_conversion, safe_dict_get, safe_numpy_convert, safe_field_extraction
from model.physics_components import mom_components
from pyutils.pycut import CutManager
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyplot import Plot
from pyutils.pyprint import Print
from pyutils.pyselect import Select
from pyutils.pyvector import Vector

class AnaProcessor(Skeleton):
    """custom file processor 
    
    This class inherits from the Skeleton defined in pyutils/pyprocess base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self, file_list_path, jobs=1, cuts=None, location='disk', mom_lo = 97, mom_hi = 115):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = file_list_path

        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
                "trig_apr_TrkDe_80m70p",
                "trig_cpr_TrkDe_80m70p",
                "trig_tpr_TrkDe_80m70p"
            ],
            "crv" : [
                "crvcoincs.time",
                "crvcoincs.nHits",
                "crvcoincs.PEs",
                "crvcoincs.timeStart",
                "crvcoincs.timeEnd"
            ],
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trk.status",
                "trkqual.valid",
                "trkqual.result",
                "trkpid.valid",
                "trkpid.result"
            ],
            "trkfit" : [
                "trksegs",
                "trksegsmc",
                "trksegpars_lh"
            ],
            "trkmc" : [
                "trkmcsim",
                "trkmc.valid"
            ]
        }
        self.tree_path = "ntuple"
        self.use_remote = True     # Use remote file via mdh
        if str(location)  == "local":
          self.use_remote = False
        self.location = str(location)     # File location
        self.max_workers = jobs      # Limit the number of workers
        self.verbosity = GLOBAL_VERBOSITY         # Set verbosity from global config
        self.use_processes = True  # Use processes rather than threads
        
        cuts_list = cuts if cuts is not None else []
        self.analyse = Analyze(verbosity=self.verbosity, cut_switch=cuts_list, mom_lo = mom_lo, mom_hi = mom_hi)
            
        self.print_prefix = "[AnaProcessor] "
        self.logger = Logger(print_prefix=self.print_prefix, verbosity=self.verbosity)
        self.logger.log("Initialised", "info")
    
    # This method overrides the parent class's process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """
        try:
            # Create a local pyprocess Processor to extract data from this file
            # This uses the configuration parameters from our class
            processor = Processor(
                use_remote=self.use_remote,     # Use remote file via mdh
                location=self.location,         # File location
                verbosity=self.verbosity
            )
            
            # Process the files using multithreading
            data = processor.process_data(
                file_name = file_name,
                branches = self.branches
            )
            
            # ---- Analysis ----            
            results = self.analyse.execute(data, file_name)

            # Clean up
            gc.collect()

            return results 
        
        except Exception as e:
            self.logger.log(f"Error processing {file_name}: {e}", "error")
            self.logger.log(traceback.format_exc(), "max")
            return None
            
def combine_cut_flows( cut_flow_list, csv_basename: str = None):
    """Combine a list of cut flows after multiprocessing 
    
    Args:
        cut_flows: List of cut statistics lists from different files

    Returns:
        list: Combined cut statistics
    """        
    # Use the first (now filtered) list as template
    template = cut_flow_list[0]
    # Use the template to initialise combined stats
    combined_cut_flow = []
    for cut in template:
        cut_copy = {k: v for k, v in cut.items()}
        cut_copy["events_passing"] = 0
        combined_cut_flow.append(cut_copy)

    # Create a mapping of cut names to indices in combined_stats 
    cut_name_to_index = {cut["name"]: i for i, cut in enumerate(combined_cut_flow)}

    # Sum up events_passing for each cut across all files
    for cut_flow in cut_flow_list:
        for cut in cut_flow:
            cut_name = cut["name"]
            if cut_name in cut_name_to_index:
                idx = cut_name_to_index[cut_name]
                combined_cut_flow[idx]["events_passing"] += cut["events_passing"]

    # Recalculate percentages
    if combined_cut_flow and combined_cut_flow[0]["events_passing"] > 0:
        total_events = combined_cut_flow[0]["events_passing"]
        for i, cut in enumerate(combined_cut_flow):
            events = cut["events_passing"]
            cut["absolute_frac"] = (events / total_events) * 100.0
            if i == 0:
                cut["relative_frac"] = 100.0
            else:
                prev_events = combined_cut_flow[i-1]["events_passing"]
                cut["relative_frac"] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0


    cut_manager = CutManager(verbosity=GLOBAL_VERBOSITY)
    print("================== Total Cut Flow =======================")
    try:
        df = cut_manager.format_cut_flow(combined_cut_flow)
        print(df)
    except Exception as e_print:
        module_logger.log(f'[combine_cut_flows] Failed to format cut flow: {e_print}', 'error')

    csv_name = "cut_stats.csv" if not csv_basename else f"{csv_basename}.csv"
    try:
        if combined_cut_flow:
            fieldnames = list(combined_cut_flow[0].keys())
        else:
            fieldnames = ["name", "events_passing", "absolute_frac", "relative_frac"]
        with open(csv_name, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in combined_cut_flow:
                out_row = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(out_row)
        module_logger.log(f'Wrote combined cut flow to {csv_name}', 'info')
    except Exception as e_csv:
        module_logger.log(f'Failed to write combined cut flow to {csv_name}: {e_csv}', 'error')

    return combined_cut_flow

        
def combine_arrays(results):
    """Combine filtered arrays from multiple files
    """
    arrays_to_combine = []
    # Check if we have results
    if not results:
        return None
    # Loop through all files
    for i, result in enumerate(results):
        if len(result) == 0:
            continue
        # Concatenate arrays
        arrays_to_combine.append(result["filtered_data"])
    return ak.concatenate(arrays_to_combine)


def save_fit_npz(basename, fitresult, par=None, loss=None, nlls=None, extra=None):
    """Save a compact NPZ summary of a fit result for systematics studies.

    Produces <basename>_fit.npz containing:
      - param_names: array of parameter names
      - param_values: array of float values
      - param_errors: array of float (nan if unavailable)
      - loss: scalar (if present)
      - nlls: array/list (if present)
      - extra: saved under 'extra' if provided (must be numpy-able)
    """
    try:
        if not basename:
            basename = 'fitresult'
        base_dir = Path(__file__).resolve().parent
        fits_dir = base_dir / 'fits'
        fits_dir.mkdir(parents=True, exist_ok=True)
        pattern = f"{basename}_fit*.npz"
        existing = sorted(fits_dir.glob(pattern))
        if not existing:
            fname = fits_dir / f"{basename}_fit.npz"
        else:
            max_v = 0
            for p in existing:
                nm = p.stem
                if nm.endswith('_fit'):
                    max_v = max(max_v, 0)
                else:
                    parts = nm.rsplit('_v', 1)
                    if len(parts) == 2:
                        vnum = safe_float_conversion(parts[1], default=0)
                        max_v = max(max_v, int(vnum))
            next_v = max_v + 1
            fname = fits_dir / f"{basename}_fit_v{next_v:03d}.npz"
        
        # Extract parameters using safe utilities (no nested try/except)
        param_names = []
        param_values = []
        param_errors = []
        
        # Get params from fitresult (dict-like or object)
        params = safe_dict_get(fitresult, 'params')
        
        if params is not None:
            for name in params:
                entry = params[name]
                val = safe_dict_get(entry, 'value')
                err = safe_dict_get(entry, 'error')
                
                param_names.append(name)
                param_values.append(safe_float_conversion(val))
                param_errors.append(safe_float_conversion(err))
        
        # Build output dictionary
        tosave = {}
        tosave['param_names'] = np.array(param_names, dtype=object)
        tosave['param_values'] = np.array(param_values, dtype=float)
        tosave['param_errors'] = np.array(param_errors, dtype=float)
        
        if loss is not None:
            tosave['loss'] = safe_numpy_convert(loss, dtype=float)
        
        if nlls is not None:
            tosave['nlls'] = safe_numpy_convert(nlls, dtype=float)
        
        if extra is not None:
            if isinstance(extra, dict):
                for k, v in extra.items():
                    tosave[k] = safe_numpy_convert(v, default_repr=True)
            else:
                tosave['extra'] = safe_numpy_convert(extra, default_repr=True)
        
        np.savez_compressed(str(fname), **tosave)
        module_logger.log(f'[process] Wrote fit summary to {fname}', 'info')
    except Exception as e:
        module_logger.log(f'[process] Failed to write fit NPZ: {e}', 'error')


def process_offspill_filelist(filelist_path: str = 'OffSpill_10.txt', #FIXME should use dataset definitions instead of hardcoding
                             out_prefix: str = 'offspill_control',
                             location: str = 'local',
                             cuts=None,
                             mom_lo: float = 95.0,
                             mom_hi: float = 110.0,
                             jobs: int = 16):
    """Process a text file listing OffSpill files and save combined filtered results.

    The function will instantiate `AnaProcessor` (with `location`), call
    `process_file` for each entry in `filelist_path`, combine results using
    `combine_arrays`, and write a pickle of the combined filtered data.

    It will also attempt to extract a flattened `mom_mag` array and save it
    as a small `.npz` file for downstream control-region fits.
    """
    try:
        with open(filelist_path, 'r') as f:
            files = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    except Exception as e:
        module_logger.log(f'[process_offspill_filelist] Failed to read {filelist_path}: {e}', 'error')
        return None

    if len(files) == 0:
        module_logger.log(f'[process_offspill_filelist] No files found in {filelist_path}', 'error')
        return None

    # Create AnaProcessor with the same selection/cuts as main analysis
    ana = AnaProcessor(file_list_path=filelist_path, jobs=jobs, cuts=cuts, location=location, mom_lo=mom_lo, mom_hi=mom_hi)

    results = []
    for fn in files:
        try:
            r = ana.process_file(fn)
            if r is not None:
                results.append(r)
        except Exception as e:
            module_logger.log(f'[process_offspill_filelist] Error processing {fn}: {e}', 'error')

    if len(results) == 0:
        module_logger.log('[process_offspill_filelist] No results produced', 'error')
        return None

    # Combine filtered arrays and save
    try:
        combined = combine_arrays(results)
        cutlist = []
        for i, result in enumerate(results):
            cutlist.append(result["cut_stats"])
        combine_cutflows = combine_cut_flows(cutlist, csv_basename=out_prefix)
    except Exception as e:
        module_logger.log(f'[process_offspill_filelist] Failed to combine arrays: {e}', 'error')
        combined = None

    
    try:
        with open(out_prefix + '_filtered.pkl', 'wb') as pf:
            pickle.dump({'results': results, 'combined_filtered': combined}, pf)
        module_logger.log(f'[process_offspill_filelist] Wrote {out_prefix}_filtered.pkl', 'info')
    except Exception as e:
        module_logger.log(f'[process_offspill_filelist] Failed to write pickle: {e}', 'error')

    # Try to extract a flattened mom_mag for convenience.
    try:
        mom_flat = None
        
        # Primary method: use Vector utility
        try:
            selector = Select()
            trk_front = selector.select_surface(combined['trkfit'], surface_name='TT_Front')
            trkfit_ent = ak.mask(combined['trkfit']["trksegs"], trk_front)
            vector = Vector(verbosity=GLOBAL_VERBOSITY)
            mom_per_event = vector.get_mag(trkfit_ent, 'mom')
            mom_flat = ak.to_numpy(ak.flatten(mom_per_event, axis=None))
        except Exception as e_primary:
            # Fallback 1: Try field extraction
            mom_arr = safe_field_extraction(combined, 'trk', 'mom')
            if mom_arr is None:
                mom_arr = safe_field_extraction(combined, 'trkfit', 'trksegs', 'mom')
            
            if mom_arr is not None:
                mom_flat = ak.to_numpy(ak.flatten(mom_arr, axis=None))
            else:
                # Fallback: Try Vector with direct access
                try:
                    _vec = Vector(verbosity=GLOBAL_VERBOSITY)
                    trkfit_ent = combined['trkfit']
                    mom_per_event = _vec.get_mag(trkfit_ent, 'mom')
                    mom_flat = ak.to_numpy(ak.flatten(mom_per_event, axis=None))
                except Exception as e2:
                    module_logger.log(f'[process_offspill_filelist] Could not extract mom_mag: primary: {e_primary}, fallback2: {e2}', 'error')
                    mom_flat = None

        if mom_flat is not None:
            np.savez(out_prefix + '_mom_mag.npz', mom_mag=mom_flat)
            module_logger.log(f'[process_offspill_filelist] Wrote {out_prefix}_mom_mag.npz', 'info')
            try:
                cr = ControlRegion(mom_flat)
                fit_out = cr.fit_cosmic(fit_range=(mom_lo, mom_hi), plot=True)
                fig = fit_out.get('figure', None)
                if fig is not None:
                    fname_plot = out_prefix + '_cosmic_fit.png'
                    try:
                        fig.savefig(fname_plot)
                        module_logger.log(f'[process_offspill_filelist] Wrote cosmic fit figure to {fname_plot}', 'info')
                    except Exception as e_save:
                        module_logger.log(f'[process_offspill_filelist] Failed to save cosmic fit figure: {e_save}', 'error')
                params = fit_out.get('params', {})
                hesse = fit_out.get('hesse', {})
                module_logger.log('[process_offspill_filelist] Cosmic fit parameters:', 'info')
                for k, v in params.items():
                    err = None
                    try:
                        err = hesse.get(next(iter([p for p in hesse.keys() if getattr(p, 'name', p) == k]), None), None)
                    except Exception:
                        err = None
                    if isinstance(err, dict) and 'error' in err:
                        module_logger.log(f'  {k}: {v} ± {err["error"]}', 'info')
                    else:
                        module_logger.log(f'  {k}: {v}', 'info')
            except Exception as e_cr:
                module_logger.log(f'[process_offspill_filelist] ControlRegion fit failed: {e_cr}', 'error')
                try:
                    module_logger.log(f'[process_offspill_filelist] mom_flat type: {type(mom_flat)}, shape: {getattr(mom_flat, "shape", None)}', 'error')
                    try:
                        sample = repr(mom_flat[:50])
                    except Exception:
                        sample = str(mom_flat)
                    module_logger.log(f'[process_offspill_filelist] mom_flat sample (first 50): {sample}', 'error')
                    arr = _np.asarray(mom_flat)
                    if arr.size == 0:
                        module_logger.log('[process_offspill_filelist] mom_flat is empty', 'error')
                    else:
                        try:
                            vmin = _np.nanmin(arr)
                            vmax = _np.nanmax(arr)
                            vmean = _np.nanmean(arr)
                            vstd = _np.nanstd(arr)
                            module_logger.log(f'[process_offspill_filelist] mom_flat min/max/mean/std: {vmin:.6g}/{vmax:.6g}/{vmean:.6g}/{vstd:.6g}', 'info')
                        except Exception as e_stat:
                            module_logger.log(f'[process_offspill_filelist] Failed computing stats: {e_stat}', 'error')
                        try:

                            fig_debug = _plt.figure()
                            finite = _np.isfinite(arr)
                            if finite.any():
                                _plt.hist(arr[finite], bins=80)
                            else:
                                _plt.hist(arr, bins=80)
                            _plt.xlabel('mom_mag')
                            _plt.ylabel('counts')
                            _plt.title('Debug: mom_mag distribution')
                            debug_fname = out_prefix + '_mom_mag_debug.png'
                            fig_debug.savefig(debug_fname)
                            module_logger.log(f'[process_offspill_filelist] Wrote debug histogram to {debug_fname}', 'info')
                            _plt.close(fig_debug)
                        except Exception as e_plot:
                            module_logger.log(f'[process_offspill_filelist] Failed to create debug histogram: {e_plot}', 'error')
                except Exception as e_diag:
                    module_logger.log(f'[process_offspill_filelist] Failed to print mom_flat diagnostics: {e_diag}', 'error')
        else:
            module_logger.log(f'[process_offspill_filelist] Skipped writing mom_mag: could not extract from combined array', 'error')
    except Exception as e:
        module_logger.log(f'[process_offspill_filelist] Could not extract mom_mag: {e}', 'error')

    return {'results': results, 'combined_filtered': combined}



    


def count_particle_types(data):
  """
  Counts the occurrences of different particle types based on
  simulation data, leveraging the properties of Awkward Arrays.

  Args:
      data (ak.Array): An Awkward Array containing simulation data,
                       including 'trkmc' with 'trkmcsim' nested field.

  Returns:
      list: A list containing particle type identifiers for each event.
  """

  # Check for empty data
  if ak.num(data['trkmc'], axis=0) == 0:
      if module_logger:
          module_logger.log('No events found in the data.', 'info')
      else:
          print("No events found in the data.")
      return []

  # Vectorized approach for efficiency using Awkward Array operations
  #  This is generally faster than looping through events individually for large datasets.

  # Get startCode for the first track in each event, handling empty lists
  # Use ak.firsts to safely get the first element or None if the list is empty
  proc_codes = ak.firsts(data['trkmc']['trkmcsim', 'startCode'], axis=1) 
  gen_codes = ak.firsts(data['trkmc']['trkmcsim', 'gen'], axis=1)
  vector = Vector(verbosity=GLOBAL_VERBOSITY)

  #rhos = vector.get_rho(data['trkmc','trkmcsim'],'pos')
  vec = vector.get_vector(branch=data['trkmc','trkmcsim'],vector_name='pos')
  rhos = vec.rho
  position = ak.firsts(rhos, axis=1) 

  #position = ak.firsts(sim_pos_vec.rho, axis = 1)
  # Use vectorized comparisons and selection for counting
  dio_mask = (proc_codes == 166) & (position <= 75) # Create boolean mask for DIO events
  ipa_mask = (proc_codes == 166) & (position > 75) # Create boolean mask for IPA DIO events
  cem_mask = ((proc_codes == 168)  | (proc_codes == 167)  ) # Create boolean mask for CE events
  cep_mask = ((proc_codes == 176) | (proc_codes == 169) )  # Create boolean mask for CE events
  erpc_mask = (proc_codes == 178)  # Create boolean mask for external RPC events
  irpc_mask = (proc_codes == 179)  # Create boolean mask for internal RPC events
  ermc_mask = (proc_codes == 172)  # Create boolean mask for external RMC events
  irmc_mask = (proc_codes == 171)  # Create boolean mask for internal RMC events
  cosmic_mask = ((gen_codes == 44) | (gen_codes == 38))  # Create boolean mask for cosmic events

  # Combine masks to identify 'other' events
  other_mask = ~(dio_mask | cem_mask | erpc_mask | irpc_mask | cosmic_mask | ipa_mask | irmc_mask | ermc_mask | cep_mask)

  # Initialize particle_count with -2 for 'others'
  particle_count = ak.zeros_like(proc_codes, dtype=int) - 2
  
  # Assign particle types based on masks
  particle_count = ak.where(dio_mask, 166, particle_count)
  particle_count = ak.where(ipa_mask, 0, particle_count)
  particle_count = ak.where(cosmic_mask, -1, particle_count)
  particle_count = ak.where(other_mask, -2, particle_count)
  particle_count = ak.where(irpc_mask, 179, particle_count)
  particle_count = ak.where(erpc_mask, 178, particle_count)
  particle_count = ak.where(irmc_mask, 171, particle_count)
  particle_count = ak.where(ermc_mask, 172, particle_count)
  particle_count = ak.where(cem_mask, 168, particle_count)
  particle_count = ak.where(cep_mask, 176, particle_count)
  particle_count_return = particle_count
  #particle_count = ak.any(dio_mask, axis=1)
  # Count the occurrences of each particle type
  counts = {
      166: (len(particle_count[ak.any(dio_mask, axis=1)==True])),
      0: (len(particle_count[ak.any(ipa_mask, axis=1)==True])),
      168:  (len(particle_count[ak.any(cem_mask, axis=1)==True])),
      176:  (len(particle_count[ak.any(cep_mask, axis=1)==True])),
      178:  (len(particle_count[ak.any(erpc_mask, axis=1)==True])),
      179:  (len(particle_count[ak.any(irpc_mask, axis=1)==True])),
      171:  (len(particle_count[ak.any(irmc_mask, axis=1)==True])),
      172:  (len(particle_count[ak.any(ermc_mask, axis=1)==True])), 
      -1:  (len(particle_count[ak.any(cosmic_mask, axis=1)==True])),
      -2:  (len(particle_count[ak.any(other_mask, axis=1)==True])),
  }
    
  # Print the yields to terminal for cross-check
  if module_logger:
      module_logger.log('===== MC truth yields for full momentum and time range=====', 'info')
      module_logger.log(f'N_DIO: {counts[166]}', 'info')
      module_logger.log(f'N_IPA: {counts[0]}', 'info')
      module_logger.log(f'N_CEM: {counts[168]}', 'info')
      module_logger.log(f'N_CEP: {counts[176]}', 'info')
      module_logger.log(f'N_eRPC: {counts[178]}', 'info')
      module_logger.log(f'N_iRPC: {counts[179]}', 'info')
      module_logger.log(f'N_eRMC: {counts[171]}', 'info')
      module_logger.log(f'N_iRMC: {counts[172]}', 'info')
      module_logger.log(f'N_cosmic: {counts[-1]}', 'info')
      module_logger.log(f'N_others: {counts[-2]}', 'info')
  else:
      print("===== MC truth yields for full momentum and time range=====")
      print("N_DIO: ", counts[166])
      print("N_IPA: ", counts[0])
      print("N_CEM: ", counts[168])
      print("N_CEP: ", counts[176])
      print("N_eRPC: ", counts[178])
      print("N_iRPC: ", counts[179])
      print("N_eRMC: ", counts[171])
      print("N_iRMC: ", counts[172])
      print("N_cosmic: ", counts[-1])
      print("N_others: ", counts[-2])
  
  # Now return a 1D list with one element per event corresponding to the primary trk
  #particle_count_return = ak.flatten(particle_count_return, axis=None)
  #    The mask will be True for values that are not -2.
  primary_mask = particle_count_return != -2

  # Apply the mask to the flattened array to select desired elements
  particle_count_return = particle_count_return[primary_mask]
  particle_count_return = [[sublist[0]] for sublist in particle_count_return]
  particle_count_return = ak.flatten(particle_count_return, axis=None)
  if module_logger:
      module_logger.log(f'returned particle count length {len(particle_count_return)}', 'max')
  else:
      print("returned particle count length",len(particle_count_return))
  
  return particle_count_return
  
# Create an instance of our custom processor
def main(args):
    """Main driver function to run analysis."""

    # list which cuts to switch on/off (positional):
    # sw(0)=is_reco_electron, sw(1)=has_downstream, sw(2)=good_trkqual, sw(3)=good_trkpid,
    # sw(4)=has_hits, sw(5)=within_t0, sw(6)=within_t0err, sw(7)=within_lhr_max,
    # sw(8)=within_d0, sw(9)=within_pitch_angle, sw(10)=no_crv_veto, sw(11)=has_st,
    # sw(12)=no_opa, sw(13)=in_mom_range

   
    #off_spill_cosmics = [True, True, True, True, True, False, True, False, False, False, True, True, True, True]
    nocuts = [False] * 16


    # Convert positional list to named switches for robustness
    new = [
        True,  # 0 is_reco_electron
        True,  # 1 has_downstream
        True, # 2 has trk front
        True,  # 3 good_trkqual
        True,  # 4 good_trkpid
        True, # 5 within_t0
        True,  # 6 within_t0err
        True,  # 7 has_hits
        True,  #8 has_st
        True,  #9 no_opa
        True,  #10 no_crv_veto
        True,  #11 no_crv_quality
        True,  #12 no_crv_timewindow
        True,  #13 pz/pt
        True,  #14 triggers
        True,  #15 in_mom_range
    ]
    cut_names = [
        "is_reco_electron", #True
        "has_downstream",
        "has_trk_front", 
        "good_trkqual",
        "good_trkpid",
        "within_t0",
        "has_hits",
        "within_t0err",
        "no_crv_veto",
        "no_crv_quality",
        "no_crv_timewindow",
        "has_st",
        "no_opa",
        "pz_over_pt",
        "good_trigger",
        "in_mom_range"
    ]

    
    # now run main analysis
    named_switches = dict(zip(cut_names, new))
    module_logger.log(f"selection cuts to be applied : {named_switches}", "info")


    # run control sample analysis:
    named_switches_offspill = dict(zip(cut_names, new))
    module_logger.log(f"selection cuts to be applied : {named_switches_offspill}", "info")
    if getattr(args, 'control_fit', False):
        # run OffSpill mom-spectrum control-region fit (poly2) if the file exists
        try:
            process_offspill_filelist('OffSpill_10.txt', 
            out_prefix='offspill_control', 
            location='tape', 
            cuts=named_switches_offspill, 
            mom_lo=args.fitrange_low[0], 
            mom_hi=args.fitrange_hi[0], 
            jobs=16)
        except Exception as e:
            module_logger.log(f'OffSpill control-region fit failed: {e}', 'error')

    ana_processor = AnaProcessor(args.file, args.jobs, named_switches, args.loc, args.fitrange_low[0], args.fitrange_hi[0])
    results = ana_processor.execute()

    # Combine arrays and cut statistics
    pre_fit = combine_arrays(results)
    cutlist = []
    for i, result in enumerate(results):
        cutlist.append(result["cut_stats"])
    # Derive a basename for the CSV from the input filename (nts.mu2e.NAME.version.seq.root -> NAME)
    try:
        file_basename = os.path.basename(args.file)
        parts = file_basename.split('.')
        if len(parts) >= 3:
            csv_base = parts[2]
        else:
            csv_base = os.path.splitext(file_basename)[0]
    except Exception:
        csv_base = None

    combine_cutflows = combine_cut_flows(cutlist, csv_basename=csv_base)

    # Run mc_count
    mc_count = count_particle_types(pre_fit)

    # select only track front to fit to
    selector = Select()
    trk_front = selector.select_surface(pre_fit['trkfit'], surface_name='TT_Front')

    trkfit_ent = ak.mask(pre_fit['trkfit']["trksegs"], trk_front)

    vector = Vector(verbosity=GLOBAL_VERBOSITY)
    # make vector mag branch
    mom_mag = vector.get_mag(trkfit_ent, 'mom')

    time = DataPreparationManager.clean_awkward_array(trkfit_ent['time'])

    # Build track-aligned mc_count arrays for momentum and time using per-event counts
    try:
        event_mc = np.asarray(mc_count).flatten()
        counts_mom = ak.to_numpy(ak.num(mom_mag, axis=1))
        counts_time = ak.to_numpy(ak.num(trkfit_ent['time'], axis=1))

        if len(event_mc) != len(counts_mom) or len(event_mc) != len(counts_time):
            raise ValueError(f'mc_count length ({len(event_mc)}) does not match number of events (mom:{len(counts_mom)}, time:{len(counts_time)})')

        # Repeat event-level labels per track and unflatten to match nested structure
        if counts_mom.sum() > 0:
            rep_mom = np.repeat(event_mc, counts_mom)
            mc_count_track_mom = ak.unflatten(rep_mom, counts_mom)
        else:
            mc_count_track_mom = ak.Array([])

        if counts_time.sum() > 0:
            rep_time = np.repeat(event_mc, counts_time)
            mc_count_track_time = ak.unflatten(rep_time, counts_time)
        else:
            mc_count_track_time = ak.Array([])
    except Exception as e_mc_expand:
        # As a final fallback, attempt to broadcast; but prefer failing loudly
        try:
            mc_count_track_mom = ak.broadcast_arrays(ak.fill_none(mom_mag, None), mc_count)[1]
            mc_count_track_time = ak.broadcast_arrays(ak.fill_none(trkfit_ent['time'], None), mc_count)[1]
        except Exception:
            raise RuntimeError(f'Failed to build track-aligned mc_count arrays: {e_mc_expand}')

    # check mom and time sizes mathch and log counts
    if 'mom_mag' in locals():
        try:
            n_mom = len(ak.flatten(mom_mag, axis=None))
        except Exception:
            n_mom = 'unknown'
    else:
        n_mom = 'missing'
    if 'time' in locals():
        try:
            n_time = len(ak.flatten(time, axis=None))
        except Exception:
            n_time = 'unknown'
    else:
        n_time = 'missing'
    module_logger.log(f"mom entries: {n_mom}, time entries: {n_time}", "info")

    # Save nominal momentum (and time) arrays for downstream sensitivity studies
    try:
        mom_flat = ak.to_numpy(ak.flatten(mom_mag, axis=None))
        time_flat = ak.to_numpy(ak.flatten(time, axis=None))
        
        # Save to file for sensitivity scan
        if csv_base:
            npz_out = f"{csv_base}_mom_mag.npz"
            tosave_sens = {'mom_mag': mom_flat}
            if len(time_flat) > 0 and len(time_flat) == len(mom_flat):
                tosave_sens['time'] = time_flat
            np.savez_compressed(npz_out, **tosave_sens)
            module_logger.log(f'Saved nominal data for sensitivity studies to {npz_out}', 'info')
    except Exception as e_sens:
        module_logger.log(f'[process] Failed to save nominal data NPZ for sensitivity scan: {e_sens}', 'warning')

    if args.fittype == "mom1D":
        module_logger.log(f"Building mom 1D fit", "info")
        fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
            mom_mag,
            mc_count_track_mom,
            args.fitrange_low[0],
            args.fitrange_hi[0],
            plot_truth=True,
            verbose=args.verbose,
            constraints_dir='uncertainties/outputs',
            plot_NLL=False,
            plot_results=True
        )
        # Print full fit result
        print("\n================== 1D Momentum Fit Result =======================")
        try:
            if hasattr(fitresult, 'params'):
                print(fitresult)
            else:
                print(fitresult)
        except Exception as e:
            print(f"Could not print fit result: {e}")
        print("="*60 + "\n")
        # Save fit summary for systematics studies
        try:
            save_fit_npz(f"{csv_base}_mom1D", fitresult, par=par, loss=loss, nlls=nlls, extra={'mom_mag': mom_mag})
        except Exception as e_save:
            module_logger.log(f'[process] Failed to save mom1D fit npz: {e_save}', 'error')
        if int(args.interpret) == 1:
            result_output = ResultsClass(mom_mag, fitresult, args.verbose)
            result_output.WriteFittedData(args.fitrange_low[0], args.fitrange_hi[0])
            result_output.WriteResult()
            result_output.GetSignifcance(par, loss, 'asym')
            if int(args.setlimit) == 1:
                result_output.GetUL(
                    poi=par,
                    loss=loss,
                    nlls=nlls,
                    model=combine_pdf,
                    constraints=constraints,
                    fit_range_low=args.fitrange_low[0],
                    fit_range_hi=args.fitrange_hi[0],
                    signal_yield=fitresult.params['N_CE']['value'],
                    cl=0.90,
                    method='asym',
                )

    elif args.fittype == "time1D":
        module_logger.log(f"Building time 1D fit", "info")
        fitresult, par, loss, combine_pdf = Unbinned_fit_time(
            time,
            mc_count_track_time,
            float(args.fitrange_low[1]),
            float(args.fitrange_hi[1]),
            plot_truth=True,
            verbose=args.verbose,
            plot_NLL=True,
            plot_results=True
        )

        # Print full fit result
        print("\n================== 1D Time Fit Result =======================")
        try:
            if hasattr(fitresult, 'params'):
                print(fitresult)
            else:
                print(fitresult)
        except Exception as e:
            print(f"Could not print fit result: {e}")
        print("="*60 + "\n")
        # Save time fit summary
        try:
            save_fit_npz(f"{csv_base}_time1D", fitresult, par=par, loss=loss, extra={'time': time})
        except Exception as e_save:
            module_logger.log(f'[process] Failed to save time1D fit npz: {e_save}', 'error')

    elif args.fittype == "2D":
        module_logger.log(f"Building 2D fit", "info")
        mom_flat_len = len(ak.flatten(mom_mag, axis=None))
        time_flat_len = len(ak.flatten(time, axis=None))
        if mom_flat_len != time_flat_len:
            raise ValueError(f'Cannot run 2D fit: flattened mom length ({mom_flat_len}) != flattened time length ({time_flat_len}). Provide track-aligned mc_count or fix selections.')
        fitresult, par, loss, combine_pdf, norms = Unbinned_2d_fit_mom_time(
            mom_mag,
            time,
            mc_count_track_mom,
            [args.fitrange_low[0], args.fitrange_hi[0]],
            [args.fitrange_low[1], args.fitrange_hi[1]],
            plot_truth=False,
            verbose=args.verbose,
            plot_NLL=False,
            plot_results=True,
            constraints_dir='uncertainties/outputs',
        )
        if int(args.interpret) == 1:
            result_output = ResultsClass(mom_mag, fitresult, args.verbose)
            result_output.WriteFittedData(args.fitrange_low[0], args.fitrange_hi[0])
            result_output.WriteResult()
            result_output.GetSignifcance(par, loss, 'asym')
            if int(args.setlimit) == 1:
                result_output.GetUL(
                    poi=par,
                    loss=loss,
                    nlls=nlls,
                    model=combine_pdf,
                    constraints=constraints,
                    fit_range_low=args.fitrange_low[0],
                    fit_range_hi=args.fitrange_hi[0],
                    signal_yield=fitresult.params['N_CE']['value'],
                    cl=0.90,
                    method='asym',
                )
        # Print full fit result
        print("\n================== 2D Fit Result =======================")
        try:
            if hasattr(fitresult, 'params'):
                print(fitresult)
            else:
                print(fitresult)
        except Exception as e:
            print(f"Could not print fit result: {e}")
        print("="*60 + "\n")
        # Save 2D fit summary
        try:
            save_fit_npz(f"{csv_base}_2D", fitresult, par=par, loss=loss, extra={'mom_mag': mom_mag, 'time': time})
        except Exception as e_save:
            module_logger.log(f'[process] Failed to save 2D fit npz: {e_save}', 'error')

    else:
        raise Exception(
            "[py-fitter/main] ❌ ERROR: choice of fit type does not exist, please choose: mom1D, time1D or momtime2D"
        )

        
  
def PrintArgs(args):
    """
    prints users input parameters
    """
    print("========= [py-fitter/main]✅  Analyzing with user opts: ===========")
    module_logger.log('Analyzing with user opts', 'info')
    module_logger.log(f'file: {args.file}', 'info')
    module_logger.log(f'location: {args.loc}', 'info')
    module_logger.log(f'number of processes (njobs - optimal is 1 per file): {args.jobs}', 'info')
    module_logger.log(f'fittype: {args.fittype}', 'info')
    module_logger.log(f'range: {args.fitrange_low} {args.fitrange_hi}', 'info')
    module_logger.log(f'verbose: {args.verbose}', 'info')
    module_logger.log(f'interpret: {args.interpret}', 'info')
    module_logger.log(f'setlimit: {args.setlimit}', 'info')

if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=False, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--fittype", type=str, default="mom1D", help="fittype implemented opts: mom1D, time1D, momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=[97,475], nargs='+', help="minimum to fit ordered mom, time")
    parser.add_argument("--fitrange_hi", type=float, default=[110,1650], nargs='+',help="maximum to fit  ordered mom, time")
    parser.add_argument("--interpret", type=int, default=0, help="allows for significance evaluation")
    parser.add_argument("--setlimit", type=int, default=0, help="assumes low signal and will try to set limit")
    parser.add_argument("--verbose", default=1, help="verbose")
    parser.add_argument("--loc", type=str, required=False, default='disk', help="location of files")
    parser.add_argument("--control-fit", dest='control_fit', action='store_true', help="Run control-region fit for OffSpill (default: off)")
    args = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)




