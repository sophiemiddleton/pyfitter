import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse
import csv
import traceback
from pyutils.pylogger import Logger

# Module-level logger
try:
    module_logger = Logger(print_prefix='[process] ', verbosity=2)
except Exception:
    module_logger = None

from fit_module import *
from results_module import ResultsClass
from analyze import Analyze
from mom_components import mom_components
from cut_manager import CutManager
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
    def __init__(self, file_list_path, jobs=1, cuts=None, location='disk', mom_lo = 95, mom_hi = 115):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        # Call the parent class's __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = file_list_path

        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
            ],
            "crv" : [
                "crvcoincs.time",
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
        #self.filelist = "filelist.txt"          # text file containing list of files
        self.use_remote = True     # Use remote file via mdh
        if str(location)  == "local":
          self.use_remote = False
        self.location = str(location)     # File location
        self.max_workers = jobs      # Limit the number of workers
        self.verbosity = 2         # Set verbosity 
        self.use_processes = True  # Use processes rather than threads
        
        # Now add your own analysis-specific parameters 

        # Init analysis methods
        # Avoid mutable default for cuts
        cuts_list = cuts if cuts is not None else []
        # Would be good to load an analysis config here 
        self.analyse = Analyze(verbosity=self.verbosity, cut_switch=cuts_list, mom_lo = mom_lo, mom_hi = mom_hi)
            
        # Custom prefix for log messages from this processor
        self.print_prefix = "[AnaProcessor] "
        # Module-level logger (kept lightweight)
        try:
            self.logger = Logger(print_prefix=self.print_prefix, verbosity=self.verbosity)
            self.logger.log("Initialised", "info")
        except Exception:
            # Fallback to simple print if Logger isn't available
            if module_logger:
                module_logger.log(f"{self.print_prefix}Initialised", "info")
            else:
                print(f"{self.print_prefix}Initialised")
    
    # ==========================================
    # Define the core processing logic
    # ==========================================
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
            # Handle any errors that occur during processing
            try:
                self.logger.log(f"Error processing {file_name}: {e}", "error")
                self.logger.log(traceback.format_exc(), "max")
            except Exception:
                if module_logger:
                    module_logger.log(f"{self.print_prefix}Error processing {file_name}: {e}", "error")
                    module_logger.log(traceback.format_exc(), "max")
                else:
                    print(f"{self.print_prefix}Error processing {file_name}: {e}")
                    print(traceback.format_exc())
            return None
            
def combine_cut_flows( cut_flow_list):
    """Combine a list of cut flows after multiprocessing 
    
    Args:
        cut_flows: List of cut statistics lists from different files

    Returns:
        list: Combined cut statistics
    """        
    try:
        # Use the first (now filtered) list as template
        template = cut_flow_list[0]
        
        # Use the template to initialise combined stats
        combined_cut_flow = []
        for cut in template:
            # Create a copy (needed?)
            cut_copy = {k: v for k, v in cut.items()}
            # Reset the event count
            cut_copy["events_passing"] = 0
            combined_cut_flow.append(cut_copy)
        
        # Create a mapping of cut names to indices in combined_stats 
        cut_name_to_index = {cut["name"]: i for i, cut in enumerate(combined_cut_flow)}
        
        # Sum up events_passing for each cut across all files
        for cut_flow in cut_flow_list:
            for cut in cut_flow:
                cut_name = cut["name"]
                # Only process cuts that are in our combined_stats
                if cut_name in cut_name_to_index:
                    idx = cut_name_to_index[cut_name]
                    combined_cut_flow[idx]["events_passing"] += cut["events_passing"]
        
        # Recalculate percentages
        if combined_cut_flow and combined_cut_flow[0]["events_passing"] > 0:
            total_events = combined_cut_flow[0]["events_passing"]
            
            for i, cut in enumerate(combined_cut_flow):
                events = cut["events_passing"]
                
                # Absolute percentage
                cut["absolute_frac"] = (events / total_events) * 100.0
                
                # Relative percentage
                if i == 0:  # "No cuts"
                    cut["relative_frac"] = 100.0
                else:
                    prev_events = combined_cut_flow[i-1]["events_passing"]
                    cut["relative_frac"] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0

        cut_manager = CutManager(verbosity=2)
        try:
            # Use module logger if available
            logger = Logger(print_prefix="[combine_cut_flows] ", verbosity=2)
            logger.log("================== Total Cut Flow =======================", "info")
        except Exception:
            print("================== Total Cut Flow =======================")

        cut_manager.print_cut_stats(stats=combined_cut_flow, active_only=True, csv_name="cut_stats.csv")
        return combined_cut_flow
    
    except Exception as e:
        if module_logger:
            module_logger.log(f"Exception when combining cut flows: {e}", "error")
        else:
            print(f"Exception when combining cut flows: {e}")
        raise
        
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

    
    
def categorize_tracks( data, mismatch=False):
    array_tmp = ak.copy(data['trkmc'])

    i_mask = (array_tmp['trkmcsim']['rank'] == 0) & (array_tmp['trkmcsim']['nhits'] > 0)
    for branch in ak.fields(array_tmp):
        for leaf in ak.fields(array_tmp[branch]):
            if array_tmp[branch].layout.minmax_depth[1] > 2:
                mask_vec = ak.broadcast_arrays(array_tmp[branch],i_mask,depth_limit=3)[1]
                array_tmp[branch,leaf] = ak.mask(array_tmp[branch,leaf], mask_vec)
            else:
                array_tmp[branch,leaf] = ak.mask(array_tmp[branch,leaf], i_mask)

    if mismatch:
        pStartCode = ak.max(ak.flatten(array_tmp['trkmcsim']['startCode'],axis=2),axis=1,mask_identity=True)
        pGenCode = ak.max(ak.flatten(array_tmp['trkmcsim']['gen'],axis=2),axis=1,mask_identity=True)

    else:
        pStartCode = ak.flatten(ak.drop_none(array_tmp['trkmcsim']['startCode']),axis=2,mask_identity=True)
        pGenCode = ak.flatten(ak.drop_none(array_tmp['trkmcsim']['gen']),axis=2,mask_identity=True)
    pStartCode = ak.fill_none(pStartCode,-1)
    pGenCode = ak.fill_none(pGenCode,-1)
  
    categories = ak.zeros_like(pStartCode)
    for icat, idict in enumerate(mom_components.values()):
        startCodes = idict['startCode']
        genCodes = idict['genCode']
        goodCode = ak.zeros_like(pStartCode,dtype=bool)
        for startCode in startCodes:
            for genCode in genCodes:
                goodStartCode = ak.ones_like(pStartCode,dtype=bool) if startCode is None else (pStartCode == startCode)
                goodGenCode = ak.ones_like(pGenCode,dtype=bool) if genCode is None else (pGenCode == genCode)
                goodCode = goodCode | (goodStartCode & goodGenCode)
        
        categories = categories + (icat+1) * (goodCode)
    return categories

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
  vector = Vector()

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
    PrintArgs(args)
    # list which cuts to switch on/off (positional):
    # sw(0)=is_reco_electron, sw(1)=has_downstream, sw(2)=good_trkqual, sw(3)=good_trkpid,
    # sw(4)=has_hits, sw(5)=within_t0, sw(6)=within_t0err, sw(7)=within_lhr_max,
    # sw(8)=within_d0, sw(9)=within_pitch_angle, sw(10)=no_crv_veto, sw(11)=has_st,
    # sw(12)=no_opa, sw(13)=in_mom_range

    new = [True, True, True, True, True, True, True, False, False, False, True, True, True, True]
    nocuts = [False] * 14

    # Convert positional list to named switches for robustness
    cut_names = [
        "is_reco_electron",
        "has_downstream",
        "good_trkqual",
        "good_trkpid",
        "has_hits",
        "within_t0",
        "within_t0err",
        "within_lhr_max",
        "within_d0",
        "within_pitch_angle",
        "no_crv_veto",
        "has_st",
        "no_opa",
        "in_mom_range",
    ]

    named_switches = dict(zip(cut_names, new))
    if module_logger:
        module_logger.log(f"selection cuts to be applied : {named_switches}", "info")
    else:
        print("selection cuts to be applied : ", named_switches)
    ana_processor = AnaProcessor(args.file, args.jobs, named_switches, args.loc, args.fitrange_low[0], args.fitrange_hi[0])
    results = ana_processor.execute()

    # Combine arrays and cut statistics
    pre_fit = combine_arrays(results)
    cutlist = []
    for i, result in enumerate(results):
        cutlist.append(result["cut_stats"])
    combine_cutflows = combine_cut_flows(cutlist)

    # Categorize tracks if requested
    if int(args.cat) == 1:
        track_cat = categorize_tracks(pre_fit, args.mismatch)  # just pre-fit here
        track_cat = ak.broadcast_arrays(pre_fit['trkfit']['trksegs', 'time'], track_cat)[1]
    else:
        track_cat = []

    # Run mc_count
    mc_count = count_particle_types(pre_fit)

    # select only track front to fit to
    selector = Select()
    trk_front = selector.select_surface(pre_fit['trkfit'], surface_name='TT_Front')

    trkfit_ent = ak.mask(pre_fit['trkfit']["trksegs"], trk_front)

    if int(args.cat) == 1:
        track_cat = ak.mask(track_cat, trk_front)
        track_cat = ak.flatten(track_cat, axis=None)
    else:
        track_cat = []

    vector = Vector()
    # make vector mag branch
    mom_mag = vector.get_mag(trkfit_ent, 'mom')

    time = ak.nan_to_none(trkfit_ent['time'])  # FIXME
    time = ak.drop_none(time)

    # call the fitter
    # Marker: confirm we reached the fitter dispatch
    if module_logger:
        module_logger.log("=== ENTER FITTER DISPATCH ===", "info")
    else:
        print("=== ENTER FITTER DISPATCH ===")

    # Diagnostic: report fittype and existence of fit functions
    fittype_norm = str(args.fittype).strip().lower()
    try:
        if module_logger:
            module_logger.log(f"Requested fittype: {args.fittype!r} (normalized: {fittype_norm})", "info")
            module_logger.log(f"Unbinned_fit_mom in globals: {'Unbinned_fit_mom' in globals()}", "info")
            module_logger.log(f"Unbinned_fit_time in globals: {'Unbinned_fit_time' in globals()}", "info")
        else:
            print(f"Requested fittype: {args.fittype!r} (normalized: {fittype_norm})")
            print(f"Unbinned_fit_mom in globals: {'Unbinned_fit_mom' in globals()}")
            print(f"Unbinned_fit_time in globals: {'Unbinned_fit_time' in globals()}")
    except Exception:
        print(f"Requested fittype: {args.fittype!r} (normalized: {fittype_norm})")
        print(f"Unbinned_fit_mom in globals: {'Unbinned_fit_mom' in globals()}")
        print(f"Unbinned_fit_time in globals: {'Unbinned_fit_time' in globals()}")

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
    if module_logger:
        module_logger.log(f"mom entries: {n_mom}, time entries: {n_time}", "info")
    else:
        print(f"mom entries: {n_mom}, time entries: {n_time}")

    if fittype_norm.startswith("mom") or fittype_norm == "mom1d":
        fitresult, par, loss, nlls, combine_pdf, constraints = Unbinned_fit_mom(
            mom_mag,
            track_cat,
            mc_count,
            args.fitrange_low[0],
            args.fitrange_hi[0],
            True,
            args.verbose,
        )
        if module_logger:
            module_logger.log(f'Fit result: {fitresult}', 'success')
        else:
            print('[py-fitter/main] ✅  Fit result: ', fitresult, '\n', 'for  fit')

        if int(args.interpret) == 1:
            result_output = ResultsClass(mom_mag, fitresult, args.verbose)
            result_output.WriteFittedData(args.fitrange_low[0], args.fitrange_hi[0])
            result_output.WriteResult()
            result_output.GetSignifcance(par, loss, 'asym')
            if int(args.setlimit) == 1:
                result_output.GetUL(
                    par,
                    loss,
                    nlls,
                    combine_pdf,
                    constraints,
                    args.fitrange_low[0],
                    args.fitrange_hi[0],
                    fitresult.params['N_CE']['value'],
                    0.90,
                    'asym',
                )

    elif args.fittype == "time1D":
        fitresult, par, loss, combine_pdf = Unbinned_fit_time(
            time,
            track_cat,
            mc_count,
            float(args.fitrange_low[1]),
            float(args.fitrange_hi[1]),
            True,
            args.verbose,
        )
        if module_logger:
            module_logger.log(f'Fit result: {fitresult} for {args.fittype}', 'success')
        else:
            print('[py-fitter/main] ✅ Fit result: ', fitresult, '\n', 'for ', args.fittype, ' fit')

    elif args.fittype == "2D":
        fitresult, par, loss, combine_pdf = Unbinned_2d_fit_mom_time(
            mom_mag,
            time,
            track_cat,
            mc_count,
            [args.fitrange_low[0], args.fitrange_hi[0]],
            [args.fitrange_low[1], args.fitrange_hi[1]],
            bool(args.cat),
            args.verbose,
        )
        if module_logger:
            module_logger.log(f'Fit result: {fitresult} for {args.fittype}', 'success')
        else:
            print('[py-fitter/main]✅  Fit result: ', fitresult, '\n', 'for ', args.fittype, ' fit')

    else:
        raise Exception(
            "[py-fitter/main] ❌ ERROR: choice of fit type does not exist, please choose: mom1D, time1D or momtime2D"
        )

        
  
def PrintArgs(args):
    """
    prints users input parameters
    """
    print("========= [py-fitter/main]✅  Analyzing with user opts: ===========")
    if module_logger:
        module_logger.log('Analyzing with user opts', 'info')
        module_logger.log(f'file: {args.file}', 'info')
        module_logger.log(f'location: {args.loc}', 'info')
        module_logger.log(f'number of processes (njobs - optimal is 1 per file): {args.jobs}', 'info')
        module_logger.log(f'fittype: {args.fittype}', 'info')
        module_logger.log(f'range: {args.fitrange_low} {args.fitrange_hi}', 'info')
        module_logger.log(f'categorize: {args.cat}', 'info')
        module_logger.log(f'mismatch: {args.mismatch}', 'info')
        module_logger.log(f'verbose: {args.verbose}', 'info')
        module_logger.log(f'interpret: {args.interpret}', 'info')
        module_logger.log(f'setlimit: {args.setlimit}', 'info')
    else:
        print("========= [py-fitter/main]✅  Analyzing with user opts: ===========")
        print("file:", args.file)
        print("location: ", args.loc)
        print("number of processes (njobs - optimal is 1 per file):", args.jobs)
        print("fittype: ", args.fittype)
        print("range: ", args.fitrange_low, args.fitrange_hi)
        print("categorize: ", args.cat)
        print("mismatch: ", args.mismatch)
        print("verbose: ", args.verbose)
        print("interpret: ", args.interpret)
        print("setlimit: ", args.setlimit)

if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--fittype", type=str, default="mom1D", help="fittype implemented opts: mom1D, time1D, momtime2D")
    parser.add_argument("--fitrange_low", type=float, default=[95,475], nargs='+', help="minimum to fit ordered mom, time")
    parser.add_argument("--fitrange_hi", type=float, default=[110,1650], nargs='+',help="maximum to fit  ordered mom, time")
    parser.add_argument("--interpret", type=int, default=0, help="allows for significance evaluation")
    parser.add_argument("--setlimit", type=int, default=0, help="assumes low signal and will try to set limit")
    parser.add_argument("--cat", type=int, default=0, help="Categorize tracks by MC matching")
    parser.add_argument("--mismatch", type=int, default=0, help="This is an old sample with MC - reco trk mismatch")
    parser.add_argument("--verbose", default=1, help="verbose")
    parser.add_argument("--loc", type=str, required=False, default='disk', help="location of files")
    args = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)




