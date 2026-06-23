"""
Build and execute fits on scaled MC component samples.

This module enables fitting to highest-stats scaled MC samples instead of single
combined datasets. It reuses the existing fitting, PDF building, and plotting
infrastructure while handling the scaling and component preparation.

Key Features:
- Loads separate MC component files with independent cuts
- Dynamically scales components by expected yields within the explicit fit range
- Uses random downsampling to preserve true high-statistics shapes without artifacts
- Extracts fit variables while preserving component information
- Integrates seamlessly with existing Unbinned_fit_mom and Unbinned_2d_fit_mom_time
"""

import argparse
import numpy as np
import awkward as ak
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, List, Any
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import zfit
import tensorflow as tf
from PIL import Image

# Import existing custom framework tools
from pyutils.pylogger import Logger
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pycut import CutManager
from process import AnaProcessor
from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
from data_prep import DataPreparationManager
from style import FONTS, COLORS
from model.physics_components import mom_components
from config import GLOBAL_VERBOSITY

# Map component names to their MC truth process codes (from process.py)
COMPONENT_TO_MC_CODE = {
    'dio': 166,           # DIO process
    'ipa': 0,             # Inclusive pion absorption
    'ce': 168,            # Charged lepton exchange (CEM)
    'cem': 168,            # CEM variant
    'cep': 176,            # CE+ variant
    'cosmic': -1,         # Cosmic-induced
    'rpc_ext': 178,       # External RPC
    'rpc_int': 179,       # Internal RPC
    'rmc_ext': 172,       # External rare muon capture
    'rmc_int': 171,       # Internal rare muon capture
}

# Display colors and names for components
COMPONENT_DISPLAY = {
    'dio': {'color': '#e377c2', 'label': 'DIO'},
    'cosmic': {'color': '#1f77b4', 'label': 'Cosmic'},
    'rpc_ext': {'color': '#2ca02c', 'label': 'RPC (ext)'},
    'rpc_int': {'color': '#2ca02c', 'label': 'RPC (int)'},
    'rmc_ext': {'color': '#d62728', 'label': 'RMC (ext)'},
    'rmc_int': {'color': '#9467bd', 'label': 'RMC (int)'},
    'ipa': {'color': '#8c564b', 'label': 'IPA'},
    'ce': {'color': '#ff8000', 'label': 'Signal (CE)'},
    'cem': {'color': '#ff8000', 'label': 'Signal (CE)'},
    'cep': {'color': '#ff7f0e', 'label': 'Signal (CE+)'},
}


def plot_fit_with_true_shapes(mom_mag, combine_pdf=None, fit_result=None, component_data_dict=None, component_names=None, 
                              scale_factors_dict=None, fit_range=(97, 110), nbins=25, output_file=None,
                              title="Fit with Scaled MC Components", verbosity=1, real_data_mom_mag=None,
                              plot_obs='mom'):
    """Plot fit results with pre-computed true MC component histograms and pull plot."""
    logger = Logger(print_prefix="[plot_fit_with_true_shapes]", verbosity=verbosity)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.05)
    
    bin_edges = np.linspace(fit_range[0], fit_range[1], nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (fit_range[1] - fit_range[0]) / nbins
    
    component_hists = {}
    logger.log("="*70, "info")
    logger.log("Plotting component histograms", "info")
    logger.log("="*70, "info")
    
    for comp_name in component_names:
        if comp_name in component_data_dict:
            data = component_data_dict[comp_name]
            if data is not None and len(data) > 0:
                counts, _ = np.histogram(data, bins=bin_edges)
                
                logger.log(f"{comp_name}:", "info")
                logger.log(f"  Events in window: {len(data)}", "info")
                logger.log(f"  Data range: {np.min(data):.2f} - {np.max(data):.2f}", "info")
                logger.log(f"  Histogram counts: {np.sum(counts):.0f}", "info")
                
                component_hists[comp_name] = counts
            else:
                logger.log(f"{comp_name}: NO DATA", "warning")
                component_hists[comp_name] = np.zeros(nbins)
        else:
            logger.log(f"{comp_name}: NOT IN DICT", "warning")
            component_hists[comp_name] = np.zeros(nbins)
    
    logger.log("="*70, "info")
    
    bottom = np.zeros(nbins)
    desired_order = ['rpc_ext', 'rpc_int','cosmic', 'dio',  'rmc_ext', 'rmc_int', 'ipa', 'ce']
    
    for comp_name in desired_order:
        if comp_name in component_hists and np.sum(component_hists[comp_name]) > 0:
            display_info = COMPONENT_DISPLAY.get(comp_name, {'color': 'C0', 'label': comp_name})
            ax1.bar(bin_centers, component_hists[comp_name], width=bin_width, 
                   bottom=bottom, label=display_info['label'], 
                   color=display_info['color'], alpha=0.8)
            bottom += component_hists[comp_name]
    
    total_hist = bottom.copy()
    
    if real_data_mom_mag is not None and len(real_data_mom_mag) > 0:
        try:
            data_counts, _ = np.histogram(real_data_mom_mag, bins=bin_edges)
            data_errors = np.sqrt(data_counts)
            ax1.errorbar(bin_centers, data_counts, yerr=data_errors, 
                        fmt='ko', markersize=6, capsize=3, capthick=1.5,
                        elinewidth=1.5, label='Data (MDS)', zorder=5)
            logger.log(f"Overlaid real data: {len(real_data_mom_mag)} events", "info")
        except Exception as e:
            logger.log(f"Could not overlay real data: {e}", "warning")
    
    fit_vals_for_plot = None
    if combine_pdf is not None:
        try:
            logger.log("Attempting to plot fit model curve", "debug")
            mom_vals = np.linspace(fit_range[0], fit_range[1], 200)
            pdf_obs = combine_pdf.obs

            if isinstance(pdf_obs, tuple):
                n_obs = len(pdf_obs)
            elif isinstance(pdf_obs, (list, set)):
                n_obs = len(list(pdf_obs))
            else:
                n_obs = 1

            if n_obs == 1:
                obs_space = zfit.Space(pdf_obs, limits=fit_range)
                eval_tensor = tf.constant(mom_vals, dtype=tf.float32)
            else:
                obs_space = combine_pdf.space
                obs_names = list(pdf_obs)
                scan_idx = 1 if plot_obs == 'time' and len(obs_names) > 1 else 0
                eval_tensor = None

                limits_low, limits_high = obs_space.rect_limits
                lo_arr = np.asarray(limits_low, dtype=float).reshape(-1)
                hi_arr = np.asarray(limits_high, dtype=float).reshape(-1)

                def _project_pdf_1d(pdf_obj):
                    if n_obs != 2:
                        fixed_val = 0.5 * (lo_arr[0] + hi_arr[0])
                        eval_points = np.zeros((len(mom_vals), n_obs), dtype=np.float32)
                        eval_points[:, scan_idx] = mom_vals.astype(np.float32)
                        for j in range(n_obs):
                            if j != scan_idx:
                                eval_points[:, j] = fixed_val
                        vals = pdf_obj.pdf(tf.constant(eval_points, dtype=tf.float32), norm_range=obs_space)
                        if hasattr(vals, 'numpy'):
                            vals = vals.numpy()
                        return np.array(vals)

                    other_idx = 1 - scan_idx
                    other_grid = np.linspace(lo_arr[other_idx], hi_arr[other_idx], 120)

                    scan_rep = np.tile(mom_vals, len(other_grid))
                    other_rep = np.repeat(other_grid, len(mom_vals))
                    pts = np.zeros((len(scan_rep), 2), dtype=np.float32)
                    pts[:, scan_idx] = scan_rep.astype(np.float32)
                    pts[:, other_idx] = other_rep.astype(np.float32)

                    vals = pdf_obj.pdf(tf.constant(pts, dtype=tf.float32), norm_range=obs_space)
                    if hasattr(vals, 'numpy'):
                        vals = vals.numpy()
                    vals = np.array(vals).reshape(len(other_grid), len(mom_vals))
                    return np.trapz(vals, other_grid, axis=0)

            if n_obs == 1:
                pdf_vals = combine_pdf.pdf(eval_tensor, norm_range=obs_space)
            else:
                pdf_vals = _project_pdf_1d(combine_pdf)
            
            if hasattr(pdf_vals, 'numpy'):
                pdf_vals = pdf_vals.numpy()
            else:
                pdf_vals = np.array(pdf_vals)
            
            area = np.trapz(pdf_vals, mom_vals)
            if area > 0:
                pdf_vals = pdf_vals / area

            total_events = len(mom_mag)
            scaled_pdf = pdf_vals * total_events * bin_width
            fit_vals_for_plot = (mom_vals, scaled_pdf)
            
            ax1.plot(mom_vals, scaled_pdf, 'k-', linewidth=2.5, label='Total Fit', zorder=10)
            
            if hasattr(combine_pdf, 'pdfs') and fit_result is not None:
                try:
                    fitline_colors = {'CE': 'red', 'DIO': 'green', 'Cosmic': 'cyan', 'RPC': '#ff7f0e'}
                    component_names_list = ['CE', 'Cosmic', 'RPC', 'DIO']
                    pdfs_list = combine_pdf.pdfs
                    
                    fitted_yields = {}
                    if hasattr(fit_result, 'params'):
                        for param in fit_result.params:
                            param_name = str(param.name) if hasattr(param, 'name') else str(param)
                            if param_name.startswith('N_'):
                                comp = param_name.replace('N_', '')
                                if hasattr(param, 'numpy'):
                                    fitted_yields[comp] = float(param.numpy())
                                elif hasattr(param, 'value'):
                                    fitted_yields[comp] = float(param.value())
                                else:
                                    fitted_yields[comp] = float(param)
                    
                    for i, pdf in enumerate(pdfs_list):
                        try:
                            comp_name = getattr(pdf, 'name', None)
                            if comp_name is None or comp_name not in fitted_yields:
                                if i < len(component_names_list):
                                    comp_name = component_names_list[i]
                            
                            if comp_name in fitted_yields:
                                yield_val = fitted_yields[comp_name]
                            else:
                                continue
                            
                            if n_obs == 1:
                                comp_pdf_vals = pdf.pdf(eval_tensor, norm_range=obs_space)
                                if hasattr(comp_pdf_vals, 'numpy'):
                                    comp_pdf_vals = comp_pdf_vals.numpy()
                                else:
                                    comp_pdf_vals = np.array(comp_pdf_vals)
                            else:
                                comp_pdf_vals = _project_pdf_1d(pdf)

                            comp_area = np.trapz(comp_pdf_vals, mom_vals)
                            if comp_area > 0:
                                comp_pdf_vals = comp_pdf_vals / comp_area
                            
                            scaled_comp_pdf = comp_pdf_vals * yield_val * bin_width
                            comp_color = fitline_colors.get(comp_name, f'C{i}')
                            
                            ax1.plot(mom_vals, scaled_comp_pdf, '--', linewidth=1.5, 
                                label=comp_name, color=comp_color, alpha=0.95, zorder=12)
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception as e:
            logger.log(f"Could not plot fit model: {type(e).__name__}: {e}", "warning")
    
    ax1.set_ylabel('Events', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1, ymax=1500)
    ax1.legend(loc='upper right', fontsize=FONTS['legend']['size'])
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax1.set_xlim(fit_range)
    
    if fit_vals_for_plot is not None:
        mom_vals, scaled_pdf = fit_vals_for_plot
        fit_at_bins = np.interp(bin_centers, mom_vals, scaled_pdf)
        
        err = []
        dev = []
        for i, data_val in enumerate(total_hist):
            fit_val = fit_at_bins[i]
            if data_val > 0:
                data_err = np.sqrt(data_val)
                fit_err = np.sqrt(fit_val)
                total_err = np.sqrt(data_err**2 + fit_err**2)
                if total_err > 0:
                    err.append(total_err / data_val if data_val > 0 else 0)
                    dev.append((fit_val - data_val) / total_err)
                else:
                    err.append(0); dev.append(0)
            else:
                err.append(0); dev.append(0)
        
        ax2.errorbar(bin_centers, dev, yerr=err, color='None', marker='+', 
                    markerfacecolor='black', ecolor='black', capsize=3, markersize=8)
        ax2.axhline(0, color='red', linewidth=1.5)
        ax2.set_ylabel(r'Pull [$\sigma$]', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
        ax2.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
        ax2.yaxis.set_ticks(np.arange(-5, 6, 2))
        ax2.set_xlim(fit_range)
        ax2.tick_params(axis='both', labelsize=FONTS['tick']['size'])
    else:
        ax2.axis('off')
    
    try:
        logo = Image.open("mu2e_logo_oval.png")
        ax_logo = fig.add_axes([0.02, 0.93, 0.1, 0.09])
        ax_logo.imshow(logo)
        ax_logo.axis('off')
    except Exception:
        pass
    
    fig.text(0.15, 0.98, "Mu2e Simulation (Preliminary - Summer 2026)", 
            fontsize=FONTS['label']['size'], fontweight='bold', ha='left', va='top', zorder=100)
    
    ax1.grid(False)
    if fit_vals_for_plot is not None:
        ax2.grid(False)
    
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.log(f"Saved plot to {output_file}", "info")
    
    return fig


class ScaledFitBuilder:
    """Build and execute fits on scaled MC component samples."""
    
    def __init__(self, verbosity=1, jobs=1):
        self.logger = Logger(print_prefix="[ScaledFitBuilder]", verbosity=verbosity)
        self.verbosity = verbosity
        self.jobs = jobs
        self.components = {}  
        self.scaled_components = {}  
        self.component_yields = {}  
        self.data = None  
        self.fit_range_lo = 97  
        self.fit_range_hi = 110  
        
        self.default_yields = { 
            'dio': 1427,           
            'cosmic': 333,         
            'rpc_ext': 5,          
            'rpc_int': 4,          
            'rmc_ext': None,       
            'rmc_int': None,       
            'ipa': None,           
            'ce': 73               
        }
        self.component_yields = self.default_yields.copy()
    
    def set_component_yields(self, yields_dict: Dict[str, Optional[float]]):
        self.component_yields.update(yields_dict)
        self.logger.log(f"Set component yields: {self.component_yields}", "info")

    def _save_scan_npz(self, out_path: str, mom_mag: np.ndarray, times: Optional[np.ndarray] = None, categories: Optional[np.ndarray] = None):
        try:
            payload = {'mom_mag': np.asarray(mom_mag)}
            if times is not None:
                payload['time'] = np.asarray(times)
            if categories is not None:
                payload['categories'] = np.asarray(categories)
            np.savez_compressed(out_path, **payload)
            self.logger.log(f"Saved component tracking matrix to: {out_path}", "info")
        except Exception as e:
            self.logger.log(f"Failed to save scan matrix: {e}", "warning")
    
    def _process_file_list(self, file_path: str, sign: str = "minus", 
                          location: str = "disk", component_name: str = "unknown",
                          mom_lo: Optional[float] = None, mom_hi: Optional[float] = None) -> Optional[ak.Array]:
        if mom_lo is None:
            mom_lo = self.fit_range_lo
        if mom_hi is None:
            mom_hi = self.fit_range_hi
        
        cut_switches = [
            True, True, True, True, True, True, True, False, False, False, True, 
            True, True, True, True, True, True, True, False, False, False, False
        ]
        
        try:
            processor = AnaProcessor(
                file_list_path=file_path, jobs=self.jobs, cuts=cut_switches,
                location=location, mom_lo=mom_lo, mom_hi=mom_hi
            )
            results_list = processor.execute()
            if not results_list:
                return None
            
            data_list = []
            for result in results_list:
                if result is not None:
                    if isinstance(result, dict) and "filtered_data" in result:
                        data_list.append(result["filtered_data"])
                    else:
                        data_list.append(result)
            
            if data_list:
                return ak.concatenate(data_list)
            return None
        except Exception as e:
            self.logger.log(f"Error processing file list {file_path}: {e}", "error")
            return None
    
    def load_and_scale_components(self, component_files: Dict[str, str], 
                                  sign: str = "minus", 
                                  location: str = "disk",
                                  normalize_to_max: bool = False,
                                  apply_cuts: bool = True,
                                  fit_range: Optional[Tuple[float, float]] = None):
        self.components = {}
        self.scaled_components = {}
        
        if fit_range is not None:
            self.fit_range_lo = fit_range[0]
            self.fit_range_hi = fit_range[1]
            self.logger.log(f"Set momentum range for loading cuts: {self.fit_range_lo} - {self.fit_range_hi} MeV/c", "info")
        
        for component_name, file_path in component_files.items():
            if file_path is None or not Path(file_path).exists():
                continue
            
            if apply_cuts:
                data = self._process_file_list(file_path, sign=sign, location=location, 
                                            component_name=component_name,
                                            mom_lo=self.fit_range_lo, mom_hi=self.fit_range_hi)
            else:
                try:
                    processor = AnaProcessor(
                        file_list_path=file_path, jobs=self.jobs, cuts=[False] * 22,
                        location=location, mom_lo=self.fit_range_lo, mom_hi=self.fit_range_hi
                    )
                    results_list = processor.execute()
                    if results_list:
                        from process import combine_arrays
                        data = combine_arrays(results_list)
                    else:
                        data = None
                except Exception:
                    data = None
            
            if data is not None:
                self.components[component_name] = data
                self.scaled_components[component_name] = (data, 1.0)
                self.logger.log(f"Loaded high-stats component '{component_name}': {len(data)} events", "info")
    
    def load_real_data(self, data_file: str, sign: str = "minus", location: str = "local") -> bool:
        self.data = self._process_file_list(data_file, sign=sign, location=location, 
                                           component_name="real_data",
                                           mom_lo=self.fit_range_lo, mom_hi=self.fit_range_hi)
        return self.data is not None
    
    def _extract_variable(self, data: ak.Array, var_name: str) -> Optional[np.ndarray]:
        try:
            if var_name.lower() == "recomom_ttfront":
                selector = Select(verbosity=0)
                vector = Vector()
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                mom_mag = vector.get_mag(trkfit_ent, 'mom')
                mom_mag = ak.drop_none(mom_mag)
                return np.array(ak.flatten(mom_mag, axis=None))
            elif var_name.lower() == "recomom_mc_ttfront":
                selector = Select(verbosity=0)
                vector = Vector()
                trk_front_mc = selector.select_surface(data['trkfit'], surface_name="TT_Front", branch_name="trksegsmc")
                trkfit_ent_mc = ak.mask(data['trkfit']["trksegsmc"], trk_front_mc)
                mom_mag_mc = vector.get_mag(trkfit_ent_mc, 'mom')
                mom_mag_mc = ak.drop_none(mom_mag_mc)
                return np.array(ak.flatten(mom_mag_mc, axis=None))
            elif var_name.lower() == "tracktime_ttfront":
                selector = Select(verbosity=0)
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                time = DataPreparationManager.clean_awkward_array(trkfit_ent['time'])
                return np.array(ak.flatten(time, axis=None))
            else:
                parts = var_name.split('.')
                val = data
                for part in parts:
                    val = val[part]
                val = ak.drop_none(val)
                return np.array(ak.flatten(val, axis=None))
        except Exception as e:
            self.logger.log(f"Error extracting '{var_name}': {e}", "error")
            return None

    def _build_combined_data(self, variables: Dict[str, List[np.ndarray]], 
                            scale_factors: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Combine lists of component arrays using Random Downsampling to preserve high-stats shape profiles."""
        combined = {}
        combined_component_cats = []
        
        comp_names = list(self.scaled_components.keys())
        
        # Initialize a deterministic random state for reproducible pseudo-experiments
        rng = np.random.default_rng(seed=42)
        
        for var_name, comp_list in variables.items():
            combined_var_elements = []
            
            for i, comp_name in enumerate(comp_names):
                array = comp_list[i]
                if array is not None and len(array) > 0:
                    sf = scale_factors.get(comp_name, 1.0)
                    n_target = int(np.round(len(array) * sf))
                    
                    if n_target == 0:
                        continue
                    
                    # Core fix: Use random downsampling to protect shape distributions from discretization artifacts
                    if n_target <= len(array):
                        # High-statistics sample -> downsample smoothly without any duplication
                        indices = rng.choice(len(array), size=n_target, replace=False)
                    else:
                        # Sparse tracking data -> sample with replacement distributed uniformly across phase space
                        indices = rng.choice(len(array), size=n_target, replace=True)
                    
                    # Sort indices to guarantee multi-variable phase space synchronization remains matched
                    indices = np.sort(indices)
                    sampled_array = array[indices]
                    
                    combined_var_elements.append(sampled_array)
                    
                    if var_name == list(variables.keys())[0]:
                        mc_code = COMPONENT_TO_MC_CODE.get(comp_name.lower(), -2)
                        cat_array = np.full_like(sampled_array, fill_value=mc_code, dtype=np.int32)
                        combined_component_cats.append(cat_array)
            
            if combined_var_elements:
                combined[var_name] = np.concatenate(combined_var_elements)
            else:
                combined[var_name] = np.array([])
                
        if combined_component_cats:
            combined_component_cats = np.concatenate(combined_component_cats)
        else:
            combined_component_cats = np.array([], dtype=np.int32)
            
        # --- DIAGNOSTIC TRUE YIELD PRINT BLOCK ---
        self.logger.log("="*70, "info")
        self.logger.log("EXACT TRUE YIELDS SENT TO ZFIT (AFTER RANDOM SAMPLING & ALIGNMENT):", "info")
        self.logger.log("="*70, "info")
        
        mc_code_to_name = {v: k for k, v in COMPONENT_TO_MC_CODE.items()}
        unique_codes, final_counts = np.unique(combined_component_cats, return_counts=True)
        for code, count in zip(unique_codes, final_counts):
            comp_name = mc_code_to_name.get(code, f"Unknown ({code})")
            target_yield = self.component_yields.get(comp_name.lower(), "N/A")
            self.logger.log(f"  {comp_name.upper():<10} -> Exact Yield: {count:<6} (Target asked for: {target_yield})", "info")
            
        self.logger.log("="*70, "info")
        
        return combined, combined_component_cats
    
    def fit_mom_1d(self, 
                   variable: str = 'recomom_ttfront',
                   fit_range: Tuple[float, float] = (97, 110),
                   plot: bool = True,
                   minos: bool = False,
                   use_constraints: bool = False,
                   constraints_dir: str = 'uncertainties/outputs') -> Any:
        if not self.scaled_components:
            self.logger.log("No scaled components loaded.", "error")
            return None
            
        self.logger.log("Building clean unbinned sample via Window-Accurate Slicing", "info")
        comp_names = list(self.scaled_components.keys())
        
        variables = {variable: []}
        scale_factors = {}
        
        for comp_name in comp_names:
            data, _ = self.scaled_components[comp_name]
            
            self.logger.log(f"\nComponent: {comp_name}", "info")
            self.logger.log(f"  Total raw tracks loaded: {len(data)}", "info")
            
            var_array = self._extract_variable(data, variable)
            
            if var_array is None or len(var_array) == 0:
                variables[variable].append(None)
                scale_factors[comp_name] = 1.0
            else:
                # Isolate entries that exclusively fall inside the active window
                in_range = np.isfinite(var_array) & (var_array >= fit_range[0]) & (var_array <= fit_range[1])
                filtered_mom = var_array[in_range]
                
                # DYNAMIC LOCAL SCALE FACTOR
                target_yield = self.component_yields.get(comp_name.lower())
                if target_yield is not None and len(filtered_mom) > 0:
                    local_scale_factor = target_yield / len(filtered_mom)
                    self.logger.log(f"  ✓ Extracted {len(filtered_mom)} window events. Scale Factor: {local_scale_factor:.5f} -> Target: {target_yield}", "info")
                else:
                    local_scale_factor = 1.0
                
                variables[variable].append(filtered_mom)
                scale_factors[comp_name] = local_scale_factor
                
        # Build pseudo-dataset using local window scales and random sampling
        combined_data, component_cats = self._build_combined_data(variables, scale_factors)
        fit_data_mom = combined_data[variable]
        
        if len(fit_data_mom) == 0:
            self.logger.log("Zero total entries remained across components.", "error")
            return None
            
        self._save_scan_npz("scaled_fit_mom_mag.npz", fit_data_mom, categories=component_cats)
        constraints_dir_to_use = constraints_dir if use_constraints else None
        
        fit_tuple = Unbinned_fit_mom(
            mom_mag=fit_data_mom,
            count_particle_types=component_cats,
            fit_range_low=fit_range[0],
            fit_range_hi=fit_range[1],
            plot_truth=False,
            verbose=self.verbosity,
            minos=minos,
            plot_NLL=True,
            plot_results=True,
            constraints_dir=constraints_dir_to_use
        )
        
        result, poi, loss, aux_nlls, combine_pdf, constraints = fit_tuple
        
        if plot:
            try:
                # Reconstruct localized dictionary shapes for truth curves overlaying plotting module
                plot_data_dict = {}
                for comp_name in comp_names:
                    idx = comp_names.index(comp_name)
                    arr = variables[variable][idx]
                    if arr is not None and len(arr) > 0:
                        sf = scale_factors.get(comp_name, 1.0)
                        nt = int(np.round(len(arr) * sf))
                        if nt > 0:
                            # Use exact identical sampling logic so plots reflect exact fitter conditions
                            rng_plot = np.random.default_rng(seed=42)
                            if nt <= len(arr):
                                plot_data_dict[comp_name] = rng_plot.choice(arr, size=nt, replace=False)
                            else:
                                plot_data_dict[comp_name] = rng_plot.choice(arr, size=nt, replace=True)
                
                real_data_mom = None
                if self.data is not None:
                    real_data_mom = self._extract_variable(self.data, variable)
                    if real_data_mom is not None and len(real_data_mom) > 0:
                        real_data_mom = real_data_mom[(real_data_mom >= fit_range[0]) & (real_data_mom <= fit_range[1])]
                
                plot_fit_with_true_shapes(
                    mom_mag=fit_data_mom, combine_pdf=combine_pdf, fit_result=result,
                    component_data_dict=plot_data_dict, component_names=comp_names,
                    scale_factors_dict=scale_factors, fit_range=fit_range, nbins=25,
                    output_file="fit_momentum_scaled_components.png", title="1D Fit: Downsampled MC",
                    verbosity=self.verbosity, real_data_mom_mag=real_data_mom
                )
            except Exception as e:
                self.logger.log(f"Warning: plot generation failed: {e}", "warning")
        
        return result
    
    def fit_mom_time_2d(self,
                       mom_variable: str = 'recomom_ttfront',
                       time_variable: str = 'tracktime_ttfront',
                       fit_range_mom: Tuple[float, float] = (97, 110),
                       fit_range_time: Tuple[float, float] = (475, 1650),
                       plot: bool = True,
                       use_constraints: bool = False,
                       constraints_dir: str = 'uncertainties/outputs') -> Any:
        if not self.scaled_components:
            self.logger.log("No scaled components loaded.", "error")
            return None
        
        self.logger.log("Building clean 2D unbinned sample via Window-Accurate Slicing", "info")
        comp_names = list(self.scaled_components.keys())
        
        # Use internal keys ('mom', 'time') to resolve key tracking issues
        variables = {'mom': [], 'time': []}
        scale_factors = {}
        
        for comp_name in comp_names:
            data, _ = self.scaled_components[comp_name]
            self.logger.log(f"\nComponent: {comp_name}", "info")
            
            mom_array = self._extract_variable(data, mom_variable)
            time_array = self._extract_variable(data, time_variable)
            
            if (mom_array is None or len(mom_array) == 0 or 
                time_array is None or len(time_array) == 0):
                variables['mom'].append(None)
                variables['time'].append(None)
                scale_factors[comp_name] = 1.0
            else:
                if len(mom_array) != len(time_array):
                    n_min = min(len(mom_array), len(time_array))
                    mom_array = mom_array[:n_min]
                    time_array = time_array[:n_min]

                # Apply explicit 2D filter boundaries
                valid_mask = np.isfinite(mom_array) & np.isfinite(time_array)
                range_mask = (
                    (mom_array >= fit_range_mom[0]) & (mom_array <= fit_range_mom[1]) &
                    (time_array >= fit_range_time[0]) & (time_array <= fit_range_time[1])
                )
                combined_mask = valid_mask & range_mask
                filtered_mom = mom_array[combined_mask]
                filtered_time = time_array[combined_mask]
                
                # DYNAMIC LOCAL SCALE FACTOR
                target_yield = self.component_yields.get(comp_name.lower())
                if target_yield is not None and len(filtered_mom) > 0:
                    local_scale_factor = target_yield / len(filtered_mom)
                    self.logger.log(f"  ✓ Extracted {len(filtered_mom)} synchronized 2D entries. Scale Factor: {local_scale_factor:.5f} -> Target: {target_yield}", "info")
                else:
                    local_scale_factor = 1.0
                
                variables['mom'].append(filtered_mom)
                variables['time'].append(filtered_time)
                scale_factors[comp_name] = local_scale_factor
        
        combined_data, component_cats = self._build_combined_data(variables, scale_factors)
        fit_data_mom = combined_data['mom']
        fit_data_time = combined_data['time']
        
        if len(fit_data_mom) == 0:
            self.logger.log("Zero total components remained after 2D range cuts.", "error")
            return None
            
        self._save_scan_npz("scaled_fit_mom_time.npz", fit_data_mom, fit_data_time, categories=component_cats)
        constraints_dir_to_use = constraints_dir if use_constraints else None
        
        fit_tuple = Unbinned_2d_fit_mom_time(
            mom_mag=fit_data_mom,
            times=fit_data_time,
            count_particle_types=component_cats,
            fit_range_mom=fit_range_mom,
            fit_range_time=fit_range_time,
            plot_truth=False,
            verbose=self.verbosity,
            plot_NLL=False,
            plot_results=False,
            constraints_dir=constraints_dir_to_use
        )
        
        result, poi, loss, combine_pdf, norms = fit_tuple
        
        if plot:
            try:
                # Reconstruct synchronized plotting tracking matrices via matching random states
                plot_mom_dict = {}
                plot_time_dict = {}
                for comp_name in comp_names:
                    idx = comp_names.index(comp_name)
                    m_arr = variables['mom'][idx]
                    t_arr = variables['time'][idx]
                    
                    if m_arr is not None and len(m_arr) > 0:
                        sf = scale_factors.get(comp_name, 1.0)
                        nt = int(np.round(len(m_arr) * sf))
                        
                        if nt > 0:
                            rng_plot = np.random.default_rng(seed=42)
                            if nt <= len(m_arr):
                                chosen_idx = rng_plot.choice(len(m_arr), size=nt, replace=False)
                            else:
                                chosen_idx = rng_plot.choice(len(m_arr), size=nt, replace=True)
                            
                            chosen_idx = np.sort(chosen_idx)
                            plot_mom_dict[comp_name] = m_arr[chosen_idx]
                            plot_time_dict[comp_name] = t_arr[chosen_idx]
                
                real_data_mom = None; real_data_time = None
                if self.data is not None:
                    real_data_mom = self._extract_variable(self.data, mom_variable)
                    real_data_time = self._extract_variable(self.data, time_variable)
                    if (real_data_mom is not None and real_data_time is not None):
                        n_min_data = min(len(real_data_mom), len(real_data_time))
                        real_data_mom = real_data_mom[:n_min_data]; real_data_time = real_data_time[:n_min_data]
                        data_mask = (
                            (real_data_mom >= fit_range_mom[0]) & (real_data_mom <= fit_range_mom[1]) &
                            (real_data_time >= fit_range_time[0]) & (real_data_time <= fit_range_time[1])
                        )
                        real_data_mom = real_data_mom[data_mask]
                        real_data_time = real_data_time[data_mask]
                
                plot_fit_with_true_shapes(
                    mom_mag=fit_data_mom, combine_pdf=combine_pdf, fit_result=result,
                    component_data_dict=plot_mom_dict, component_names=comp_names,
                    scale_factors_dict=scale_factors, fit_range=fit_range_mom, nbins=25,
                    output_file="fit_momentum_2d_scaled_components.png", title="2D Fit: Momentum Projection",
                    verbosity=self.verbosity, real_data_mom_mag=real_data_mom, plot_obs='mom'
                )
                
                plot_fit_with_true_shapes(
                    mom_mag=fit_data_time, combine_pdf=combine_pdf, fit_result=result,
                    component_data_dict=plot_time_dict, component_names=comp_names,
                    scale_factors_dict=scale_factors, fit_range=fit_range_time, nbins=30,
                    output_file="fit_time_2d_scaled_components.png", title="2D Fit: Time Projection",
                    verbosity=self.verbosity, real_data_mom_mag=real_data_time, plot_obs='time'
                )
            except Exception as e:
                self.logger.log(f"Warning: projection plots failed: {e}", "warning")
        
        self.logger.log("2D momentum+time fit completed", "info")
        return result


def main(args):
    logger = Logger(print_prefix="[scaled_fit_builder.main]", verbosity=1)
    builder = ScaledFitBuilder(verbosity=args.verbosity, jobs=args.jobs)
    
    component_files = {}
    if args.components:
        for item in args.components:
            name, path = item.split('=')
            component_files[name] = path
            
    if args.yields:
        yields_dict = {}
        for item in args.yields:
            name, value = item.split('=')
            yields_dict[name] = float(value) if value.lower() != 'none' else None
        builder.set_component_yields(yields_dict)
    
    builder.load_and_scale_components(
        component_files, sign=args.sign, location=args.location, 
        normalize_to_max=args.normalize_to_max, fit_range=(args.fit_range_lo, args.fit_range_hi)
    )
    
    if args.data:
        builder.load_real_data(args.data, sign=args.sign, location=args.location)
    
    if args.fit_type == '1d' or args.fit_type == 'both':
        logger.log("Running 1D momentum fit", "info")
        builder.fit_mom_1d(
            variable=args.variable, fit_range=(args.fit_range_lo, args.fit_range_hi),
            plot=not args.no_plot, minos=args.minos, constraints_dir=args.constraints_dir
        )
    
    if args.fit_type == '2d' or args.fit_type == 'both':
        logger.log("Running 2D momentum+time fit", "info")
        builder.fit_mom_time_2d(
            mom_variable=args.variable, time_variable=args.time_variable,
            fit_range_mom=(args.fit_range_lo, args.fit_range_hi),
            fit_range_time=(args.time_range_lo, args.time_range_hi),
            plot=not args.no_plot, constraints_dir=args.constraints_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit scaled MC component samples using local window-accurate random downsampling",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--fit-type', choices=['1d', '2d', 'both'], default='1d')
    parser.add_argument('--components', nargs='+', required=True)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--variable', type=str, default='recomom_ttfront')
    parser.add_argument('--time-variable', type=str, default='tracktime_ttfront')
    parser.add_argument('--fit-range', type=float, nargs=2, default=[97, 110], dest='fit_range', metavar=('LO', 'HI'))
    parser.add_argument('--fit-range-lo', type=float, default=97, dest='fit_range_lo')
    parser.add_argument('--fit-range-hi', type=float, default=110, dest='fit_range_hi')
    parser.add_argument('--time-range', type=float, nargs=2, default=[475, 1650], dest='time_range', metavar=('LO', 'HI'))
    parser.add_argument('--time-range-lo', type=float, default=475, dest='time_range_lo')
    parser.add_argument('--time-range-hi', type=float, default=1650, dest='time_range_hi')
    parser.add_argument('--yields', nargs='+', default=None)
    parser.add_argument('--normalize-to-max', action='store_true')
    parser.add_argument('--sign', choices=['minus', 'plus'], default='minus')
    parser.add_argument('--location', choices=['disk', 'local'], default='disk')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--constraints-dir', type=str, default='uncertainties/outputs')
    parser.add_argument('--minos', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--verbosity', type=int, default=1)
    
    args = parser.parse_args()
    
    if args.fit_range is not None:
        args.fit_range_lo = args.fit_range[0]
        args.fit_range_hi = args.fit_range[1]
    if args.time_range is not None:
        args.time_range_lo = args.time_range[0]
        args.time_range_hi = args.time_range[1]
    
    main(args)