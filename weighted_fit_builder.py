"""
Build and execute fits on scaled MC component samples using continuous event weights.
Optimized for 2D Unbinned Maximum Likelihood Fits with robust dynamic model interrogation.
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

# Standardized MC truth process codes
COMPONENT_TO_MC_CODE = {
    'dio': 166,
    'ipa': 0,
    'ce': 168,
    'cem': 168,
    'cep': 176,
    'cosmic': -1,
    'rpc': 178,
    'rmc_ext': 172,
    'rmc_int': 171,
}

# Reverse mapping to identify names from process codes during validation
MC_CODE_TO_COMPONENT = {v: k.upper() for k, v in COMPONENT_TO_MC_CODE.items()}

COMPONENT_DISPLAY = {
    'dio': {'color': '#e377c2', 'label': 'DIO'},
    'cosmic': {'color': '#1f77b4', 'label': 'Cosmic'},
    'rpc': {'color': '#2ca02c', 'label': 'RPC (combined)'},
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
                              plot_obs='mom', target_yields_dict=None):
    """Plot fit projection slices with identical component accumulation for both 1D and 2D tracks."""
    logger = Logger(print_prefix="[plot_fit_with_true_shapes]", verbosity=verbosity)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.05)
    
    bin_edges = np.linspace(fit_range[0], fit_range[1], nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (fit_range[1] - fit_range[0]) / nbins
    
    # 1. Generate reference stacked histograms from true high-statistics MC entries
    component_hists = {}
    for comp_name in component_names:
        if comp_name in component_data_dict:
            data = component_data_dict[comp_name]
            if data is not None and len(data) > 0:
                counts, _ = np.histogram(data, bins=bin_edges)
                if scale_factors_dict and comp_name in scale_factors_dict:
                    counts = counts * scale_factors_dict[comp_name]
                component_hists[comp_name] = counts
            else:
                component_hists[comp_name] = np.zeros(nbins)
        else:
            component_hists[comp_name] = np.zeros(nbins)
    
    bottom = np.zeros(nbins)
    desired_order = ['rpc',  'cosmic', 'dio', 'rmc_ext', 'rmc_int', 'ipa', 'ce']
    
    for comp_name in desired_order:
        if comp_name in component_hists and np.sum(component_hists[comp_name]) > 0:
            display_info = COMPONENT_DISPLAY.get(comp_name, {'color': 'C0', 'label': comp_name.upper()})
            ax1.bar(bin_centers, component_hists[comp_name], width=bin_width, 
                   bottom=bottom, label=display_info['label'], 
                   color=display_info['color'], alpha=0.8)
            bottom += component_hists[comp_name]
    
    total_hist = bottom.copy()
    total_events = np.sum(total_hist)
    
    # 2. Extract exact parameter yields with a clean lowercase lookup dictionary
    fitted_yields = {}
    if fit_result and hasattr(fit_result, 'params'):
        for p, p_data in fit_result.params.items():
            p_name = p.name if hasattr(p, 'name') else str(p)
            fitted_yields[p_name.lower()] = p_data['value']

    fit_vals_for_plot = None
    if combine_pdf is not None:
        try:
            mom_vals = np.linspace(fit_range[0], fit_range[1], 200)
            pdf_obs = combine_pdf.obs
            n_obs = len(list(pdf_obs)) if isinstance(pdf_obs, (tuple, list, set)) else 1

            # Style mapping configured using strict normalized lowercase tokens
            fitline_styles = {
                'ce': {'color': 'red', 'label': 'CE Fit Line'},
                'cosmic': {'color': 'cyan', 'label': 'Cosmic Fit Line'},
                'dio': {'color': 'magenta', 'label': 'DIO Fit Line'},
                'rpc': {'color': 'green', 'label': 'RPC Fit Line'}
            }

            if n_obs == 1:
                # Standard 1D Projection
                obs_space = zfit.Space(pdf_obs, limits=fit_range)
                eval_tensor = tf.constant(mom_vals, dtype=tf.float32)
                pdf_vals = combine_pdf.pdf(eval_tensor, norm_range=obs_space)
                pdf_vals = np.array(pdf_vals.numpy() if hasattr(pdf_vals, 'numpy') else pdf_vals)
                
                area = np.trapz(pdf_vals, mom_vals)
                if area > 0: pdf_vals = pdf_vals / area
                scaled_pdf = pdf_vals * total_events * bin_width
                fit_vals_for_plot = (mom_vals, scaled_pdf)
                ax1.plot(mom_vals, scaled_pdf, 'k-', linewidth=2.5, label='Total Fit', zorder=10)
                
                if hasattr(combine_pdf, 'pdfs'):
                    for i, pdf in enumerate(combine_pdf.pdfs):
                        param_names = [p.name.lower() for p in pdf.get_params()]
                        
                        component_token = None
                        for token in ['ce', 'cem', 'cosmic', 'dio', 'rpc']:
                            if any(token in p_str for p_str in param_names):
                                component_token = 'ce' if token == 'cem' else token
                                break
                        
                        # Fallback to structure sequence check if parameter strings match multi-channels
                        if component_token is None:
                            fallback_sequence = {0: 'dio', 1: 'rpc', 2: 'rpc', 3: 'cosmic', 4: 'ce'}
                            component_token = fallback_sequence.get(i, None)
                            
                        if not component_token:
                            continue
                        
                        yield_val = 0.0
                        if target_yields_dict:
                            yield_val = target_yields_dict.get(component_token, 0.0)
                            if yield_val == 0.0 and component_token == 'ce':
                                yield_val = target_yields_dict.get('ce', 73.0)

                        fit_yield = next((v for k, v in fitted_yields.items() if component_token in k), 0.0)
                        if fit_yield > 0.0:
                    
                            yield_val = fit_yield

                        local_norm = pdf.norm_range if hasattr(pdf, 'norm_range') and pdf.norm_range else obs_space
                        comp_vals = pdf.pdf(eval_tensor, norm_range=local_norm)
                        comp_vals_np = np.array(comp_vals.numpy() if hasattr(comp_vals, 'numpy') else comp_vals)
                        c_area = np.trapz(comp_vals_np, mom_vals)
                        if c_area > 0: comp_vals_np = comp_vals_np / c_area
                        
                        scaled_1d_pdf = comp_vals_np * yield_val * bin_width
                        style = fitline_styles.get(component_token, {'color': f'C{i}', 'label': f'{component_token.upper()} Fit Line'})
                        
                        # Group multi-channel sub-PDF tracks smoothly on 1D projections
                        existing_line = None
                        for line in ax1.lines:
                            if line.get_label() == style['label']:
                                existing_line = line
                                break
                                
                        if existing_line is not None:
                            old_x, old_y = existing_line.get_data()
                            existing_line.set_ydata(old_y + scaled_1d_pdf)
                        else:
                            ax1.plot(mom_vals, scaled_1d_pdf, '--', 
                                     linewidth=1.8, label=style['label'], color=style['color'], alpha=0.95, zorder=12)

            else:
                # Robust 2D PDF Projection Handling
                obs_space = combine_pdf.space
                obs_names = list(pdf_obs)
                scan_idx = 1 if plot_obs == 'time' and len(obs_names) > 1 else 0
                other_idx = 1 - scan_idx
                
                limits_low, limits_high = obs_space.rect_limits
                lo_arr = np.asarray(limits_low, dtype=float).reshape(-1)
                hi_arr = np.asarray(limits_high, dtype=float).reshape(-1)
                other_grid = np.linspace(lo_arr[other_idx], hi_arr[other_idx], 120)

                scan_rep = np.tile(mom_vals, len(other_grid))
                other_rep = np.repeat(other_grid, len(mom_vals))
                pts = np.zeros((len(scan_rep), 2), dtype=np.float32)
                pts[:, scan_idx] = scan_rep.astype(np.float32)
                pts[:, other_idx] = other_rep.astype(np.float32)

                vals = combine_pdf.pdf(tf.constant(pts, dtype=tf.float32), norm_range=obs_space)
                vals_np = np.array(vals.numpy() if hasattr(vals, 'numpy') else vals)
                pdf_vals = np.trapz(vals_np.reshape(len(other_grid), len(mom_vals)), other_grid, axis=0)
                
                area = np.trapz(pdf_vals, mom_vals)
                if area > 0: pdf_vals = pdf_vals / area
                scaled_pdf = pdf_vals * total_events * bin_width
                fit_vals_for_plot = (mom_vals, scaled_pdf)
                ax1.plot(mom_vals, scaled_pdf, 'k-', linewidth=2.5, label='Total Fit', zorder=10)

                if hasattr(combine_pdf, 'pdfs'):
                    for i, pdf in enumerate(combine_pdf.pdfs):
                        param_names = [p.name.lower() for p in pdf.get_params()]
                        
                        component_token = None
                        for token in ['ce', 'cem', 'cosmic', 'dio', 'rpc']:
                            if any(token in p_str for p_str in param_names):
                                component_token = 'ce' if token == 'cem' else token
                                break
                        
                        if component_token is None:
                            fallback_sequence = {0: 'dio', 1: 'rpc', 2: 'rpc', 3: 'cosmic', 4: 'ce'}
                            component_token = fallback_sequence.get(i, None)
                        
                        if not component_token:
                            continue
                                
                        local_norm = pdf.space if hasattr(pdf, 'space') and pdf.space else obs_space
                        comp_vals = pdf.pdf(tf.constant(pts, dtype=tf.float32), norm_range=local_norm)
                        comp_vals_np = np.array(comp_vals.numpy() if hasattr(comp_vals, 'numpy') else comp_vals)
                        comp_proj = np.trapz(comp_vals_np.reshape(len(other_grid), len(mom_vals)), other_grid, axis=0)
                        
                        c_area = np.trapz(comp_proj, mom_vals)
                        if c_area > 0: 
                            comp_proj = comp_proj / c_area
                        
                        current_comp_yield = 0.0
                        if target_yields_dict:
            
                            current_comp_yield = target_yields_dict.get(component_token, 0.0)
                            if current_comp_yield == 0.0 and component_token == 'ce':
                                current_comp_yield = target_yields_dict.get('ce', 73.0)

                        fit_yield = next((v for k, v in fitted_yields.items() if component_token in k), 0.0)
                        if fit_yield > 0.0:
                            current_comp_yield = fit_yield

                        scaled_comp_pdf = comp_proj * current_comp_yield * bin_width
                        style = fitline_styles.get(component_token, {'color': f'C{i}', 'label': f'{component_token.upper()} Fit Line'})
                        
                        existing_line = None
                        for line in ax1.lines:
                            if line.get_label() == style['label']:
                                existing_line = line
                                break
                                
                        if existing_line is not None:
                            old_x, old_y = existing_line.get_data()
                            existing_line.set_ydata(old_y + scaled_comp_pdf)
                        else:
                            ax1.plot(mom_vals, scaled_comp_pdf, '--', linewidth=2.0, 
                                     label=style['label'], color=style['color'], alpha=0.95, zorder=12)

        except Exception as e:
            logger.log(f"Could not render component sub-shapes: {e}", "warning")
    
    ax1.set_ylabel('Events / Bin', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=0.1, ymax=2000)
    
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=FONTS['legend']['size'])
    
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
                    err.append(total_err / data_val)
                    dev.append((fit_val - data_val) / total_err)
                else:
                    err.append(0); dev.append(0)
            else:
                err.append(0); dev.append(0)
        
        ax2.errorbar(bin_centers, dev, yerr=err, color='None', marker='+', 
                    markerfacecolor='black', ecolor='black', capsize=3, markersize=8)
        ax2.axhline(0, color='red', linewidth=1.5)
        ax2.set_ylabel(r'Pull [$\sigma$]', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
        x_label = 'Reconstructed Momentum [MeV/c]' if plot_obs == 'mom' else 'Track Time [ns]'
        ax2.set_xlabel(x_label, fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
        ax2.yaxis.set_ticks(np.arange(-5, 6, 2))
        ax2.set_xlim(fit_range)
        ax2.tick_params(axis='both', labelsize=FONTS['tick']['size'])
    else:
        ax2.axis('off')
    
    fig.text(0.15, 0.98, f"Mu2e Simulation ({title})", fontsize=FONTS['label']['size'], fontweight='bold', ha='left', va='top')
    ax1.grid(False)
    if fit_vals_for_plot is not None:
        ax2.grid(False)
        
    if output_file:
        plt.tight_layout()
        fig.savefig(output_file)
        logger.log(f"Saved projection to {output_file}", "info")
    plt.close(fig)


class ScaledFitBuilder:
    """Build and fit scaled MC component samples using continuous dataset weights."""
    
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
        
        self.component_yields = { 
            'dio': 1427.0,           
            'cosmic': 333.0,         
            'rpc': 9.0,              
            'rmc_ext': 0.0,         
            'rmc_int': 0.0,         
            'ipa': 0.0,             
            'ce': 0.0                 
        }
    
    def set_component_yields(self, yields_dict: Dict[str, Optional[float]]):
        self.component_yields.update(yields_dict)
        self.logger.log(f"Updated physics target yields: {self.component_yields}", "info")

    def verify_dataset_scaling_truth(self, particle_categories: np.ndarray, continuous_weights: np.ndarray):
        """Calculates and logs the sum of weights for each channel in the active fitting window."""
        self.logger.log("\n" + "="*90, "info")
        self.logger.log(f"                 MC TRUTH DATASET SCALING INJECTION PROOF TABLE", "info")
        self.logger.log("="*90, "info")
        self.logger.log(f"{'CHANNEL CONTEXT':<18} | {'MC CODE':<8} | {'RAW EVENTS IN WINDOW':<22} | {'SUM OF WEIGHTS IN FIT':<25}", "info")
        self.logger.log("-"*90, "info")
        
        unique_codes = np.unique(particle_categories)
        for code in unique_codes:
            mask = (particle_categories == code)
            raw_count = np.sum(mask)
            weight_sum = np.sum(continuous_weights[mask])
            label = MC_CODE_TO_COMPONENT.get(code, f"CODE_{code}")
            
            rpc_flag = " (Part of Grouped RPC)" if code in [2000] else ""
            self.logger.log(f"{label:<18} | {code:<8} | {raw_count:<22} | {weight_sum:<25.2f}{rpc_flag}", "info")
            
        self.logger.log("="*90 + "\n", "info")

    def print_fit_comparison(self, fit_result):
        """Extract parameters dynamically matching N_process formatting."""
        print("Fit Result:",fit_result)
        self.logger.log("\n" + "="*80, "info")
        self.logger.log(f"{'COMPONENT':<15} | {'TARGET (TRUE) YIELD':<22} | {'FITTED UNBINNED YIELD':<25}", "info")
        self.logger.log("="*80, "info")
        
        fitted_params = {}
        if fit_result and hasattr(fit_result, 'params'):
            for param, val in fit_result.params.items():
                p_name = (param.name if hasattr(param, 'name') else str(param)).upper()
                fitted_params[p_name] = {
                    'val': val['value'],
                    'err': val.get('minuit_hesse', {}).get('error', 0.0) if 'minuit_hesse' in val else val.get('error', 0.0)
                }

        for comp, target in self.component_yields.items():
            if target is None: continue
            
            fit_val_str = "Not Found in Model"
            lookup_token = comp.lower()
            
            if 'rpc' in lookup_token:
                param_targets = ["N_RPC", "RPC"]
            elif 'cosmic' in lookup_token:
                param_targets = ["N_COSMIC", "COSMIC"]
            elif 'dio' in lookup_token:
                param_targets = ["N_DIO", "DIO"]
            elif 'ce' in lookup_token or 'cem' in lookup_token:
                param_targets = ["N_CE", "N_CEM", "CE", "CEM"]
            else:
                param_targets = [f"N_{comp.upper()}", comp.upper()]
                
            for target_name in param_targets:
                if target_name in fitted_params:
                    p_data = fitted_params[target_name]
                    fit_val_str = f"{p_data['val']:>8.2f} +/- {p_data['err']:<7.2f}"
                    break
                        
            self.logger.log(f"{comp.upper():<15} | {target:<22.1f} | {fit_val_str}", "info")
            
        self.logger.log("="*80 + "\n", "info")

    def _process_file_list(self, file_path: str, sign: str = "minus", 
                          location: str = "disk", component_name: str = "unknown",
                          mom_lo: Optional[float] = None, mom_hi: Optional[float] = None) -> Optional[ak.Array]:
        if mom_lo is None: mom_lo = self.fit_range_lo
        if mom_hi is None: mom_hi = self.fit_range_hi
        
        cut_switches = [
            True, True, True, True, True, True, True, False, False, False, 
            True, True, True, True, True, True, True, True, False, False, False, False
        ]
        try:
            processor = AnaProcessor(
                file_list_path=file_path, jobs=self.jobs, cuts=cut_switches,
                location=location, mom_lo=mom_lo, mom_hi=mom_hi
            )
            results_list = processor.execute()
            if not results_list: return None
            
            data_list = []
            for result in results_list:
                if result is not None:
                    if isinstance(result, dict) and "filtered_data" in result:
                        data_list.append(result["filtered_data"])
                    else:
                        data_list.append(result)
            return ak.concatenate(data_list) if data_list else None
        except Exception as e:
            self.logger.log(f"Error processing file list {file_path}: {e}", "error")
            return None
    
    def load_and_scale_components(self, component_files: Dict[str, str], sign: str = "minus", location: str = "disk", fit_range: Optional[Tuple[float, float]] = None):
        self.components = {}
        self.scaled_components = {}
        
        if fit_range is not None:
            self.fit_range_lo, self.fit_range_hi = fit_range
        
        for component_name, file_path in component_files.items():
            if file_path is None or not Path(file_path).exists(): continue
            data = self._process_file_list(file_path, sign=sign, location=location, component_name=component_name, mom_lo=self.fit_range_lo, mom_hi=self.fit_range_hi)
            if data is not None:
                self.components[component_name] = data
    
    def _extract_synchronized_2d(self, data: ak.Array) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            selector = Select(verbosity=0)
            vector = Vector()
            
            trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
            trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
            
            mom_mag_ak = vector.get_mag(trkfit_ent, 'mom')
            time_ak = trkfit_ent['time']
            
            valid_mask = (~ak.is_none(mom_mag_ak, axis=-1)) & (~ak.is_none(time_ak, axis=-1))
            
            clean_mom = ak.flatten(mom_mag_ak[valid_mask], axis=None)
            clean_time = ak.flatten(time_ak[valid_mask], axis=None)
            
            return np.array(clean_mom), np.array(clean_time)
        except Exception:
            return None, None

    def _extract_variable(self, data: ak.Array, var_name: str) -> Optional[np.ndarray]:
        try:
            if var_name.lower() == "recomom_ttfront":
                selector = Select(verbosity=0)
                vector = Vector()
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                mom_mag = ak.drop_none(vector.get_mag(trkfit_ent, 'mom'))
                return np.array(ak.flatten(mom_mag, axis=None))
            elif var_name.lower() == "tracktime_ttfront":
                selector = Select(verbosity=0)
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                time = DataPreparationManager.clean_awkward_array(trkfit_ent['time'])
                return np.array(ak.flatten(time, axis=None))
            else:
                parts = var_name.split('.')
                val = data
                for part in parts: val = val[part]
                return np.array(ak.flatten(ak.drop_none(val), axis=None))
        except Exception:
            return None
    
    def _build_combined_data(self, variables: Dict[str, List[np.ndarray]], scale_factors: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        combined = {}
        combined_component_cats = []
        combined_weights = []
        comp_names = list(self.scaled_components.keys())
        
        for var_name, arrays_per_comp in variables.items():
            combined_arrays = []
            component_cats = []
            weights_array = []
            
            for comp_name, array in zip(comp_names, arrays_per_comp):
                if array is None or len(array) == 0: continue
                scale_factor = scale_factors[comp_name]
                combined_arrays.append(array)
                weights_array.append(np.full(len(array), fill_value=scale_factor, dtype=np.float32))
                
                lookup_name = comp_name.lower()
                mc_code = COMPONENT_TO_MC_CODE.get(lookup_name, -2)
                    
                component_cats.extend([mc_code] * len(array))
            
            combined[var_name] = np.concatenate(combined_arrays)
            if var_name == list(variables.keys())[0]:
                combined_component_cats = np.array(component_cats)
                combined_weights = np.concatenate(weights_array)
        
        return combined, combined_component_cats, combined_weights
    
    def fit_mom_1d(self, variable: str = 'recomom_ttfront', fit_range: Tuple[float, float] = (97, 110),
                   plot: bool = True, minos: bool = False, use_constraints: bool = False, constraints_dir: str = 'uncertainties/outputs') -> Any:
        if not self.components: return None
        
        comp_names = list(self.components.keys())
        variables = {variable: []}
        scale_factors = {}
        component_data_for_plot = {}  
        
        for comp_name in comp_names:
            data = self.components[comp_name]
            var_array = self._extract_variable(data, variable)
            
            if var_array is not None and len(var_array) > 0:
                variables[variable].append(var_array)
                component_data_for_plot[comp_name] = var_array
                
                target = self.component_yields.get(comp_name.lower(), 0.0)
                scale_factor = (target / len(var_array)) if (len(var_array) > 0 and target is not None and target > 0) else 0.0
            else:
                variables[variable].append(None)
                component_data_for_plot[comp_name] = None
                scale_factor = 0.0
                
            scale_factors[comp_name] = scale_factor
            self.scaled_components[comp_name] = (data, scale_factor)
            self.logger.log(f"Normalized {comp_name:<10} | Window entries: {len(var_array) if var_array is not None else 0:<6} | Weight: {scale_factor:.6f}", "info")
        
        combined_data, component_cats, combined_weights = self._build_combined_data(variables, scale_factors)
        mom_mag = combined_data[variable]
        
        self.verify_dataset_scaling_truth(component_cats, combined_weights)
        
        fit_tuple = Unbinned_fit_mom(
            mom_mag=mom_mag, count_particle_types=component_cats, fit_range_low=fit_range[0], fit_range_hi=fit_range[1],
            weights=combined_weights, plot_truth=False, verbose=self.verbosity, minos=minos,
            plot_NLL=False, plot_results=False, constraints_dir=constraints_dir if use_constraints else None
        )
        result, poi, loss, aux_nlls, combine_pdf, constraints = fit_tuple
        
        self.print_fit_comparison(result)
        
        if plot:
            plot_fit_with_true_shapes(
                mom_mag=mom_mag, combine_pdf=combine_pdf, fit_result=result, component_data_dict=component_data_for_plot,
                component_names=comp_names, scale_factors_dict=scale_factors, fit_range=fit_range, nbins=25,
                output_file="fit_momentum_1d_projections.png", title="1D Momentum Projection", verbosity=self.verbosity,
                target_yields_dict=self.component_yields
            )
        return result
    
    def fit_mom_time_2d(self, mom_variable: str = 'recomom_ttfront', time_variable: str = 'tracktime_ttfront',
                       fit_range_mom: Tuple[float, float] = (97, 110), fit_range_time: Tuple[float, float] = (475, 1650),
                       plot: bool = True, use_constraints: bool = False, constraints_dir: str = 'uncertainties/outputs') -> Any:
        if not self.components: return None
        
        comp_names = list(self.components.keys())
        variables = {'mom': [], 'time': []}
        scale_factors = {}
        component_mom_for_plot = {}
        component_time_for_plot = {}
        
        for comp_name in comp_names:
            data = self.components[comp_name]
            
            mom_array, time_array = self._extract_synchronized_2d(data)
            
            if mom_array is not None and time_array is not None and len(mom_array) > 0:
                in_bounds = (mom_array >= fit_range_mom[0]) & (mom_array <= fit_range_mom[1]) & \
                            (time_array >= fit_range_time[0]) & (time_array <= fit_range_time[1])
                            
                mom_array = mom_array[in_bounds]
                time_array = time_array[in_bounds]
                
            if mom_array is not None and len(mom_array) > 0:
                variables['mom'].append(mom_array)
                variables['time'].append(time_array)
                component_mom_for_plot[comp_name] = mom_array
                component_time_for_plot[comp_name] = time_array
                
                target = self.component_yields.get(comp_name.lower(), 0.0)
                scale_factor = (target / len(mom_array)) if (len(mom_array) > 0 and target is not None and target > 0) else 0.0
            else:
                variables['mom'].append(None); variables['time'].append(None)
                component_mom_for_plot[comp_name] = None; component_time_for_plot[comp_name] = None
                scale_factor = 0.0
                
            scale_factors[comp_name] = scale_factor
            self.scaled_components[comp_name] = (data, scale_factor)
            self.logger.log(f"Normalized 2D {comp_name:<10} | Window entries: {len(mom_array) if mom_array is not None else 0:<6} | Weight: {scale_factor:.6f}", "info")
        
        combined_data, component_cats, combined_weights = self._build_combined_data(variables, scale_factors)
        mom_mag, times = combined_data['mom'], combined_data['time']

        self.verify_dataset_scaling_truth(component_cats, combined_weights)

        fit_tuple = Unbinned_2d_fit_mom_time(
            mom_mag=mom_mag, times=times, count_particle_types=component_cats,
            fit_range_mom=fit_range_mom, fit_range_time=fit_range_time, weights=combined_weights,
            plot_truth=False, verbose=self.verbosity, plot_NLL=False, plot_results=False,
            constraints_dir=constraints_dir if use_constraints else None
        )
        result, poi, loss, combine_pdf, norms = fit_tuple
        
        self.print_fit_comparison(result)
        
        if plot:
            plot_fit_with_true_shapes(
                mom_mag=mom_mag, combine_pdf=combine_pdf, fit_result=result, component_data_dict=component_mom_for_plot,
                component_names=comp_names, scale_factors_dict=scale_factors, fit_range=fit_range_mom, nbins=25,
                output_file="fit_momentum_2d_projection.png", title="2D Fit: Momentum Projection", verbosity=self.verbosity, 
                plot_obs='mom', target_yields_dict=self.component_yields
            )
            plot_fit_with_true_shapes(
                mom_mag=times, combine_pdf=combine_pdf, fit_result=result, component_data_dict=component_time_for_plot,
                component_names=comp_names, scale_factors_dict=scale_factors, fit_range=fit_range_time, nbins=30,
                output_file="fit_time_2d_projection.png", title="2D Fit: Time Projection", verbosity=self.verbosity, 
                plot_obs='time', target_yields_dict=self.component_yields
            )

        # store the mom and time for subsequent fits

        # store the mom and time for subsequent fits
        try:
            if 'args' in globals() and hasattr(globals()['args'], 'export_npz') and globals()['args'].export_npz:
                # --- REDESIGNED NPZ EXPORT BLOCK ---
                # Safeguard lookup for when running programmatically without a global 'args'
                current_args = globals().get('args', None) or getattr(builtins, 'args', None)
                
                if current_args and hasattr(current_args, 'export_npz') and current_args.export_npz:
                    export_path = Path(current_args.export_npz)
                    
                    fit_param_vals = {}
                    fit_param_errs = {}
                    
                    # 1. Extract Fitted Yields and Uncertainties from the Result object
                    if result and hasattr(result, 'params'):
                        for p, p_data in result.params.items():
                            p_name = p.name if hasattr(p, 'name') else str(p)
                            fit_param_vals[f"val_{p_name}"] = p_data['value']
                            fit_param_errs[f"err_{p_name}"] = p_data.get('error', 0.0)

                    # 2. Extract Shape Parameters dynamically from all underlying PDFs
                    if combine_pdf and hasattr(combine_pdf, 'pdfs'):
                        for i, sub_pdf in enumerate(combine_pdf.pdfs):
                            # Query all parameters associated with this specific shape instance
                            for shape_param in sub_pdf.get_params():
                                p_name = shape_param.name
                                # Only grab if we haven't already extracted it from the fit result
                                if f"val_{p_name}" not in fit_param_vals:
                                    fit_param_vals[f"val_{p_name}"] = float(shape_param.numpy())
                                    fit_param_errs[f"err_{p_name}"] = 0.0  # Constant shape parameter

                    # 3. Save everything down to a comprehensive dictionary
                    export_dict = {
                        'mom': mom_mag,
                        'time': times,
                        'weights': combined_weights,
                        'categories': component_cats,
                        'fit_ranges': np.array([fit_range_mom[0], fit_range_mom[1], fit_range_time[0], fit_range_time[1]]),
                    }
                    # Append dynamic parameter keys cleanly
                    export_dict.update(fit_param_vals)
                    export_dict.update(fit_param_errs)

                    np.savez(export_path, **export_dict)
                    self.logger.log(f"Redesigned config exported. Packed {len(fit_param_vals)} model parameters into: {export_path}", "success")
                # --- END OF REDESIGNED BLOCK ---
        except Exception:
            pass
        
        return result


def main(args):
    builder = ScaledFitBuilder(verbosity=args.verbosity, jobs=args.jobs)
    component_files = {item.split('=')[0]: item.split('=')[1] for item in args.components} if args.components else {}
    
    if args.yields:
        yields_dict = {item.split('=')[0]: (float(item.split('=')[1]) if item.split('=')[1].lower() != 'none' else None) for item in args.yields}
        builder.set_component_yields(yields_dict)
        
    builder.load_and_scale_components(component_files, sign=args.sign, location=args.location, fit_range=(args.fit_range_lo, args.fit_range_hi))
    
    if args.fit_type in ['1d', 'both']:
        builder.fit_mom_1d(variable=args.variable, fit_range=(args.fit_range_lo, args.fit_range_hi), plot=not args.no_plot, minos=args.minos, constraints_dir=args.constraints_dir)
    if args.fit_type in ['2d', 'both']:
        builder.fit_mom_time_2d(mom_variable=args.variable, time_variable=args.time_variable, fit_range_mom=(args.fit_range_lo, args.fit_range_hi), fit_range_time=(args.time_range_lo, args.time_range_hi), plot=not args.no_plot, constraints_dir=args.constraints_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Weighted Unbinned Asimov Fitter")
    parser.add_argument('--fit-type', choices=['1d', '2d', 'both'], default='2d')
    parser.add_argument('--components', nargs='+', required=True)
    parser.add_argument('--variable', type=str, default='recomom_ttfront')
    parser.add_argument('--time-variable', type=str, default='tracktime_ttfront')
    parser.add_argument('--fit-range-lo', type=float, default=97)
    parser.add_argument('--fit-range-hi', type=float, default=110)
    parser.add_argument('--time-range-lo', type=float, default=475)
    parser.add_argument('--time-range-hi', type=float, default=1650)
    parser.add_argument('--yields', nargs='+', default=None)
    parser.add_argument('--sign', choices=['minus', 'plus'], default='minus')
    parser.add_argument('--location', choices=['disk', 'local'], default='disk')
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--constraints-dir', type=str, default='uncertainties/outputs')
    parser.add_argument('--minos', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--export-npz', type=str, default=None)
    args = parser.parse_args()
    main(args)