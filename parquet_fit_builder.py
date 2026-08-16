"""
Build and execute fits on pre-cut parquet data with momentum and time variables.
Skips root file processing and directly loads pre-prepared parquet files.
"""

# Suppress TensorFlow and graph optimization warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow log messages
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent OOM issues

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import awkward as ak
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple, List, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import zfit
import tensorflow as tf
from PIL import Image

# Configure logging to suppress verbose warnings
import logging
tf.get_logger().setLevel('ERROR')
logging.getLogger('absl').setLevel(logging.ERROR)

from pyutils.pylogger import Logger
from fit_module import Unbinned_fit_mom, Unbinned_2d_fit_mom_time
from style import FONTS, COLORS
from model.physics_components import mom_components
from config import GLOBAL_VERBOSITY
from datacard import DataCard
from generate_datacard import datacard_from_snapshot

# Standardized MC truth process codes
COMPONENT_TO_MC_CODE = {
    'dio': 166,
    'ipa': 0,
    'ce': 168,
    'cem': 168,
    'cep': 176,
    'cosmic': -1,
    'rpc': 178,
    'rmc': 173,  # Combined RMC0N (uses 0N external code as primary)
}

# Reverse mapping to identify names from process codes during validation
MC_CODE_TO_COMPONENT = {v: k.upper() for k, v in COMPONENT_TO_MC_CODE.items()}

COMPONENT_DISPLAY = {
    'dio': {'color': '#e377c2', 'label': 'DIO'},
    'cosmic': {'color': '#1f77b4', 'label': 'Cosmic'},
    'rpc': {'color': '#2ca02c', 'label': 'RPC (combined)'},
    'rmc': {'color': '#dc143c', 'label': 'RMC (combined)'},
    'ipa': {'color': '#8c564b', 'label': 'IPA'},
    'ce': {'color': '#ff8000', 'label': 'Signal (CE)'},
    'cem': {'color': '#ff8000', 'label': 'Signal (CE)'},
    'cep': {'color': '#ff7f0e', 'label': 'Signal (CE+)'},
}


def plot_fit_with_true_shapes(mom_mag, combine_pdf=None, fit_result=None, component_data_dict=None, component_names=None, 
                              scale_factors_dict=None, fit_range=(100, 110), nbins=25, output_file=None,
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
    
    # Debug: log which components have data
    logger.log(f"Component names passed to plot: {component_names}", "info")
    logger.log(f"Component data dict keys: {list(component_data_dict.keys())}", "info")
    for comp_name in component_names:
        data = component_data_dict.get(comp_name)
        n_events = len(data) if data is not None else 0
        logger.log(f"  {comp_name}: {n_events} events in fit range", "info")
    
    bottom = np.zeros(nbins)
    desired_order = [ 'rpc','cosmic',  'rmc', 'dio', 'ipa', 'ce']
    
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
                'ce': {'color': 'red', 'label': 'CE (Signal)'},
                'cosmic': {'color': 'cyan', 'label': 'Cosmic-Induced'},
                'dio': {'color': 'magenta', 'label': 'DIO'},
                'rpc': {'color': 'green', 'label': 'RPC'}
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
                        for token in ['ce', 'cem', 'cosmic', 'dio', 'rpc', 'rmc']:
                            if any(token in p_str for p_str in param_names):
                                component_token = 'ce' if token == 'cem' else token
                                break
                        
                        # Fallback to structure sequence check if parameter strings match multi-channels
                        if component_token is None:
                            fallback_sequence = {0: 'dio', 1: 'rpc', 2: 'rpc', 3: 'cosmic', 4: 'ce', 5: 'rmc'}
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
                        for token in ['ce', 'cem', 'cosmic', 'dio', 'rpc', 'rmc']:
                            if any(token in p_str for p_str in param_names):
                                component_token = 'ce' if token == 'cem' else token
                                break
                        
                        if component_token is None:
                            fallback_sequence = {0: 'dio', 1: 'rpc', 2: 'rpc', 3: 'cosmic', 4: 'ce', 5: 'rmc'}
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
    ax1.set_ylim(ymin=0.1, ymax=500)
    
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', ncol=2, fontsize=FONTS['legend']['size'])
    
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
    
    ax1.text(
        x=0.05,
        y=1.02,
        s="Mu2e Simulation", 
        transform=ax1.transAxes,
        fontsize=FONTS['label']['size'], 
        weight='bold', 
        family='serif',
        va='bottom',
        ha='left'
    )

    ax1.grid(False)
    if fit_vals_for_plot is not None:
        ax2.grid(False)
        
    if output_file:
        plt.tight_layout()
        fig.savefig(output_file)
        logger.log(f"Saved projection to {output_file}", "info")
    plt.close(fig)


class ParquetFitBuilder:
    """Build and fit pre-cut parquet data with momentum and time variables."""
    
    def __init__(self, verbosity=1, jobs=1):
        self.logger = Logger(print_prefix="[ParquetFitBuilder]", verbosity=verbosity)
        self.verbosity = verbosity
        self.jobs = jobs
        self.components = {}  
        self.scaled_components = {}  
        self.component_yields = {}  # Will be populated from input card via --card argument
        self.data = None  
        self.fit_range_lo = 100  
        self.fit_range_hi = 110  
        self.input_card = None  # Input datacard (Combine-style) with expected yields + systematics

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

    def _load_parquet_file(self, file_path: str, mom_column: str = 'momentum', 
                           time_column: str = 'time', component_name: str = "unknown") -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load momentum and time data from a parquet file."""
        try:
            import pyarrow.parquet as pq
            
            if not Path(file_path).exists():
                self.logger.log(f"Parquet file not found: {file_path}", "error")
                return None
            
            # Read the parquet file
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            self.logger.log(f"Loaded parquet file: {file_path}", "info")
            self.logger.log(f"  Available columns: {df.columns.tolist()}", "info")
            self.logger.log(f"  Shape: {df.shape}", "info")
            
            # Extract momentum and time columns
            if mom_column not in df.columns:
                self.logger.log(f"  Warning: momentum column '{mom_column}' not found. Using available columns: {df.columns.tolist()}", "warning")
                # Try common alternative names
                mom_col_options = ['momentum', 'mom', 'p', 'recomom', 'recomom_ttfront']
                mom_column = next((col for col in mom_col_options if col in df.columns), None)
                if mom_column is None:
                    self.logger.log(f"  Error: Could not find momentum column", "error")
                    return None
            
            if time_column not in df.columns:
                self.logger.log(f"  Warning: time column '{time_column}' not found", "warning")
                # Try common alternative names
                time_col_options = ['time', 'tracktime', 'tracktime_ttfront', 't']
                time_column = next((col for col in time_col_options if col in df.columns), None)
                if time_column is None:
                    self.logger.log(f"  Warning: Could not find time column, will return None for times", "warning")
            
            mom_data = np.array(df[mom_column].values, dtype=np.float32)
            time_data = np.array(df[time_column].values, dtype=np.float32) if time_column else np.zeros_like(mom_data)
            
            self.logger.log(f"  Extracted {len(mom_data)} momentum values", "info")
            
            return mom_data, time_data
            
        except ImportError:
            self.logger.log("PyArrow not available. Attempting to use pickle/fallback loader.", "warning")
            try:
                # Fallback: try using pickle if available
                import pickle
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    mom_data = np.array(data.get(mom_column, []), dtype=np.float32)
                    time_data = np.array(data.get(time_column, np.zeros_like(mom_data)), dtype=np.float32)
                    return mom_data, time_data
            except Exception as e:
                self.logger.log(f"Fallback loader failed: {e}", "error")
            return None
        except Exception as e:
            self.logger.log(f"Error loading parquet file {file_path}: {e}", "error")
            return None
    
    def load_parquet_components(self, component_files: Dict[str, str], 
                               mom_column: str = 'momentum', time_column: str = 'time',
                               fit_range: Optional[Tuple[float, float]] = None,
                               time_range: Optional[Tuple[float, float]] = None):
        """Load pre-cut parquet files for each component."""
        self.components = {}
        self.scaled_components = {}
        
        if fit_range is not None:
            self.fit_range_lo, self.fit_range_hi = fit_range
        
        self.time_range_lo = time_range[0] if time_range else 475
        self.time_range_hi = time_range[1] if time_range else 1650
        
        for component_name, file_path in component_files.items():
            if file_path is None:
                continue
                
            result = self._load_parquet_file(file_path, mom_column=mom_column, 
                                            time_column=time_column, component_name=component_name)
            if result is not None:
                mom_data, time_data = result
                
                # Store as dictionary for compatibility with the fitter
                self.components[component_name] = {
                    'momentum': mom_data,
                    'time': time_data
                }
                self.logger.log(f"Loaded component '{component_name}': {len(mom_data)} events", "info")
            else:
                self.logger.log(f"Failed to load component '{component_name}'", "warning")
        
        # Combine RMC0N components (ext + int) into single 'rmc' component, like RPC
        if 'rmc0n_ext' in self.components or 'rmc0n_int' in self.components:
            rmc_moms = []
            rmc_times = []
            
            if 'rmc0n_ext' in self.components:
                rmc_moms.append(self.components['rmc0n_ext']['momentum'])
                rmc_times.append(self.components['rmc0n_ext']['time'])
            
            if 'rmc0n_int' in self.components:
                rmc_moms.append(self.components['rmc0n_int']['momentum'])
                rmc_times.append(self.components['rmc0n_int']['time'])
            
            if rmc_moms:
                self.components['rmc'] = {
                    'momentum': np.concatenate(rmc_moms),
                    'time': np.concatenate(rmc_times)
                }
                self.logger.log(f"Combined RMC0N components into 'rmc': {len(self.components['rmc']['momentum'])} total events", "info")
                
                # Remove the individual components
                self.components.pop('rmc0n_ext', None)
                self.components.pop('rmc0n_int', None)
    
    def _build_combined_data(self, variables: Dict[str, List[np.ndarray]], scale_factors: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Combine data from all components with proper weighting."""
        combined = {}
        combined_component_cats = []
        combined_weights = []
        comp_names = list(self.scaled_components.keys())
        
        for var_name, arrays_per_comp in variables.items():
            combined_arrays = []
            component_cats = []
            weights_array = []
            
            for comp_name, array in zip(comp_names, arrays_per_comp):
                if array is None or len(array) == 0:
                    continue
                    
                scale_factor = scale_factors[comp_name]
                combined_arrays.append(array)
                weights_array.append(np.full(len(array), fill_value=scale_factor, dtype=np.float32))
                
                lookup_name = comp_name.lower()
                mc_code = COMPONENT_TO_MC_CODE.get(lookup_name, -2)
                    
                component_cats.extend([mc_code] * len(array))
            
            combined[var_name] = np.concatenate(combined_arrays) if combined_arrays else np.array([])
            if var_name == list(variables.keys())[0]:
                combined_component_cats = np.array(component_cats)
                combined_weights = np.concatenate(weights_array) if weights_array else np.array([])
        
        return combined, combined_component_cats, combined_weights
    
    def _save_datacard_snapshot(self, snapshot_path: str, fit_result: Any, 
                                combine_pdf: Any, fit_range_mom: Tuple[float, float],
                                fit_range_time: Tuple[float, float], output_card_path: Optional[str] = None):
        """
        Save fit results in datacard-compatible snapshot format.
        
        Creates an NPZ file with:
        - fit_ranges: [mom_min, mom_max, time_min, time_max]
        - val_<param>: Fitted parameter values (N_CE, N_DIO, etc.)
        - err_<param>: Parameter uncertainties
        - Shape parameters from mom_components (CE DSCB mu/sigma/alpha/n, Cosmic/RPC c1/c2, decay rates)
        - systematics: (if input datacard provided) Systematic uncertainties from input card for traceability
        
        This snapshot can be loaded by generate_datacard.py to create data cards.
        Systematics are preserved from the input card (Combine-style workflow).
        
        If output_card_path is provided, automatically generates YAML datacard from snapshot.
        
        Args:
            snapshot_path: Where to save the snapshot
            fit_result: Fit result object
            combine_pdf: Combined PDF object
            fit_range_mom: Momentum range tuple
            fit_range_time: Time range tuple
            output_card_path: Optional path to save generated YAML datacard
        """
        export_dict = {
            'fit_ranges': np.array([fit_range_mom[0], fit_range_mom[1], 
                                   fit_range_time[0], fit_range_time[1]])
        }
        
        # Extract fitted parameter values
        if fit_result and hasattr(fit_result, 'params'):
            for p, p_data in fit_result.params.items():
                p_name = p.name if hasattr(p, 'name') else str(p)
                export_dict[f"val_{p_name}"] = float(p_data.get('value', 0.0))
                err_val = p_data.get('error', 0.0)
                if err_val is not None:
                    export_dict[f"err_{p_name}"] = float(err_val)
        
        # Add shape parameters from mom_components (physics constants)
        # These define the fixed CE DSCB parameters, Cosmic/RPC polynomial coefficients, etc.
        try:
            for component_name, component_config in mom_components.items():
                pars = component_config.get('pars', {})
                for par_name, par_values in pars.items():
                    # par_values is typically (nominal, lower, upper) tuple
                    # Use the first value (nominal) as the parameter value
                    if isinstance(par_values, (tuple, list)) and len(par_values) > 0:
                        nominal_value = float(par_values[0])
                    else:
                        nominal_value = float(par_values)
                    
                    # Store with component prefix to avoid conflicts
                    export_dict[f"val_{par_name}_{component_name}"] = nominal_value
                    export_dict[f"err_{par_name}_{component_name}"] = 0.0  # Shape params are fixed
                    
                    # Also store without component suffix for CE (backward compatibility)
                    if component_name == 'CE':
                        export_dict[f"val_{par_name}_CE"] = nominal_value
                        export_dict[f"err_{par_name}_CE"] = 0.0
        except Exception as e:
            self.logger.log(f"Warning: Could not extract shape parameters from mom_components: {e}", "warn")
        
        # Extract shape parameters from PDFs if available (may override above if fitted)
        if combine_pdf and hasattr(combine_pdf, 'pdfs'):
            for sub_pdf in combine_pdf.pdfs:
                if hasattr(sub_pdf, 'get_params'):
                    for shape_param in sub_pdf.get_params():
                        p_name = shape_param.name if hasattr(shape_param, 'name') else str(shape_param)
                        if f"val_{p_name}" not in export_dict:
                            try:
                                param_val = float(shape_param.numpy()) if hasattr(shape_param, 'numpy') else float(shape_param)
                                export_dict[f"val_{p_name}"] = param_val
                                export_dict[f"err_{p_name}"] = 0.0
                            except:
                                pass
        
        # Add decay rate constants (physics constants, not fit parameters)
        export_dict["val_decay_rate_mu"] = -0.001131  # Muon decay rate
        export_dict["val_decay_rate_pi"] = -0.0553     # Pion decay rate
        
        # Store systematics from input datacard if available (full traceability)
        if self.input_card and hasattr(self.input_card, 'systematics'):
            export_dict["systematics"] = np.array([self.input_card.systematics], dtype=object)
            self.logger.log(f"Stored {len(self.input_card.systematics)} systematics from input card", "info")
        
        # Save as NPZ
        snapshot_path = Path(snapshot_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(snapshot_path, **export_dict)
        
        self.logger.log(f"Datacard snapshot saved: {snapshot_path}", "success")
        self.logger.log(f"  Contains {len(export_dict)} entries", "info")
        self.logger.log(f"  Shape parameters: CE DSCB (mu, sigma, alphaL/R, nL/R), Cosmic/RPC (c1, c2), decay rates", "info")
        
        # Log the key yields for verification
        yield_keys = ['N_CE', 'N_DIO', 'N_Cosmic', 'N_RPC', 'N_RMC']
        self.logger.log("  Fitted yields:", "info")
        for key in yield_keys:
            if f'val_{key}' in export_dict:
                yield_val = export_dict[f'val_{key}']
                err_val = export_dict.get(f'err_{key}', 0.0)
                self.logger.log(f"    {key}: {yield_val:.4f} ± {err_val:.4f}", "info")
        
        # Automatically generate YAML datacard from snapshot if output path provided
        if output_card_path:
            try:
                output_card_path = Path(output_card_path)
                output_card_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Generate datacard from the snapshot we just saved
                card = datacard_from_snapshot(str(snapshot_path), output_path=str(output_card_path))
                self.logger.log(f"Datacard YAML generated: {output_card_path}", "success")
                self.logger.log(f"  Datacard name: {card.name}", "info")
            except Exception as e:
                self.logger.log(f"Failed to generate YAML datacard: {e}", "error")
    
    def fit_mom_1d(self, fit_range: Tuple[float, float] = (100, 110),
                   plot: bool = True, minos: bool = False, use_constraints: bool = False, 
                   constraints_dir: str = 'uncertainties/outputs', components_to_fit: Optional[List[str]] = None,
                   signal_component: str = 'CE') -> Any:
        """Perform 1D momentum fit on pre-cut parquet data."""
        if not self.components:
            self.logger.log("No components loaded", "error")
            return None
        
        comp_names = list(self.components.keys())
        variables = {'mom': []}
        scale_factors = {}
        component_data_for_plot = {}  
        
        self.logger.log(f"Components loaded for 1D fit: {comp_names}", "info")
        
        for comp_name in comp_names:
            data = self.components[comp_name]
            mom_array = data.get('momentum')
            
            if mom_array is not None and len(mom_array) > 0:
                # Apply fit range cut
                in_range = (mom_array >= fit_range[0]) & (mom_array <= fit_range[1])
                mom_array = mom_array[in_range]
                
                variables['mom'].append(mom_array)
                component_data_for_plot[comp_name] = mom_array
                
                target = self.component_yields.get(comp_name.lower(), 0.0)
                scale_factor = (target / len(mom_array)) if (len(mom_array) > 0 and target is not None and target > 0) else 0.0
            else:
                variables['mom'].append(None)
                component_data_for_plot[comp_name] = None
                scale_factor = 0.0
                
            scale_factors[comp_name] = scale_factor
            self.scaled_components[comp_name] = (data, scale_factor)
            self.logger.log(f"Normalized {comp_name:<10} | Window entries: {len(mom_array) if mom_array is not None else 0:<6} | Weight: {scale_factor:.6f}", "info")
        
        combined_data, component_cats, combined_weights = self._build_combined_data(variables, scale_factors)
        mom_mag = combined_data['mom']
        
        self.verify_dataset_scaling_truth(component_cats, combined_weights)
        
        fit_tuple = Unbinned_fit_mom(
            mom_mag=mom_mag, count_particle_types=component_cats, fit_range_low=fit_range[0], fit_range_hi=fit_range[1],
            weights=combined_weights, plot_truth=False, verbose=self.verbosity, minos=minos,
            plot_NLL=False, plot_results=False, constraints_dir=constraints_dir if use_constraints else None,
            components_to_fit=components_to_fit, signal_component=signal_component
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
        
        # Export datacard snapshot if requested
        try:
            import builtins
            if 'args' in globals() and hasattr(globals()['args'], 'datacard_snapshot') and globals()['args'].datacard_snapshot:
                current_args = globals().get('args', None) or getattr(builtins, 'args', None)
                if current_args and hasattr(current_args, 'datacard_snapshot') and current_args.datacard_snapshot:
                    output_card = getattr(current_args, 'output_card', None)
                    self._save_datacard_snapshot(
                        snapshot_path=current_args.datacard_snapshot,
                        fit_result=result,
                        combine_pdf=combine_pdf,
                        fit_range_mom=fit_range,
                        fit_range_time=(0, 1000),  # Default for 1D fit
                        output_card_path=output_card
                    )
        except Exception as e:
            self.logger.log(f"Datacard snapshot export failed (non-critical): {e}", "warning")
        
        return result
    
    def fit_mom_time_2d(self, fit_range_mom: Tuple[float, float] = (100, 110), 
                       fit_range_time: Tuple[float, float] = (475, 1650),
                       plot: bool = True, use_constraints: bool = False, 
                       constraints_dir: str = 'uncertainties/outputs', components_to_fit: Optional[List[str]] = None,
                       signal_component: str = 'CE') -> Any:
        """Perform 2D momentum-time fit on pre-cut parquet data."""
        if not self.components:
            self.logger.log("No components loaded", "error")
            return None
        
        comp_names = list(self.components.keys())
        variables = {'mom': [], 'time': []}
        scale_factors = {}
        component_mom_for_plot = {}
        component_time_for_plot = {}
        
        for comp_name in comp_names:
            data = self.components[comp_name]
            
            mom_array = data.get('momentum')
            time_array = data.get('time')
            
            if mom_array is not None and time_array is not None and len(mom_array) > 0:
                # Apply 2D range cuts
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
                variables['mom'].append(None)
                variables['time'].append(None)
                component_mom_for_plot[comp_name] = None
                component_time_for_plot[comp_name] = None
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
            constraints_dir=constraints_dir if use_constraints else None, components_to_fit=components_to_fit,
            signal_component=signal_component
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
        try:
            import builtins
            if 'args' in globals() and hasattr(globals()['args'], 'export_npz') and globals()['args'].export_npz:
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
                            for shape_param in sub_pdf.get_params():
                                p_name = shape_param.name
                                if f"val_{p_name}" not in fit_param_vals:
                                    fit_param_vals[f"val_{p_name}"] = float(shape_param.numpy())
                                    fit_param_errs[f"err_{p_name}"] = 0.0

                    # 3. Save everything down to a comprehensive dictionary
                    export_dict = {
                        'mom': mom_mag,
                        'time': times,
                        'weights': combined_weights,
                        'categories': component_cats,
                        'fit_ranges': np.array([fit_range_mom[0], fit_range_mom[1], fit_range_time[0], fit_range_time[1]]),
                    }
                    export_dict.update(fit_param_vals)
                    export_dict.update(fit_param_errs)

                    np.savez(export_path, **export_dict)
                    self.logger.log(f"Config exported. Packed {len(fit_param_vals)} model parameters into: {export_path}", "success")
        except Exception as e:
            self.logger.log(f"Export failed (non-critical): {e}", "warning")
        
        # Export datacard snapshot if requested
        try:
            import builtins
            if 'args' in globals() and hasattr(globals()['args'], 'datacard_snapshot') and globals()['args'].datacard_snapshot:
                current_args = globals().get('args', None) or getattr(builtins, 'args', None)
                if current_args and hasattr(current_args, 'datacard_snapshot') and current_args.datacard_snapshot:
                    output_card = getattr(current_args, 'output_card', None)
                    self._save_datacard_snapshot(
                        snapshot_path=current_args.datacard_snapshot,
                        fit_result=result,
                        combine_pdf=combine_pdf,
                        fit_range_mom=fit_range_mom,
                        fit_range_time=fit_range_time,
                        output_card_path=output_card
                    )
        except Exception as e:
            self.logger.log(f"Datacard snapshot export failed (non-critical): {e}", "warning")
        
        return result


def main(args):
    builder = ParquetFitBuilder(verbosity=args.verbosity, jobs=args.jobs)
    
    # Load input datacard if provided (like Combine workflow)
    if args.card:
        builder.input_card = DataCard.from_yaml(args.card)
        builder.logger.log(f"Loaded input datacard from {args.card}", "info")
        card_yields = builder.input_card.get_expected_yields()
        builder.logger.log(f"  Expected yields from card: {card_yields}", "info")
        # Normalize keys to lowercase for consistency with component names
        normalized_yields = {k.lower(): v for k, v in card_yields.items()}
        # Apply card yields to component_yields (these are the source of truth)
        builder.set_component_yields(normalized_yields)
    
    # Parse component files: expect format "component_name=/path/to/file.parquet"
    component_files = {}
    if args.components:
        for item in args.components:
            parts = item.split('=')
            if len(parts) == 2:
                component_files[parts[0]] = parts[1]
            else:
                print(f"Warning: Invalid component specification: {item}. Expected format: name=/path/file.parquet")
    
    if not component_files:
        print("Error: No components specified. Use --components name=/path/file.parquet [name2=/path/file2.parquet ...]")
        sys.exit(1)
    
    # Parse custom yields if provided
    if args.yields:
        yields_dict = {}
        for item in args.yields:
            parts = item.split('=')
            if len(parts) == 2:
                try:
                    value = float(parts[1]) if parts[1].lower() != 'none' else None
                    yields_dict[parts[0]] = value
                except ValueError:
                    print(f"Warning: Invalid yield value for {parts[0]}: {parts[1]}")
            else:
                print(f"Warning: Invalid yield specification: {item}")
        if yields_dict:
            builder.set_component_yields(yields_dict)
    
    # Load parquet components
    builder.load_parquet_components(
        component_files,
        mom_column=args.mom_column,
        time_column=args.time_column,
        fit_range=(args.fit_range_lo, args.fit_range_hi),
        time_range=(args.time_range_lo, args.time_range_hi)
    )
    
    # Run fits
    if args.fit_type in ['1d', 'both']:
        builder.fit_mom_1d(
            fit_range=(args.fit_range_lo, args.fit_range_hi), 
            plot=not args.no_plot, 
            minos=args.minos, 
            constraints_dir=args.constraints_dir
        )
    
    if args.fit_type in ['2d', 'both']:
        builder.fit_mom_time_2d(
            fit_range_mom=(args.fit_range_lo, args.fit_range_hi), 
            fit_range_time=(args.time_range_lo, args.time_range_hi), 
            plot=not args.no_plot, 
            constraints_dir=args.constraints_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parquet-based Unbinned Asimov Fitter")
    parser.add_argument('--fit-type', choices=['1d', '2d', 'both'], default='2d',
                       help='Type of fit to perform')
    parser.add_argument('--components', nargs='+', required=True,
                       help='Components to fit: component_name=/path/to/file.parquet [name2=/path/file2.parquet ...]')
    parser.add_argument('--mom-column', type=str, default='momentum',
                       help='Name of momentum column in parquet files')
    parser.add_argument('--time-column', type=str, default='time',
                       help='Name of time column in parquet files')
    parser.add_argument('--fit-range-lo', type=float, default=100,
                       help='Lower momentum fit range')
    parser.add_argument('--fit-range-hi', type=float, default=110,
                       help='Upper momentum fit range')
    parser.add_argument('--time-range-lo', type=float, default=475,
                       help='Lower time fit range')
    parser.add_argument('--time-range-hi', type=float, default=1650,
                       help='Upper time fit range')
    parser.add_argument('--yields', nargs='+', default=None,
                       help='Target yields: component_name=value [name2=value2 ...]')
    parser.add_argument('--constraints-dir', type=str, default='uncertainties/outputs',
                       help='Directory for constraint files')
    parser.add_argument('--minos', action='store_true',
                       help='Use Minos for error calculation')
    parser.add_argument('--no-plot', action='store_true',
                       help='Do not generate plots')
    parser.add_argument('--jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--verbosity', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('--export-npz', type=str, default=None,
                       help='Export fit results to NPZ file')
    parser.add_argument('--datacard-snapshot', type=str, default=None,
                       help='Save fit snapshot for datacard generation (NPZ file)')
    parser.add_argument('--output-card', type=str, default='output_card.yaml',
                       help='Output YAML datacard path (auto-generated from snapshot). Default: output_card.yaml')
    parser.add_argument('--card', type=str, default=None,
                       help='Input YAML datacard with expected yields and systematics (like Combine)')
    
    args = parser.parse_args()
    main(args)
