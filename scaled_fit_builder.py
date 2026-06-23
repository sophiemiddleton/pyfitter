"""
Build and execute fits on scaled MC component samples.

This module enables fitting to highest-stats scaled MC samples instead of single
combined datasets. It reuses the existing fitting, PDF building, and plotting
infrastructure while handling the scaling and component preparation.

Key Features:
- Loads separate MC component files with independent cuts
- Scales components by expected yields (physics-based or configurable)
- Extracts fit variables while preserving component information
- Integrates seamlessly with existing Unbinned_fit_mom and Unbinned_2d_fit_mom_time

Usage Example:
    builder = ScaledFitBuilder(verbosity=1, jobs=4)
    
    # Load and scale components
    component_files = {
        'dio': 'path/to/dio.txt',
        'cosmic': 'path/to/cosmic.txt',
        'rpc_ext': 'path/to/rpc_ext.txt',
        ...
    }
    builder.load_and_scale_components(component_files, sign='minus')
    
    # Run 1D momentum fit
    result = builder.fit_mom_1d(
        variable='recomom_ttfront',
        fit_range=(97, 110),
        plot=True
    )
    
    # Or run 2D fit
    result = builder.fit_mom_time_2d(
        mom_variable='recomom_ttfront',
        time_variable='tracktime_ttfront',
        fit_range_mom=(97, 110),
        fit_range_time=(475, 1650),
        plot=True
    )
    python scaled_fit_builder.py --fit-type 2d --components dio=file_lists/DIOtail95_MDC2025an_best_nomix.txt cosmic=file_lists/Cosimcs_MDC2025an_nomix.txt rpc_ext=file_lists/ExtRPC_MDC2025an_nomix.txt rpc_int=file_lists/IntRPC_MDC2025an_nomix.txt ce=file_lists/CeMLL_MDC2025an_best_nomix.txt --variable recomom_ttfront --fit-range 97  110 --time-range 450 1650 --jobs 16
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


# Map component names to their MC truth process codes (from process.py)
# These codes allow the plotting infrastructure to identify and overlay true MC distributions
COMPONENT_TO_MC_CODE = {
    'dio': 166,           # DIO process
    'ipa': 0,             # Inclusive pion absorption
    'ce': 168,            # Charged lepton exchange (CEM)
    'cem': 168,           # CEM variant
    'cep': 176,           # CE+ variant
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
    """Plot fit results with pre-computed true MC component histograms and pull plot.
    
    This function creates a two-panel plot with:
    - Upper panel: stacked MC histograms with fit overlay and individual component lines
    - Lower panel: pull plot (data - fit) / error
    
    Args:
        mom_mag: numpy array of momentum values to fit
        combine_pdf: zfit SumPDF object from the fit (optional, for overlaying fit curve)
        fit_result: Result object from Unbinned_fit_mom or fit function
        component_data_dict: Dict mapping component_name -> numpy array of values
        component_names: List of component names in order
        scale_factors_dict: Dict mapping component_name -> scale factor
        fit_range: Tuple (lo, hi) for fit range
        nbins: Number of bins for histogram
        output_file: Optional output file path for plot
        title: Plot title
        verbosity: Verbosity level
        real_data_mom_mag: Optional numpy array of real data momentum values to overlay as points
        
    Returns:
        matplotlib figure object
    """
    logger = Logger(print_prefix="[plot_fit_with_true_shapes]", verbosity=verbosity)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.05)
    
    # Define bin structure
    bin_edges = np.linspace(fit_range[0], fit_range[1], nbins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (fit_range[1] - fit_range[0]) / nbins
    
    # Compute histograms for each component
    component_hists = {}
    logger.log("="*70, "info")
    logger.log("Plotting component histograms (AFTER CUT EXTRACTION)", "info")
    logger.log("="*70, "info")
    
    for comp_name in component_names:
        if comp_name in component_data_dict:
            data = component_data_dict[comp_name]
            if data is not None and len(data) > 0:
                counts, _ = np.histogram(data, bins=bin_edges)
                
                # Log statistics for this component
                logger.log(f"{comp_name}:", "info")
                logger.log(f"  Events in extracted data: {len(data)}", "info")
                logger.log(f"  Data range: {np.min(data):.2f} - {np.max(data):.2f} MeV/c", "info")
                logger.log(f"  Mean: {np.mean(data):.2f} ± {np.std(data):.2f}", "info")
                logger.log(f"  Histogram counts: {np.sum(counts):.0f}", "info")
                
                # Apply scaling if provided
                if scale_factors_dict and comp_name in scale_factors_dict:
                    scale_factor = scale_factors_dict[comp_name]
                    counts = counts * scale_factor
                    logger.log(f"  Scale factor: {scale_factor:.3f}", "info")
                    logger.log(f"  Scaled histogram: {np.sum(counts):.0f}", "info")
                
                component_hists[comp_name] = counts
            else:
                logger.log(f"{comp_name}: NO DATA", "warning")
                component_hists[comp_name] = np.zeros(nbins)
        else:
            logger.log(f"{comp_name}: NOT IN DICT", "warning")
            component_hists[comp_name] = np.zeros(nbins)
    
    logger.log("="*70, "info")
    
    # Stack components and plot on upper panel
    bottom = np.zeros(nbins)
    
    # Define plot order (same as in plot_scaled_mom.py)
    desired_order = ['rpc_ext', 'rpc_int','cosmic', 'dio',  'rmc_ext', 'rmc_int', 'ipa', 'ce']
    
    for comp_name in desired_order:
        if comp_name in component_hists and np.sum(component_hists[comp_name]) > 0:
            display_info = COMPONENT_DISPLAY.get(comp_name, {'color': 'C0', 'label': comp_name})
            ax1.bar(bin_centers, component_hists[comp_name], width=bin_width, 
                   bottom=bottom, label=display_info['label'], 
                   color=display_info['color'], alpha=0.8)
            bottom += component_hists[comp_name]
    
    # Total histogram for pull calculation
    total_hist = bottom.copy()
    
    # Plot real data as points with error bars if provided
    if real_data_mom_mag is not None and len(real_data_mom_mag) > 0:
        try:
            # Histogram real data (unscaled)
            data_counts, _ = np.histogram(real_data_mom_mag, bins=bin_edges)
            
            # Calculate Poisson errors
            data_errors = np.sqrt(data_counts)
            
            # Plot as black points with error bars
            ax1.errorbar(bin_centers, data_counts, yerr=data_errors, 
                        fmt='ko', markersize=6, capsize=3, capthick=1.5,
                        elinewidth=1.5, label='Data (MDS)', zorder=5)
            
            logger.log(f"Overlaid real data: {len(real_data_mom_mag)} events", "info")
        except Exception as e:
            logger.log(f"Could not overlay real data: {e}", "warning")
    
    # Plot the fit model if provided
    fit_vals_for_plot = None
    if combine_pdf is not None:
        try:
            logger.log("Attempting to plot fit model curve", "debug")
            
            # Create evaluation points for the PDF
            mom_vals = np.linspace(fit_range[0], fit_range[1], 200)
            logger.log(f"Created {len(mom_vals)} evaluation points", "debug")
            
            # Get the observable from the PDF itself
            pdf_obs = combine_pdf.obs
            logger.log(f"PDF observable: {pdf_obs}", "debug")

            # Build evaluation tensor for 1D or 2D PDFs
            if isinstance(pdf_obs, tuple):
                n_obs = len(pdf_obs)
            elif isinstance(pdf_obs, (list, set)):
                n_obs = len(list(pdf_obs))
            else:
                n_obs = 1

            if n_obs == 1:
                obs_space = zfit.Space(pdf_obs, limits=fit_range)
                eval_tensor = tf.constant(mom_vals, dtype=tf.float32)
                logger.log(f"Created 1D space with limits {fit_range}", "debug")
            else:
                # For 2D fits, compute a true 1D projection by marginalizing over the other axis.
                obs_space = combine_pdf.space
                obs_names = list(pdf_obs)
                scan_idx = 1 if plot_obs == 'time' and len(obs_names) > 1 else 0
                eval_tensor = None

                limits_low, limits_high = obs_space.rect_limits
                lo_arr = np.asarray(limits_low, dtype=float).reshape(-1)
                hi_arr = np.asarray(limits_high, dtype=float).reshape(-1)

                def _project_pdf_1d(pdf_obj):
                    """Project nD PDF to 1D by integrating over the non-scan observable."""
                    if n_obs != 2:
                        # Fallback for unexpected dimensions: evaluate a conditional slice.
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

                    # Build a full mesh and evaluate once, then integrate numerically.
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

                logger.log(
                    f"Using 2D marginal projection for '{plot_obs}' (integrating over the other observable)",
                    "info"
                )

            # Evaluate PDF at these points or projected grid
            if n_obs == 1:
                pdf_vals = combine_pdf.pdf(eval_tensor, norm_range=obs_space)
            else:
                pdf_vals = _project_pdf_1d(combine_pdf)
            
            # Convert to numpy
            if hasattr(pdf_vals, 'numpy'):
                pdf_vals = pdf_vals.numpy()
            else:
                pdf_vals = np.array(pdf_vals)
            
            logger.log(f"PDF values: min={np.min(pdf_vals):.4f}, max={np.max(pdf_vals):.4f}", "debug")
            
            # Normalize projected density to unit area before scaling to counts
            area = np.trapz(pdf_vals, mom_vals)
            if area > 0:
                pdf_vals = pdf_vals / area

            # Scale by bin width and total events to match histogram scaling
            total_events = len(mom_mag)
            scaled_pdf = pdf_vals * total_events * bin_width
            fit_vals_for_plot = (mom_vals, scaled_pdf)
            
            logger.log(f"Scaled PDF: min={np.min(scaled_pdf):.4f}, max={np.max(scaled_pdf):.4f}", "debug")
            
            # Plot as a line on upper panel
            ax1.plot(mom_vals, scaled_pdf, 'k-', linewidth=2.5, label='Total Fit', zorder=10)
            logger.log("Successfully plotted fit model curve", "info")
            
            # Try to plot individual component PDFs if this is a SumPDF
            if hasattr(combine_pdf, 'pdfs') and fit_result is not None:
                try:
                    component_colors = {
                        'CE': '#ff8000',
                        'DIO': '#e377c2',
                        'Cosmic': '#1f77b4',
                        'RPC': '#2ca02c',
                    }

                    fitline_colors = {
                        'CE': 'red',
                        'DIO': 'green',
                        'Cosmic': 'cyan',
                        'RPC': '#ff7f0e',
                    }
                    
                    component_names_list = ['CE', 'Cosmic', 'RPC', 'DIO']
                    
                    # Access the pdfs from the SumPDF
                    pdfs_list = combine_pdf.pdfs
                    
                    logger.log(f"Found {len(pdfs_list)} component PDFs", "debug")
                    
                    # Extract fitted yields from fit result parameters
                    fitted_yields = {}
                    if hasattr(fit_result, 'params'):
                        for param in fit_result.params:
                            # Get parameter name as string
                            param_name = str(param.name) if hasattr(param, 'name') else str(param)
                            param_val = param
                            
                            # Look for N_<ComponentName> parameters
                            if param_name.startswith('N_'):
                                comp = param_name.replace('N_', '')
                                if hasattr(param_val, 'numpy'):
                                    fitted_yields[comp] = float(param_val.numpy())
                                elif hasattr(param_val, 'value'):
                                    fitted_yields[comp] = float(param_val.value())
                                else:
                                    fitted_yields[comp] = float(param_val)
                        
                        logger.log(f"Extracted fitted yields: {fitted_yields}", "debug")
                    
                    # For each PDF, evaluate it with the fitted yield
                    for i, pdf in enumerate(pdfs_list):
                        try:
                            # Get component name
                            comp_name = getattr(pdf, 'name', None)
                            if comp_name is None or comp_name not in fitted_yields:
                                # Try to match by index
                                if i < len(component_names_list):
                                    comp_name = component_names_list[i]
                            
                            # Get the yield value
                            if comp_name in fitted_yields:
                                yield_val = fitted_yields[comp_name]
                            else:
                                logger.log(f"Could not find yield for component {comp_name}", "debug")
                                continue
                            
                            # Evaluate this component PDF (1D direct or marginalized 2D projection)
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
                            
                            # Scale by yield and bin width
                            scaled_comp_pdf = comp_pdf_vals * yield_val * bin_width
                            
                            comp_color = fitline_colors.get(comp_name, f'C{i}')
                            
                            ax1.plot(mom_vals, scaled_comp_pdf, '--', linewidth=1.5, 
                                label=comp_name, color=comp_color, alpha=0.95, zorder=12)
                            logger.log(f"Plotted {comp_name} component (yield={yield_val:.1f})", "debug")
                            
                        except Exception as comp_e:
                            logger.log(f"Could not plot component {i}: {comp_e}", "debug")
                    
                except Exception as comp_plot_e:
                    logger.log(f"Could not plot individual components: {comp_plot_e}", "debug")
            
        except Exception as e:
            logger.log(f"Could not plot fit model: {type(e).__name__}: {e}", "warning")
            import traceback
            logger.log(traceback.format_exc(), "debug")
    
    # Formatting upper panel
    ax1.set_ylabel('Events', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
    ax1.set_yscale('log')
    ax1.set_ylim(ymin=1,ymax= 1500)
    ax1.legend(loc='upper right', fontsize=FONTS['legend']['size'])
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(axis='y', labelsize=FONTS['tick']['size'])
    ax1.set_xlim(fit_range)
    
    # Plot pull on lower panel if fit values are available
    if fit_vals_for_plot is not None:
        mom_vals, scaled_pdf = fit_vals_for_plot
        
        # Interpolate fit values at bin centers
        fit_at_bins = np.interp(bin_centers, mom_vals, scaled_pdf)
        
        # Calculate pull: (data - fit) / error
        err = []
        dev = []
        for i, data_val in enumerate(total_hist):
            fit_val = fit_at_bins[i]
            if data_val > 0:
                # Total error from data and fit
                data_err = np.sqrt(data_val)
                fit_err = np.sqrt(fit_val)
                total_err = np.sqrt(data_err**2 + fit_err**2)
                if total_err > 0:
                    err.append(total_err / data_val if data_val > 0 else 0)
                    dev.append((fit_val - data_val) / total_err)
                else:
                    err.append(0)
                    dev.append(0)
            else:
                err.append(0)
                dev.append(0)
        
        # Plot pull points
        ax2.errorbar(bin_centers, dev, yerr=err, color='None', marker='+', 
                    markerfacecolor='black', ecolor='black', capsize=3, markersize=8)
        
        # Add horizontal line at zero
        ax2.axhline(0, color='red', linewidth=1.5)
        
        # Formatting lower panel
        ax2.set_ylabel(r'Pull [$\sigma$]', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
        ax2.set_xlabel('Reconstructed Momentum [MeV/c]', fontsize=FONTS['label']['size'], fontweight=FONTS['label']['weight'])
        ax2.yaxis.set_ticks(np.arange(-5, 6, 2))
        ax2.set_xlim(fit_range)
        ax2.tick_params(axis='both', labelsize=FONTS['tick']['size'])
    else:
        ax2.axis('off')
    
    # Add logo and metadata
    logo_to_use = "mu2e_logo_oval.png"
    try:
        logo = Image.open(logo_to_use)
        ax_logo = fig.add_axes([0.02, 0.93, 0.1, 0.09])
        ax_logo.imshow(logo)
        ax_logo.axis('off')
    except Exception as e:
        logger.log(f"Could not load logo: {e}", "debug")
    
    # Add title with metadata
    fig.text(0.15, 0.98, "Mu2e Simulation (Preliminary - Summer 2026)", 
            fontsize=FONTS['label']['size'], fontweight='bold', 
            ha='left', va='top', zorder=100)
    
    # Remove grid lines from upper panel
    ax1.grid(False)
    if fit_vals_for_plot is not None:
        ax2.grid(False)
    
    # Save if requested
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.log(f"Saved plot to {output_file}", "info")
    
    return fig


class ScaledFitBuilder:
    """Build and fit scaled MC component samples.
    
    Orchestrates loading, scaling, and fitting of separate MC component files
    to enable fits on highest-stats sample sets rather than single combined datasets.
    """
    
    def __init__(self, verbosity=1, jobs=1):
        """Initialize the builder.
        
        Args:
            verbosity: Verbosity level for logging (0-3)
            jobs: Number of parallel jobs for file processing
        """
        self.logger = Logger(print_prefix="[ScaledFitBuilder]", verbosity=verbosity)
        self.verbosity = verbosity
        self.jobs = jobs
        self.components = {}  # Dict[component_name, data]
        self.scaled_components = {}  # Dict[component_name, scaled_data]
        self.component_yields = {}  # Expected yields for scaling
        self.data = None  # Optional real data for overlay
        self.fit_range_lo = 97  # Default momentum range passed to analyze.py cuts
        self.fit_range_hi = 110  # Default momentum range passed to analyze.py cuts
        
        # Default component yields after standard cuts (physics expectations)
        self.default_yields = { 
            'dio': 1427,           # DIO > 97 MeV
            'cosmic': 333,         # Cosmics
            'rpc_ext': 5,         # RPC External
            'rpc_int': 4,         # RPC Internal
            'rmc_ext': None,         # RMC External (auto-scale if not provided)
            'rmc_int': None,         # RMC Internal (auto-scale if not provided)
            'ipa': None,             # IPA/CE (auto-scale if not provided)
            'ce': 73                 # CE/signal
        }

        self.component_yields = self.default_yields.copy()
        
        self.logger.log("Initialized", "info")
    
    def set_component_yields(self, yields_dict: Dict[str, Optional[float]]):
        """Set expected yields for component scaling.
        
        Args:
            yields_dict: Dict mapping component names to expected yields.
                        Yields set to None will auto-scale relative to other components.
                        Example:
                            {
                                'dio': 5.87e3,
                                'cosmic': 500.5,
                                'ce': 65,
                                'rpc_ext': None,  # auto-scale
                            }
        """
        self.component_yields.update(yields_dict)
        self.logger.log(f"Set component yields: {self.component_yields}", "info")

    def _save_scan_npz(self, out_path: str, mom_mag: np.ndarray, times: Optional[np.ndarray] = None):
        """Save fit input arrays in sensitivity-scan compatible NPZ format.

        The scanner in sensitivity_scan/run_sens_scan.py looks for key 'mom_mag'.
        For 2D workflows we additionally store 'time'.
        """
        try:
            payload = {'mom_mag': np.asarray(mom_mag)}
            if times is not None:
                payload['time'] = np.asarray(times)
            np.savez_compressed(out_path, **payload)
            if times is None:
                self.logger.log(f"Saved scanner NPZ: {out_path} (keys: mom_mag)", "info")
            else:
                self.logger.log(f"Saved scanner NPZ: {out_path} (keys: mom_mag, time)", "info")
        except Exception as e:
            self.logger.log(f"Failed to save scanner NPZ {out_path}: {e}", "warning")
    
    def _process_file_list(self, file_path: str, sign: str = "minus", 
                          location: str = "disk", component_name: str = "unknown",
                          mom_lo: Optional[float] = None, mom_hi: Optional[float] = None) -> Optional[ak.Array]:
        """Process a single file list with standard cuts applied.
        
        Args:
            file_path: Path to a file list (text file containing ROOT file paths)
            sign: Charge sign ("minus" or "plus")
            location: Location of files ('disk' or 'local')
            component_name: Name of the component being processed (for logging)
            mom_lo: Lower momentum cut (passed to analyze.py). If None, uses instance default
            mom_hi: Upper momentum cut (passed to analyze.py). If None, uses instance default
            
        Returns:
            Processed awkward array after cuts, or None if processing failed
        """
        # Use provided values or fall back to instance defaults
        if mom_lo is None:
            mom_lo = self.fit_range_lo
        if mom_hi is None:
            mom_hi = self.fit_range_hi
        
        self.logger.log(f"Processing file list: {file_path}", "info")
        self.logger.log(f"  Momentum cut range for analyze.py: {mom_lo} - {mom_hi} MeV/c", "info")
        
        # Standard cut switches (same as plot_scaled_mom.py)
        cut_switches = [
            True,   # 0: is_reco_electron
            True,   # 1: has_downstream
            True,   # 2: has_trk_front
            True,   # 3: good_trkqpid
            True,   # 4: good_trkqual
            True,   # 5: within_t0err
            True,   # 6: has_hits
            False,  # 7: within_lhr_maxl
            False,  # 8: within_d0
            False,  # 9: within_pitch_angle
            True,   # 10: has_st
            True,   # 11: no_opa
            True,   # 12: no_crv_veto
            True,   # 13: no_crv_quality
            True,   # 14: no_crv_timewindow
            True,   # 15: pz/pt
            True,   # 16: triggers
            True,   # 17: in_mom_range 
            False,  # 18: within_t0_early
            False,  # 19: no_reflected
            False,  # 20: within_t0
            False   # 21: signal_region_cut
        ]
        
        cut_names = [
            "is_reco_electron", "has_downstream", "has_trk_front", "good_trkqpid",
            "good_trkqual", "within_t0err", "has_hits", "within_lhr_maxl",
            "within_d0", "within_pitch_angle", "has_st", "no_opa",
            "no_crv_veto", "no_crv_quality", "no_crv_timewindow", "pz/pt",
            "triggers", "in_mom_range", "within_t0_early", "no_reflected",
            "within_t0", "signal_region"
        ]

        
        try:
            self.logger.log("="*70, "info")
            self.logger.log(f"Processing file list: {file_path}", "info")
            self.logger.log("="*70, "info")
            
            # Log which cuts are being applied
            active_cuts = [cut_names[i] for i, active in enumerate(cut_switches) if active]
            inactive_cuts = [cut_names[i] for i, active in enumerate(cut_switches) if not active]
            self.logger.log(f"ACTIVE cuts ({len(active_cuts)}): {', '.join(active_cuts)}", "info")
            self.logger.log(f"INACTIVE cuts ({len(inactive_cuts)}): {', '.join(inactive_cuts)}", "info")
            
            processor = AnaProcessor(
                file_list_path=file_path,
                jobs=self.jobs,
                cuts=cut_switches,
                location=location,
                mom_lo=mom_lo,
                mom_hi=mom_hi
            )
            
            # execute() returns a list of results from all files
            # Each result contains: (data, cut_flow) tuple
            results_list = processor.execute()
            
            if not results_list:
                self.logger.log(f"No results from file list: {file_path}", "warning")
                return None
            
            # Import combine_arrays from process module to merge results
            from process import combine_cut_flows
            
            # Extract data and cut flows separately
            # Results can be either:
            # - Raw awkward arrays (from newer Analyze.execute)
            # - Dicts with ["filtered_data"] and ["cut_stats"] keys (from older versions)
            data_list = []
            cut_flows = []
            
            for result in results_list:
                if result is not None:
                    # Check if result is a dict with filtered_data
                    if isinstance(result, dict) and "filtered_data" in result:
                        self.logger.log(f"Result is dict with filtered_data", "debug")
                        data_list.append(result["filtered_data"])
                        if "cut_stats" in result:
                            cut_flows.append(result["cut_stats"])
                    else:
                        # Result is raw awkward array
                        self.logger.log(f"Result is raw awkward array (length: {len(result)})", "debug")
                        data_list.append(result)
            
            # Print cut flow if available
            if cut_flows:
                try:
                    self.logger.log("", "info")
                    self.logger.log("="*70, "info")
                    self.logger.log(f"CUT FLOW for {component_name.upper()}", "info")
                    self.logger.log("="*70, "info")
                    combine_cut_flows(cut_flows, csv_basename=f"cutflow_{component_name}")
                    self.logger.log("="*70, "info")
                except Exception as cf_err:
                    self.logger.log(f"Could not print cut flows: {cf_err}", "debug")
            
            # Combine the filtered data from all files
            if data_list:
                import awkward as ak
                combined_data = ak.concatenate(data_list)
            else:
                combined_data = None
            
            if combined_data is not None:
                n_events = len(combined_data)
                self.logger.log(f"✓ Successfully processed: {n_events} events after cuts", "info")
                self.logger.log("="*70, "info")
                return combined_data
            else:
                self.logger.log(f"✗ No valid combined data from file list: {file_path}", "warning")
                self.logger.log("="*70, "info")
                return None
                
        except Exception as e:
            self.logger.log(f"Error processing file list {file_path}: {e}", "error")
            import traceback
            self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
            return None
    
    def load_and_scale_components(self, component_files: Dict[str, str], 
                                  sign: str = "minus", 
                                  location: str = "disk",
                                  normalize_to_max: bool = False,
                                  apply_cuts: bool = True,
                                  fit_range: Optional[Tuple[float, float]] = None):
        """Load and scale all MC component files.
        
        Args:
            component_files: Dict mapping component names to file list paths.
                            Example:
                                {
                                    'dio': 'file_lists/dio.txt',
                                    'cosmic': 'file_lists/cosmic.txt',
                                    'rpc_ext': 'file_lists/rpc_ext.txt',
                                    ...
                                }
            sign: Charge sign ("minus" or "plus")
            location: Location of files ('disk' or 'local')
            normalize_to_max: If True, scale all to max component size instead of using yields
            apply_cuts: If True, apply standard cuts; if False, load raw data
            fit_range: Optional tuple (lo, hi) for momentum range passed to analyze.py cuts.
                      Default: (97, 110). Set to (97, 110) for your momentum range, etc.
        """
        self.components = {}
        self.scaled_components = {}
        
        # Store fit range for use in _process_file_list and elsewhere
        if fit_range is not None:
            self.fit_range_lo = fit_range[0]
            self.fit_range_hi = fit_range[1]
            self.logger.log(f"Set momentum range for cuts: {self.fit_range_lo} - {self.fit_range_hi} MeV/c", "info")
        
        # Load all components
        for component_name, file_path in component_files.items():
            if file_path is None:
                self.logger.log(f"Skipping component '{component_name}' (no file)", "info")
                continue
            
            if not Path(file_path).exists():
                self.logger.log(f"File not found: {file_path}", "warning")
                continue
            
            if apply_cuts:
                data = self._process_file_list(file_path, sign=sign, location=location, 
                                            component_name=component_name,
                                            mom_lo=self.fit_range_lo, 
                                            mom_hi=self.fit_range_hi)
            else:
                # Load raw data without cuts for comparison
                try:
                    processor = AnaProcessor(
                        file_list_path=file_path,
                        jobs=self.jobs,
                        cuts=[False] * 22,  # No cuts
                        location=location,
                        mom_lo=self.fit_range_lo,
                        mom_hi=self.fit_range_hi
                    )
                    results_list = processor.execute()
                    if results_list:
                        from process import combine_arrays
                        data = combine_arrays(results_list)
                        self.logger.log(f"Loaded raw (no cuts) data: {len(data)} events", "info")
                    else:
                        data = None
                except Exception as e:
                    self.logger.log(f"Error loading raw data: {e}", "warning")
                    data = None
            
            if data is not None:
                self.components[component_name] = data
                self.logger.log(
                    f"Loaded component '{component_name}': {len(data)} events",
                    "info"
                )
            else:
                self.logger.log(f"Failed to load component '{component_name}'", "warning")
        
        # Scale components
        if normalize_to_max:
            max_events = max(len(data) for data in self.components.values())
            for comp_name, data in self.components.items():
                scale_factor = max_events / len(data)
                self.logger.log(
                    f"Scaling '{comp_name}': {len(data)} → {len(data) * scale_factor:.0f} "
                    f"events (factor: {scale_factor:.3f})",
                    "info"
                )
                self.scaled_components[comp_name] = (data, scale_factor)
        else:
            for comp_name, data in self.components.items():
                if comp_name in self.component_yields and self.component_yields[comp_name] is not None:
                    yield_target = self.component_yields[comp_name]
                    scale_factor = yield_target / len(data)
                else:
                    # Auto-scale relative to max
                    max_events = max(len(d) for d in self.components.values())
                    scale_factor = max_events / len(data)
                
                self.logger.log(
                    f"Scaling '{comp_name}': {len(data)} → {len(data) * scale_factor:.0f} "
                    f"events (factor: {scale_factor:.3f})",
                    "info"
                )
                self.scaled_components[comp_name] = (data, scale_factor)
    
    def log_component_statistics(self):
        """Log statistics about loaded components for diagnostics."""
        self.logger.log("="*60, "info")
        self.logger.log("Component Statistics", "info")
        self.logger.log("="*60, "info")
        
        for comp_name, data in self.components.items():
            if data is not None and len(data) > 0:
                try:
                    # Try to extract momentum if available
                    if 'trkfit' in data.fields and 'trkreco' in data.fields:
                        mom = np.sqrt(data.trkfit.trkreco.mom.x**2 + 
                                     data.trkfit.trkreco.mom.y**2 + 
                                     data.trkfit.trkreco.mom.z**2)
                        self.logger.log(f"{comp_name}: {len(data)} events, "
                                      f"mom: {np.mean(mom):.2f}±{np.std(mom):.2f} "
                                      f"({np.min(mom):.2f}-{np.max(mom):.2f})", "info")
                    else:
                        self.logger.log(f"{comp_name}: {len(data)} events", "info")
                except Exception as e:
                    self.logger.log(f"{comp_name}: {len(data)} events (stat error: {e})", "debug")
        
        self.logger.log("="*60, "info")
    
    def load_real_data(self, data_file: str, sign: str = "minus", 
                       location: str = "local") -> bool:
        """Load real data for optional overlay comparison.
        
        Args:
            data_file: Path to data file list
            sign: Charge sign
            location: File location
            
        Returns:
            True if successfully loaded, False otherwise
        """
        self.data = self._process_file_list(data_file, sign=sign, location=location, 
                                           component_name="real_data",
                                           mom_lo=self.fit_range_lo,
                                           mom_hi=self.fit_range_hi)
        if self.data is not None:
            self.logger.log(f"Loaded data: {len(self.data)} events", "info")
            return True
        else:
            self.logger.log(f"Failed to load data", "warning")
            return False
    
    def _extract_variable(self, data: ak.Array, var_name: str) -> Optional[np.ndarray]:
        """Extract a variable from the processed data.
        
        Supports special preprocessing for certain variables:
        - "recomom_ttfront": Reconstructed momentum at tracker front
        - "tracktime_ttfront": Track time at tracker front (matches process.py)
        - "recomom_mc_ttfront": MC true momentum at tracker front
        - Direct field access for standard variables like "trkfit.trksegpars_lh.p"
        
        Args:
            data: Awkward array with processed data
            var_name: Name of variable
            
        Returns:
            Flattened numpy array of variable values, or None on error
        """
        try:
            if var_name.lower() == "recomom_ttfront":
                self.logger.log(f"Extracting reconstructed momentum at TT_Front", "debug")
                selector = Select(verbosity=0)
                vector = Vector()
                
                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                mom_mag = vector.get_mag(trkfit_ent, 'mom')
                
                mom_mag = ak.drop_none(mom_mag)
                val = np.array(ak.flatten(mom_mag, axis=None))
                
                self.logger.log(f"Extracted {len(val)} recomom values", "debug")
                return val
            
            elif var_name.lower() == "recomom_mc_ttfront":
                self.logger.log(f"Extracting MC true momentum at TT_Front", "debug")
                selector = Select(verbosity=0)
                vector = Vector()
                
                trk_front_mc = selector.select_surface(
                    data['trkfit'], 
                    surface_name="TT_Front",
                    branch_name="trksegsmc"
                )
                trkfit_ent_mc = ak.mask(data['trkfit']["trksegsmc"], trk_front_mc)
                mom_mag_mc = vector.get_mag(trkfit_ent_mc, 'mom')
                
                mom_mag_mc = ak.drop_none(mom_mag_mc)
                val = np.array(ak.flatten(mom_mag_mc, axis=None))
                
                self.logger.log(f"Extracted {len(val)} MC momentum values", "debug")
                return val

            elif var_name.lower() == "tracktime_ttfront":
                # Match process.py exactly: TT_Front selection on trkfit.trksegs, then clean time
                self.logger.log("Extracting track time at TT_Front (process.py-compatible)", "debug")
                selector = Select(verbosity=0)

                trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
                trkfit_ent = ak.mask(data['trkfit']["trksegs"], trk_front)
                time = DataPreparationManager.clean_awkward_array(trkfit_ent['time'])

                val = np.array(ak.flatten(time, axis=None))

                self.logger.log(f"Extracted {len(val)} TT_Front time values", "debug")
                return val
            
            else:
                # Standard field access for nested paths
                parts = var_name.split('.')
                val = data
                
                for part in parts:
                    val = val[part]
                
                val = ak.drop_none(val)
                val = np.array(ak.flatten(val, axis=None))
                
                self.logger.log(f"Extracted {len(val)} values for '{var_name}'", "debug")
                return val
            
        except Exception as e:
            self.logger.log(f"Error extracting '{var_name}': {e}", "error")
            import traceback
            self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
            return None
    
    def _build_combined_data(self, variables: Dict[str, List[np.ndarray]],
                            scale_factors: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Combine scaled component data into single arrays.
        
        Maps component names to their MC truth codes so plotting infrastructure
        can properly identify and overlay true MC distributions.
        
        Args:
            variables: Dict[variable_name, List[arrays_per_component]]
            scale_factors: Dict mapping component names to scale factors
            
        Returns:
            Tuple of (
                Dict[variable_name, combined_array],
                component_category_array (with MC truth codes)
            )
        """
        combined = {}
        combined_component_cats = []
        
        # Create component index mapping using MC truth codes
        # This allows the plotting infrastructure to identify which process each event came from
        comp_names = list(self.scaled_components.keys())
        comp_to_mc_code = {}
        
        for comp_name in comp_names:
            # Use the predefined mapping, default to -2 (unknown) if not found
            mc_code = COMPONENT_TO_MC_CODE.get(comp_name.lower(), -2)
            comp_to_mc_code[comp_name] = mc_code
            
            if mc_code == -2:
                self.logger.log(
                    f"Warning: component '{comp_name}' not in known mapping, will be treated as unknown (-2)",
                    "warning"
                )
            else:
                self.logger.log(f"Component '{comp_name}' → MC code {mc_code}", "debug")
        
        for var_name, arrays_per_comp in variables.items():
            combined_arrays = []
            component_cats = []
            
            for comp_idx, (comp_name, array) in enumerate(zip(comp_names, arrays_per_comp)):
                if array is None or len(array) == 0:
                    continue
                
                # Duplicate events according to scale factor
                scale_factor = scale_factors[comp_name]
                n_repeat = int(np.ceil(scale_factor))
                
                # Create scaled array
                scaled_array = np.repeat(array, n_repeat)
                
                # Truncate to match exact scaling
                n_keep = int(np.round(len(array) * scale_factor))
                scaled_array = scaled_array[:n_keep]
                
                combined_arrays.append(scaled_array)
                
                # Use the MC code for this component
                mc_code = comp_to_mc_code[comp_name]
                component_cats.extend([mc_code] * len(scaled_array))
            
            combined[var_name] = np.concatenate(combined_arrays)
            if var_name == list(variables.keys())[0]:
                combined_component_cats = np.array(component_cats)
        
        return combined, combined_component_cats
    
    def _build_combined_full_data(self, scale_factors: Dict[str, float]) -> Optional[Tuple[ak.Array, np.ndarray]]:
        """Combine full data structures and assign synthetic MC truth codes based on component labels.
        
        Since each component file is labeled (dio, cosmic, etc.), we know the ground truth
        for every event: it's determined by which component file it came from.
        
        This method:
        1. Scales/duplicates events from each component
        2. REPLACES the MC truth codes in the data with our known codes
        3. This ensures plot_truth=True shows the correct distributions
        
        Args:
            scale_factors: Dict mapping component names to scale factors
            
        Returns:
            Tuple of (combined_data, mc_codes_array) where:
            - combined_data: awkward array with replaced MC truth codes
            - mc_codes_array: numpy array of MC codes for each event
        """
        comp_names = list(self.scaled_components.keys())
        combined_arrays = []
        combined_mc_codes = []
        
        for comp_name in comp_names:
            data, scale_factor = self.scaled_components[comp_name]
            
            if data is None or len(data) == 0:
                continue
            
            # Get the MC code for this component (this is the ground truth from the label)
            mc_code = COMPONENT_TO_MC_CODE.get(comp_name.lower(), -2)
            self.logger.log(
                f"Component '{comp_name}': assigning MC code {mc_code} to all events",
                "info"
            )
            
            # Calculate how many times to duplicate each event
            n_repeat = int(np.ceil(scale_factor))
            
            # Duplicate events in the awkward array
            scaled_data = data[np.repeat(np.arange(len(data)), n_repeat)]
            
            # Truncate to match exact scaling
            n_keep = int(np.round(len(data) * scale_factor))
            scaled_data = scaled_data[:n_keep]
            
            # Replace the MC truth codes in trkmc with our known codes
            # This is the KEY step - we're telling the data "these events are REALLY from process mc_code"
            if "trkmc" in scaled_data.fields:
                try:
                    # Create synthetic MC truth by replacing the startCode with our known code
                    # Extract the trkmc structure
                    trkmc_data = scaled_data["trkmc"]
                    
                    # Replace the startCode field with our synthetic code
                    # We create an array where each track gets the component's MC code
                    n_events = len(scaled_data)
                    synthetic_startcodes = ak.from_numpy(np.full(n_events, mc_code))
                    
                    # Build new trkmc with the replaced codes
                    # This is a bit tricky with awkward arrays...
                    # Let's try a simpler approach: just mark the data with metadata
                    scaled_data["_true_mc_code"] = ak.from_numpy(np.full(n_events, mc_code))
                    self.logger.log(
                        f"Added synthetic MC truth codes to {n_keep} scaled events from '{comp_name}'",
                        "debug"
                    )
                except Exception as e:
                    self.logger.log(
                        f"Could not modify MC truth for {comp_name}: {e}. Using metadata instead.",
                        "warning"
                    )
            
            combined_arrays.append(scaled_data)
            combined_mc_codes.extend([mc_code] * n_keep)
        
        if not combined_arrays:
            self.logger.log("No data to combine", "warning")
            return None
        
        # Combine all component data
        combined_full_data = ak.concatenate(combined_arrays)
        combined_mc_codes = np.array(combined_mc_codes)
        
        self.logger.log(
            f"Combined full data structures: {len(combined_full_data)} events "
            f"with synthetic MC truth codes",
            "info"
        )
        
        return combined_full_data, combined_mc_codes
    
    
    
    def fit_mom_1d(self, 
                   variable: str = 'recomom_ttfront',
                   fit_range: Tuple[float, float] = (97, 110),
                   plot: bool = True,
                   minos: bool = False,
                   use_constraints: bool = False,
                   constraints_dir: str = 'uncertainties/outputs') -> Any:
        """Execute 1D momentum fit on scaled components.
        
        Args:
            variable: Variable name to fit (e.g., 'recomom_ttfront')
            fit_range: Tuple of (min, max) for fit range
            plot: Whether to generate plots
            minos: Whether to compute Minos errors
            use_constraints: Whether to apply external constraints 
                           (default False for scaled fits since constraints 
                            are calibrated for full statistics)
            constraints_dir: Directory containing constraint specifications
            
        Returns:
            Result object from Unbinned_fit_mom
        """
        if not self.scaled_components:
            self.logger.log("No scaled components loaded. Call load_and_scale_components first.", "error")
            return None
        
        # Warn if fit_range differs from the cut range used in load_and_scale_components
        if fit_range[0] != self.fit_range_lo or fit_range[1] != self.fit_range_hi:
            self.logger.log(
                f"⚠ FIT RANGE MISMATCH:\n"
                f"  Cuts in analyze.py use: {self.fit_range_lo} - {self.fit_range_hi} MeV/c\n"
                f"  fit_mom_1d using:      {fit_range[0]} - {fit_range[1]} MeV/c\n"
                f"  → Data will be cut at {self.fit_range_lo}-{self.fit_range_hi}, then fit to {fit_range[0]}-{fit_range[1]}\n"
                f"  → For consistent behavior, call:\n"
                f"     builder.load_and_scale_components(files, fit_range=({fit_range[0]}, {fit_range[1]}))\n"
                f"     builder.fit_mom_1d(fit_range=({fit_range[0]}, {fit_range[1]}))",
                "warning"
            )
        
        self.logger.log("Starting 1D momentum fit on scaled components", "info")
        
        # Extract variable from each component
        comp_names = list(self.scaled_components.keys())
        variables = {variable: []}
        scale_factors = {}
        component_data_for_plot = {}  # Keep data for custom plotting
        
        self.logger.log("="*70, "info")
        self.logger.log("EXTRACTING VARIABLES FOR PLOTTING", "info")
        self.logger.log("="*70, "info")
        
        for comp_name in comp_names:
            data, scale_factor = self.scaled_components[comp_name]
            
            # Log what we're extracting from
            self.logger.log(f"\nComponent: {comp_name}", "info")
            self.logger.log(f"  Total events in component: {len(data)}", "info")
            self.logger.log(f"  Data structure fields: {list(data.fields) if hasattr(data, 'fields') else 'N/A'}", "debug")
            
            var_array = self._extract_variable(data, variable)
            
            if var_array is None or len(var_array) == 0:
                self.logger.log(f"  ✗ EMPTY extracted data for variable '{variable}'", "warning")
                variables[variable].append(None)
                component_data_for_plot[comp_name] = None
            else:
                # Enforce fit range explicitly (in addition to analyze.py cuts) for robust consistency.
                in_range = np.isfinite(var_array) & (var_array >= fit_range[0]) & (var_array <= fit_range[1])
                n_before = len(var_array)
                var_array = var_array[in_range]
                n_removed = n_before - len(var_array)

                self.logger.log(f"  ✓ Extracted {len(var_array)} values", "info")
                if n_removed > 0:
                    self.logger.log(f"    Explicit fit-range filter removed {n_removed} / {n_before} values", "warning")
                if len(var_array) > 0:
                    self.logger.log(f"    Range: {np.min(var_array):.2f} - {np.max(var_array):.2f}", "info")
                    self.logger.log(f"    Mean: {np.mean(var_array):.2f} ± {np.std(var_array):.2f}", "info")
                variables[variable].append(var_array)
                component_data_for_plot[comp_name] = var_array
            
            scale_factors[comp_name] = scale_factor
            self.logger.log(f"  Scale factor: {scale_factor:.3f}", "info")
        
        self.logger.log("="*70, "info")
        
        # Combine scaled data
        combined_data, component_cats = self._build_combined_data(variables, scale_factors)
        mom_mag = combined_data[variable]
        
        self.logger.log(
            f"Combined {len(self.scaled_components)} components into "
            f"{len(mom_mag)} events for fitting",
            "info"
        )

        # Save 1D fit input for sensitivity scanner (expects key 'mom_mag').
        self._save_scan_npz("scaled_fit_mom_mag.npz", mom_mag)
        
        # Only use constraints if explicitly requested (default False for scaled fits)
        constraints_dir_to_use = constraints_dir if use_constraints else None
        
        # Execute fit - returns (result, poi, loss, aux_nlls, combine_pdf, constraints)
        fit_tuple = Unbinned_fit_mom(
            mom_mag=mom_mag,
            count_particle_types=component_cats,
            fit_range_low=fit_range[0],
            fit_range_hi=fit_range[1],
            plot_truth=False,
            verbose=self.verbosity,
            minos=minos,
            plot_NLL=True,
            plot_results=True,  # Keep this True for proper fit result generation
            constraints_dir=constraints_dir_to_use
        )
        
        # Unpack the fit result tuple
        result, poi, loss, aux_nlls, combine_pdf, constraints = fit_tuple
        print("RESULT",result)
        # Generate custom plot with true shapes from scaled components
        if plot:
            try:
                # Extract real data momentum if available
                real_data_mom = None
                if self.data is not None:
                    try:
                        real_data_mom = self._extract_variable(self.data, variable)
                        if real_data_mom is not None and len(real_data_mom) > 0:
                            in_range_data = (
                                np.isfinite(real_data_mom)
                                & (real_data_mom >= fit_range[0])
                                & (real_data_mom <= fit_range[1])
                            )
                            n_data_before = len(real_data_mom)
                            real_data_mom = real_data_mom[in_range_data]
                            n_data_removed = n_data_before - len(real_data_mom)
                            if n_data_removed > 0:
                                self.logger.log(
                                    f"Real data explicit fit-range filter removed {n_data_removed} / {n_data_before} values",
                                    "warning"
                                )
                    except Exception as e:
                        self.logger.log(f"Could not extract {variable} from real data: {e}", "debug")
                
                plot_fit_with_true_shapes(
                    mom_mag=mom_mag,
                    combine_pdf=combine_pdf,
                    fit_result=result,
                    component_data_dict=component_data_for_plot,
                    component_names=comp_names,
                    scale_factors_dict=scale_factors,
                    fit_range=fit_range,
                    nbins=25,
                    output_file="fit_momentum_scaled_components.png",
                    title="1D Fit: Scaled MC Components",
                    verbosity=self.verbosity,
                    real_data_mom_mag=real_data_mom
                )
            except Exception as e:
                self.logger.log(f"Warning: custom plot generation failed: {e}", "warning")
        
        self.logger.log("1D momentum fit completed", "info")
        return result
    
    def fit_mom_time_2d(self,
                       mom_variable: str = 'recomom_ttfront',
                       time_variable: str = 'tracktime_ttfront',
                       fit_range_mom: Tuple[float, float] = (97, 110),
                       fit_range_time: Tuple[float, float] = (475, 1650),
                       plot: bool = True,
                       use_constraints: bool = False,
                       constraints_dir: str = 'uncertainties/outputs') -> Any:
        """Execute 2D momentum+time fit on scaled components.
        
        Args:
            mom_variable: Momentum variable name
            time_variable: Time variable name
            fit_range_mom: Tuple of (min, max) for momentum range
            fit_range_time: Tuple of (min, max) for time range
            plot: Whether to generate plots
            use_constraints: Whether to apply external constraints 
                           (default False for scaled fits since constraints 
                            are calibrated for full statistics)
            constraints_dir: Directory containing constraint specifications
            
        Returns:
            Result object from Unbinned_2d_fit_mom_time
        """
        if not self.scaled_components:
            self.logger.log("No scaled components loaded. Call load_and_scale_components first.", "error")
            return None
        
        # Warn if momentum fit_range differs from the cut range used in load_and_scale_components
        if fit_range_mom[0] != self.fit_range_lo or fit_range_mom[1] != self.fit_range_hi:
            self.logger.log(
                f"⚠ MOMENTUM RANGE MISMATCH:\n"
                f"  Cuts in analyze.py use: {self.fit_range_lo} - {self.fit_range_hi} MeV/c\n"
                f"  fit_mom_time_2d using:  {fit_range_mom[0]} - {fit_range_mom[1]} MeV/c\n"
                f"  → Data will be cut at {self.fit_range_lo}-{self.fit_range_hi}, then fit to {fit_range_mom[0]}-{fit_range_mom[1]}\n"
                f"  → For consistent behavior, call:\n"
                f"     builder.load_and_scale_components(files, fit_range=({fit_range_mom[0]}, {fit_range_mom[1]}))\n"
                f"     builder.fit_mom_time_2d(fit_range_mom=({fit_range_mom[0]}, {fit_range_mom[1]}))",
                "warning"
            )
        
        self.logger.log("Starting 2D momentum+time fit on scaled components", "info")
        
        # Extract variables from each component
        comp_names = list(self.scaled_components.keys())
        variables = {
            'mom': [],
            'time': []
        }
        scale_factors = {}
        component_mom_for_plot = {}
        component_time_for_plot = {}
        
        for comp_name in comp_names:
            data, scale_factor = self.scaled_components[comp_name]
            
            mom_array = self._extract_variable(data, mom_variable)
            time_array = self._extract_variable(data, time_variable)
            
            if (mom_array is None or len(mom_array) == 0 or
                time_array is None or len(time_array) == 0):
                self.logger.log(f"Empty variable data for {comp_name}, skipping", "warning")
                variables['mom'].append(None)
                variables['time'].append(None)
                component_mom_for_plot[comp_name] = None
                component_time_for_plot[comp_name] = None
            else:
                # Keep arrays aligned and apply explicit momentum/time fit ranges.
                if len(mom_array) != len(time_array):
                    n_min = min(len(mom_array), len(time_array))
                    self.logger.log(
                        f"{comp_name}: mom/time length mismatch ({len(mom_array)} vs {len(time_array)}), truncating to {n_min}",
                        "warning"
                    )
                    mom_array = mom_array[:n_min]
                    time_array = time_array[:n_min]

                valid_mask = np.isfinite(mom_array) & np.isfinite(time_array)
                range_mask = (
                    (mom_array >= fit_range_mom[0]) & (mom_array <= fit_range_mom[1])
                    & (time_array >= fit_range_time[0]) & (time_array <= fit_range_time[1])
                )
                combined_mask = valid_mask & range_mask
                n_before = len(mom_array)
                mom_array = mom_array[combined_mask]
                time_array = time_array[combined_mask]
                n_removed = n_before - len(mom_array)
                if n_removed > 0:
                    self.logger.log(
                        f"{comp_name}: explicit 2D range filter removed {n_removed} / {n_before} values",
                        "warning"
                    )

                variables['mom'].append(mom_array)
                variables['time'].append(time_array)
                component_mom_for_plot[comp_name] = mom_array
                component_time_for_plot[comp_name] = time_array
            
            scale_factors[comp_name] = scale_factor
        
        # Combine scaled data
        combined_data, component_cats = self._build_combined_data(variables, scale_factors)

        # Insert this right before Unbinned_fit_mom is called:
        self.logger.log("="*70, "info")
        self.logger.log("EXACT TRUE YIELDS SENT TO ZFIT (AFTER COMBINING & ROUNDING):", "info")
        self.logger.log("="*70, "info")

        # Invert the COMPONENT_TO_MC_CODE dict for easier printing
        mc_code_to_name = {v: k for k, v in COMPONENT_TO_MC_CODE.items()}

        # Count occurrences of each MC code in the final combined array
        unique_codes, final_counts = np.unique(component_cats, return_counts=True)
        for code, count in zip(unique_codes, final_counts):
            comp_name = mc_code_to_name.get(code, f"Unknown ({code})")
            target_yield = self.component_yields.get(comp_name.lower(), "N/A")
            self.logger.log(f"  {comp_name.upper():<10} -> Exact Yield: {count:<6} (Target asked for: {target_yield})", "info")


        mom_mag = combined_data['mom']
        times = combined_data['time']

        self.logger.log(
            f"Combined {len(self.scaled_components)} components into "
            f"{len(mom_mag)} events for 2D fitting",
            "info"
        )

        # Save 2D fit inputs for scanner/systematics workflows.
        self._save_scan_npz("scaled_fit_mom_time.npz", mom_mag, times)
        
        # Only use constraints if explicitly requested (default False for scaled fits)
        constraints_dir_to_use = constraints_dir if use_constraints else None
        
        # Execute fit - returns (result, poi, loss, combine_pdf, norms)
        fit_tuple = Unbinned_2d_fit_mom_time(
            mom_mag=mom_mag,
            times=times,
            count_particle_types=component_cats,
            fit_range_mom=fit_range_mom,
            fit_range_time=fit_range_time,
            plot_truth=False,
            verbose=self.verbosity,
            plot_NLL=False,
            plot_results=False,  # Keep this True for proper fit result generation
            constraints_dir=constraints_dir_to_use
        )
        
        # Unpack the fit result tuple
        result, poi, loss, combine_pdf, norms = fit_tuple
        
        # Generate custom plots with true shapes from scaled components
        if plot:
            try:
                # Extract real data variables if available
                real_data_mom = None
                real_data_time = None
                if self.data is not None:
                    try:
                        real_data_mom = self._extract_variable(self.data, mom_variable)
                    except Exception as e:
                        self.logger.log(f"Could not extract {mom_variable} from real data: {e}", "debug")
                    try:
                        real_data_time = self._extract_variable(self.data, time_variable)
                    except Exception as e:
                        self.logger.log(f"Could not extract {time_variable} from real data: {e}", "debug")

                    if (real_data_mom is not None and real_data_time is not None
                        and len(real_data_mom) > 0 and len(real_data_time) > 0):
                        if len(real_data_mom) != len(real_data_time):
                            n_min_data = min(len(real_data_mom), len(real_data_time))
                            self.logger.log(
                                f"real_data: mom/time length mismatch ({len(real_data_mom)} vs {len(real_data_time)}), truncating to {n_min_data}",
                                "warning"
                            )
                            real_data_mom = real_data_mom[:n_min_data]
                            real_data_time = real_data_time[:n_min_data]

                        data_mask = (
                            np.isfinite(real_data_mom) & np.isfinite(real_data_time)
                            & (real_data_mom >= fit_range_mom[0]) & (real_data_mom <= fit_range_mom[1])
                            & (real_data_time >= fit_range_time[0]) & (real_data_time <= fit_range_time[1])
                        )
                        n_data_before = len(real_data_mom)
                        real_data_mom = real_data_mom[data_mask]
                        real_data_time = real_data_time[data_mask]
                        n_data_removed = n_data_before - len(real_data_mom)
                        if n_data_removed > 0:
                            self.logger.log(
                                f"real_data: explicit 2D range filter removed {n_data_removed} / {n_data_before} values",
                                "warning"
                            )
                
                # Plot momentum distribution
                plot_fit_with_true_shapes(
                    mom_mag=mom_mag,
                    combine_pdf=combine_pdf,
                    fit_result=result,
                    component_data_dict=component_mom_for_plot,
                    component_names=comp_names,
                    scale_factors_dict=scale_factors,
                    fit_range=fit_range_mom,
                    nbins=25,
                    output_file="fit_momentum_2d_scaled_components.png",
                    title="2D Fit: Momentum Distribution (Scaled MC)",
                    verbosity=self.verbosity,
                    real_data_mom_mag=real_data_mom,
                    plot_obs='mom'
                )
                
                # Plot time distribution
                plot_fit_with_true_shapes(
                    mom_mag=times,
                    combine_pdf=combine_pdf,
                    fit_result=result,
                    component_data_dict=component_time_for_plot,
                    component_names=comp_names,
                    scale_factors_dict=scale_factors,
                    fit_range=fit_range_time,
                    nbins=30,
                    output_file="fit_time_2d_scaled_components.png",
                    title="2D Fit: Time Distribution (Scaled MC)",
                    verbosity=self.verbosity,
                    real_data_mom_mag=real_data_time,
                    plot_obs='time'
                )
            except Exception as e:
                self.logger.log(f"Warning: custom plot generation failed: {e}", "warning")
        
        self.logger.log("2D momentum+time fit completed", "info")
        return result
        return result


def main(args):
    """Command-line interface for scaled fitting."""
    
    logger = Logger(print_prefix="[scaled_fit_builder.main]", verbosity=1)
    
    builder = ScaledFitBuilder(verbosity=args.verbosity, jobs=args.jobs)
    
    # Parse component files
    component_files = {}
    if args.components:
        for item in args.components:
            name, path = item.split('=')
            component_files[name] = path
    
    # Load and scale components
    builder.load_and_scale_components(
        component_files,
        sign=args.sign,
        location=args.location,
        normalize_to_max=args.normalize_to_max
    )
    
    # Load optional data
    if args.data:
        builder.load_real_data(args.data, sign=args.sign, location=args.location)
    
    # Set custom yields if provided
    if args.yields:
        yields_dict = {}
        for item in args.yields:
            name, value = item.split('=')
            yields_dict[name] = float(value) if value.lower() != 'none' else None
        builder.set_component_yields(yields_dict)
    
    # Execute fit
    if args.fit_type == '1d' or args.fit_type == 'both':
        logger.log("Running 1D momentum fit", "info")
        result_1d = builder.fit_mom_1d(
            variable=args.variable,
            fit_range=(args.fit_range_lo, args.fit_range_hi),
            plot=not args.no_plot,
            minos=args.minos,
            constraints_dir=args.constraints_dir
        )
    
    if args.fit_type == '2d' or args.fit_type == 'both':
        logger.log("Running 2D momentum+time fit", "info")
        result_2d = builder.fit_mom_time_2d(
            mom_variable=args.variable,
            time_variable=args.time_variable,
            fit_range_mom=(args.fit_range_lo, args.fit_range_hi),
            fit_range_time=(args.time_range_lo, args.time_range_hi),
            plot=not args.no_plot,
            constraints_dir=args.constraints_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit scaled MC component samples using highest-stats scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1D momentum fit with scaled components
  python scaled_fit_builder.py \\
      --fit-type 1d \\
      --components dio=file_lists/dio.txt cosmic=file_lists/cosmic.txt \\
      --variable recomom_ttfront \\
            --time-variable tracktime_ttfront \
      --jobs 4

  # 2D momentum+time fit
  python scaled_fit_builder.py \\
      --fit-type 2d \\
      --components dio=... cosmic=... \\
      --variable recomom_ttfront \\
      --time-variable trkfit.trksegpars_lh.t0 \\
      --fit-range 97 110 \\
      --time-range 475 1650 \\
      --constraints-dir uncertainties/outputs
        """
    )
    
    parser.add_argument('--fit-type', choices=['1d', '2d', 'both'], default='1d',
                        help='Type of fit to perform')
    parser.add_argument('--components', nargs='+', required=True,
                        help='Component files as name=path pairs')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to real data file list (optional)')
    parser.add_argument('--variable', type=str, default='recomom_ttfront',
                        help='Variable to fit (default: recomom_ttfront)')
    parser.add_argument('--time-variable', type=str, default='tracktime_ttfront',
                        help='Time variable for 2D fit (default: tracktime_ttfront, matches process.py)')
    parser.add_argument('--fit-range', type=float, nargs=2, default=[97, 110],
                        dest='fit_range', metavar=('LO', 'HI'),
                        help='Fit range [lo hi]')
    parser.add_argument('--fit-range-lo', type=float, default=97, dest='fit_range_lo')
    parser.add_argument('--fit-range-hi', type=float, default=110, dest='fit_range_hi')
    parser.add_argument('--time-range', type=float, nargs=2, default=[475, 1650],
                        dest='time_range', metavar=('LO', 'HI'),
                        help='Time range for 2D fit [lo hi]')
    parser.add_argument('--time-range-lo', type=float, default=475, dest='time_range_lo')
    parser.add_argument('--time-range-hi', type=float, default=1650, dest='time_range_hi')
    parser.add_argument('--yields', nargs='+', default=None,
                        help='Component yields as name=value pairs (default: physics expectations)')
    parser.add_argument('--normalize-to-max', action='store_true',
                        help='Scale all components to match the max instead of using yields')
    parser.add_argument('--sign', choices=['minus', 'plus'], default='minus',
                        help='Charge sign')
    parser.add_argument('--location', choices=['disk', 'local'], default='disk',
                        help='File location')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of parallel jobs')
    parser.add_argument('--constraints-dir', type=str, default='uncertainties/outputs',
                        help='Directory with constraint specifications')
    parser.add_argument('--minos', action='store_true',
                        help='Compute Minos errors (1D fit only)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='Verbosity level (0-3)')
    
    args = parser.parse_args()
    
    # Handle fit-range arguments
    if args.fit_range is not None:
        args.fit_range_lo = args.fit_range[0]
        args.fit_range_hi = args.fit_range[1]
    
    if args.time_range is not None:
        args.time_range_lo = args.time_range[0]
        args.time_range_hi = args.time_range[1]
    
    main(args)
