"""
Style configuration file for consistent plotting and visualization settings.
Import and use: from styles import COLORS, FONTS, PLOT_STYLE, MATPLOTLIB_RC, apply_axes_style
"""

import matplotlib as mpl
import matplotlib.font_manager as mfm
import matplotlib.ticker as ticker

# Color palette
COLORS = {
    'Cosmic': '#1f77b4', 
    'int. RPC': "#ffe30e",
    'ext. RPC': '#2ca02c', 
    'IPA Decays': '#8c564b', 
    'DIO': '#e377c2', 
    'Signal': '#ff8000'
}

# Dynamic font discovery
preferred_serifs = ['DejaVu Serif', 'Times New Roman', 'Times', 'Palatino']
available_fonts = {f.name for f in mfm.fontManager.ttflist}
chosen_serif = next((f for f in preferred_serifs if f in available_fonts), 'DejaVu Serif')

# Font settings
FONTS = {
    'title': {
        'size': 18,
        'weight': 'bold',
        'family': 'serif',
    },
    'label': {
        'size': 18,
        'weight': 'normal',
        'family': 'serif',
    },
    'tick': {
        'size': 16,  # Updated to match rcParams
        'weight': 'normal',
        'family': 'serif',
    },
    'legend': {
        'size': 16,   # Updated to match rcParams
        'weight': 'normal',
        'family': 'serif',
    },
}

# Plot styling
PLOT_STYLE = {
    'figure_size': (10, 6),
    'dpi': 150,      # Updated to match rcParams
    'line_width': 2,
    'marker_size': 8,
    'alpha': 0.7,
    'grid_style': '-',
    'grid_alpha': 0.7
}

# Matplotlib rcParams
MATPLOTLIB_RC = {
    'font.family': 'serif',
    'font.serif': [chosen_serif],
    'font.size': 14,
    'figure.figsize': PLOT_STYLE['figure_size'],
    'figure.dpi': PLOT_STYLE['dpi'],
    'lines.linewidth': PLOT_STYLE['line_width'],
    'lines.markersize': PLOT_STYLE['marker_size'],
    'axes.titlesize': FONTS['title']['size'],
    'axes.labelsize': FONTS['label']['size'],
    'xtick.labelsize': FONTS['tick']['size'],
    'ytick.labelsize': FONTS['tick']['size'],
    'legend.fontsize': FONTS['legend']['size'],
    'axes.titleweight': FONTS['title']['weight'],
    'axes.labelweight': FONTS['label']['weight'],
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.5,
}

# Apply base rcParams immediately upon import
mpl.rcParams.update(MATPLOTLIB_RC)


def apply_axes_style(ax1, ax2=None):
    """
    Applies the specific major/minor tick configurations to the given axes.
    
    Parameters:
    ax1: The primary matplotlib Axes object.
    ax2: Optional secondary matplotlib Axes object (e.g., twinx or twiny).
    """
    # Configure primary axis (ax1)
    ax1.minorticks_on()
    ax1.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
    ax1.yaxis.set_minor_formatter(ticker.NullFormatter()) 
    
    # Configure secondary axis (ax2) if it exists
    if ax2 is not None:
        ax2.minorticks_on()
        ax2.tick_params(direction='in', which='both', top=True, right=True, labelsize=14)
        ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())