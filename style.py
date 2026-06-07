"""
Style configuration file for consistent plotting and visualization settings.
Import and use: from styles import COLORS, FONTS, PLOT_STYLE
"""

# Color palette
COLORS = {
    'Cosmic': '#1f77b4', 
    'int. RPC' : "#ffe30e",
    'ext. RPC': '#2ca02c', 
    'IPA Decays' :'#8c564b', 
    'DIO': '#e377c2', 
    'Signal':'#ff8000'
}


# Font settings
FONTS = {
    'title': {
        'size': 16,
        'weight': 'bold',
        'family': 'serif',
    },
    'label': {
        'size': 12,
        'weight': 'normal',
        'family': 'serif',
    },
    'tick': {
        'size': 10,
        'weight': 'normal',
        'family': 'serif',
    },
    'legend': {
        'size': 10,
        'weight': 'normal',
        'family': 'serif',
    },
}

# Plot styling
PLOT_STYLE = {
    'figure_size': (10, 6),
    'dpi': 100,
    'line_width': 2,
    'marker_size': 8,
    'alpha': 0.7,
}

# Matplotlib rcParams
MATPLOTLIB_RC = {
    'figure.figsize': PLOT_STYLE['figure_size'],
    'figure.dpi': PLOT_STYLE['dpi'],
    'font.size': FONTS['label']['size'],
    'lines.linewidth': PLOT_STYLE['line_width'],
    'lines.markersize': PLOT_STYLE['marker_size'],
    'axes.labelsize': FONTS['label']['size'],
    'axes.titlesize': FONTS['title']['size'],
    'xtick.labelsize': FONTS['tick']['size'],
    'ytick.labelsize': FONTS['tick']['size'],
    'legend.fontsize': FONTS['legend']['size'],
}
