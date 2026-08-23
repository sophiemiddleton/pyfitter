"""Shared publication-quality plotting module for Mu2e style figures.

Adapted from experiment-wide design conventions (serif typography,
in-facing tick marks, customizable top tags, and colorblind-safe palettes).
"""

import matplotlib as mpl
import matplotlib.font_manager as mfm
import matplotlib.ticker as ticker

# Okabe-Ito colorblind-friendly color palette
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
}


def apply_publication_style():
    """Sets global rcParams for paper-ready figures."""
    preferred_serifs = ["DejaVu Serif", "Times New Roman", "Times", "Palatino"]
    available_fonts = {f.name for f in mfm.fontManager.ttflist}
    chosen_serif = next(
        (f for f in preferred_serifs if f in available_fonts),
        "DejaVu Serif"
    )

    mpl.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": [chosen_serif],
        "mathtext.fontset": "dejavuserif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        
        # Frame and Grid lines
        "axes.linewidth": 1.2,
        "axes.grid": False,
        "grid.linewidth": 0.5,
        
        # Output quality
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


def draw_watermark(ax, text="Mu2e Simulation", loc="left", y_offset=1.02):
    """Places a bold header tag above the plot axes frame."""
    x, ha = (0.0, "left") if loc == "left" else (1.0, "right")
    ax.text(
        x, y_offset, text,
        fontsize=20,
        fontweight="bold",
        ha=ha,
        va="bottom",
        transform=ax.transAxes,
        zorder=100
    )


def style_axis(ax, labelsize=16):
    """Configures default inward ticks and minor ticks for standard axes."""
    ax.minorticks_on()
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        labelsize=labelsize
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def style_pull_axis(ax, labelsize=16, y_step=1.0):
    """Configures the bottom residual/pull axis frame with custom tick spacing."""
    ax.minorticks_on()
    ax.tick_params(
        direction="in",
        which="both",
        top=True,
        right=True,
        labelsize=labelsize
    )
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_step / 2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())