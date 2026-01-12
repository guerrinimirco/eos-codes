"""
general_plotting_info.py
========================
Configuration module for consistent, publication-quality plot styling.

This module provides:
1. CMU Serif font configuration for all text (matching LaTeX documents)
2. Scientific figure presets for multi-panel plots
3. Standard color palettes matching Mathematica's StandardColor scheme
4. Helper functions for panel labels and styling

Usage:
    from general_plotting_info import (
        set_global_style, setup_scientific_figure, 
        STANDARD_COLORS, T_COLORS, LABELS
    )
    
    set_global_style()  # Call once at notebook start
    fig, axes = setup_scientific_figure(nrows=2, ncols=2)
    add_panel_labels(axes)
"""
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# FONT CONFIGURATIONS
# =============================================================================
FONTS = {
    'family': 'serif',
    'serif': ['CMU Serif'],       # Computer Modern Unicode - matches LaTeX
    'mathtext': 'cm',             # Computer Modern for math
    'title': 14,
    'label': 12,
    'tick': 11,
    'legend': 10,
    'annotation': 10,
    'panel_label': 14,            # For (a), (b), (c), (d) labels
}

# =============================================================================
# FIGURE CONFIGURATIONS
# =============================================================================
STYLE = {
    # Figure sizes (width, height) in inches
    'figsize_square': (6, 6),           # Single square plot
    'figsize_2x2': (8, 8),              # 2x2 grid, almost square
    'figsize_1x2': (10, 4.5),           # 1 row, 2 columns
    'figsize_2x1': (5, 9),              # 2 rows, 1 column
    'figsize_1x4': (14, 3.5),           # 1 row, 4 columns
    'figsize_wide': (12, 5),            # Wide format
    'figsize_default': (8, 6),          # Default
    
    # Resolution and line properties
    'dpi': 150,
    'linewidth': 1.8,
    'linewidth_thin': 1.2,
    'markersize': 5,
    
    # Grid properties
    'grid_alpha': 0.3,
    'grid_style': '-',                  # Solid gridlines
    'grid_linewidth': 0.5,
}

# =============================================================================
# STANDARD COLORS (Mathematica-style)
# =============================================================================
STANDARD_COLORS = {
    'Red': (0.8, 0.25, 0.33),
    'Green': (0.24, 0.6, 0.44),
    'Blue': (0.24, 0.6, 0.8),
    'Gray': (0.4, 0.4, 0.4),
    'Cyan': (0.1, 0.6, 0.6),
    'Magenta': (0.6, 0.24, 0.6),
    'Yellow': (0.95, 0.75, 0.1),
    'Brown': (0.6, 0.4, 0.2),
    'Orange': (0.9, 0.4, 0.0),
    'Pink': (0.9, 0.4, 0.6),
    'Purple': (0.5, 0.35, 0.65),
}

# Temperature colors for plots
T_COLORS = {
    0.1: STANDARD_COLORS['Gray'],
    10.0: STANDARD_COLORS['Purple'],
    50.0: STANDARD_COLORS['Orange'],
    100.0: STANDARD_COLORS['Red'],
}

# Color sequence for general use
COLORS_SEQ = [
    STANDARD_COLORS['Gray'],
    STANDARD_COLORS['Purple'],
    STANDARD_COLORS['Orange'],
    STANDARD_COLORS['Red'],
    STANDARD_COLORS['Blue'],
    STANDARD_COLORS['Green'],
    STANDARD_COLORS['Cyan'],
]

# Particle species colors
PARTICLE_COLORS = {
    'p': STANDARD_COLORS['Blue'],
    'n': STANDARD_COLORS['Orange'],
    'Lambda': STANDARD_COLORS['Green'],
    'Sigma+': STANDARD_COLORS['Red'],
    'Sigma0': STANDARD_COLORS['Purple'],
    'Sigma-': STANDARD_COLORS['Brown'],
    'Xi0': STANDARD_COLORS['Cyan'],
    'Xi-': STANDARD_COLORS['Yellow'],
    'e-': STANDARD_COLORS['Gray'],
}

# Line styles by particle type
PARTICLE_STYLES = {
    'p': '-', 'n': '-',
    'Lambda': '--', 'Sigma+': '--', 'Sigma0': '--', 'Sigma-': '--',
    'Xi0': '--', 'Xi-': '--',
    'e-': ':',
}

# =============================================================================
# LABEL TEMPLATES (LaTeX formatting)
# =============================================================================
LABELS = {
    # Density
    'nB': r'$n_B$ [fm$^{-3}$]',
    'nB_n0': r'$n_B/n_0$',
    'n0': 0.16,  # Saturation density value
    
    # Temperature
    'T': r'$T$ [MeV]',
    
    # Thermodynamic quantities
    'P': r'$P$ [MeV fm$^{-3}$]',
    'epsilon': r'$\varepsilon$ [MeV fm$^{-3}$]',
    's': r'$s$ [fm$^{-3}$]',
    's_nB': r'$s/n_B$',
    'S': r'$S = s/n_B$',
    
    # Chemical potentials
    'mu_B': r'$\mu_B$ [MeV]',
    'mu_C': r'$\mu_C$ [MeV]',
    'mu_Q': r'$\mu_Q$ [MeV]',
    'mu_S': r'$\mu_S$ [MeV]',
    
    # Meson fields
    'sigma': r'$\sigma$ [MeV]',
    'omega': r'$\omega$ [MeV]',
    'rho': r'$\rho$ [MeV]',
    'phi': r'$\phi$ [MeV]',
    
    # Fractions
    'Y_C': r'$Y_C$',
    'Y_i': r'$Y_i$',
}

# =============================================================================
# GLOBAL STYLE FUNCTION
# =============================================================================
def set_global_style():
    """
    Set global matplotlib style for publication-quality plots.
    
    Configures:
    - CMU Serif fonts for all text
    - Computer Modern math font
    - Proper minus signs (fixes the "-" symbol issue)
    - Scientific notation formatting
    """
    # Font family
    plt.rcParams['font.family'] = FONTS['family']
    plt.rcParams['font.serif'] = FONTS['serif']
    
    # Math text - Computer Modern to match serif
    plt.rcParams['mathtext.fontset'] = FONTS['mathtext']
    plt.rcParams['mathtext.rm'] = 'serif'
    plt.rcParams['mathtext.it'] = 'serif:italic'
    plt.rcParams['mathtext.bf'] = 'serif:bold'
    
    # Font sizes
    plt.rcParams['font.size'] = FONTS['label']
    plt.rcParams['axes.labelsize'] = FONTS['label']
    plt.rcParams['axes.titlesize'] = FONTS['title']
    plt.rcParams['xtick.labelsize'] = FONTS['tick']
    plt.rcParams['ytick.labelsize'] = FONTS['tick']
    plt.rcParams['legend.fontsize'] = FONTS['legend']
    
    # FIX MINUS SIGN: Use standard ASCII minus instead of Unicode
    plt.rcParams['axes.unicode_minus'] = True  # Use proper minus sign
    plt.rcParams['axes.formatter.use_mathtext'] = True
    
    # Figure defaults
    plt.rcParams['figure.dpi'] = STYLE['dpi']
    plt.rcParams['savefig.dpi'] = STYLE['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'


# =============================================================================
# FIGURE SETUP FUNCTIONS
# =============================================================================
def setup_scientific_figure(nrows=1, ncols=1, figsize=None, 
                            gray_background=False, sharex=False, sharey=False):
    """
    Create figure with publication-quality styling.
    
    Parameters:
        nrows, ncols: Subplot grid dimensions
        figsize: Custom (width, height) in inches, or auto-selected
        gray_background: Use light gray background (default False - white)
        sharex, sharey: Share axes between subplots
    
    Returns:
        fig, axes: matplotlib figure and axes array
    """
    # Auto-select figure size
    if figsize is None:
        if nrows == 2 and ncols == 2:
            figsize = STYLE['figsize_2x2']
        elif nrows == 1 and ncols == 2:
            figsize = STYLE['figsize_1x2']
        elif nrows == 2 and ncols == 1:
            figsize = STYLE['figsize_2x1']
        elif nrows == 1 and ncols == 4:
            figsize = STYLE['figsize_1x4']
        elif nrows == 1 and ncols == 1:
            figsize = STYLE['figsize_square']
        else:
            figsize = STYLE['figsize_default']
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey, dpi=STYLE['dpi'])
    
    # Apply gray background if requested
    if gray_background:
        facecolor = '#f0f0f0'
        if nrows * ncols == 1:
            axes.set_facecolor(facecolor)
        else:
            for ax in np.atleast_1d(axes).flat:
                ax.set_facecolor(facecolor)
    
    return fig, axes


def add_panel_labels(axes, labels=None, loc='upper left', fontsize=None):
    """
    Add panel labels (a), (b), (c), (d) to subplots.
    """
    if fontsize is None:
        fontsize = FONTS['panel_label']
    
    axes_flat = np.atleast_1d(axes).flat
    n_panels = len(list(axes_flat))
    
    if labels is None:
        labels = [f'({chr(ord("a") + i)})' for i in range(n_panels)]
    
    # Position mapping
    pos_map = {
        'upper left': (0.05, 0.95),
        'upper right': (0.95, 0.95),
        'lower left': (0.05, 0.05),
        'lower right': (0.95, 0.05),
    }
    x, y = pos_map.get(loc, (0.05, 0.95))
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'
    
    for ax, label in zip(np.atleast_1d(axes).flat, labels):
        ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
                fontweight='bold', ha=ha, va=va)


def apply_style(ax, grid=True, legend=True):
    """Apply consistent styling to a single axis."""
    ax.tick_params(labelsize=FONTS['tick'])
    
    if grid:
        ax.grid(True, alpha=STYLE['grid_alpha'], 
                linestyle=STYLE['grid_style'],
                linewidth=STYLE['grid_linewidth'])
    
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=FONTS['legend'], framealpha=0.9)


def get_T_color(T):
    """Get color for temperature value."""
    if T in T_COLORS:
        return T_COLORS[T]
    # Find closest
    temps = list(T_COLORS.keys())
    closest = min(temps, key=lambda t: abs(t - T))
    return T_COLORS[closest]


def save_figure(fig, filename, formats=['png', 'pdf']):
    """Save figure in multiple formats."""
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', dpi=STYLE['dpi'], 
                    bbox_inches='tight', facecolor='white')
    print(f"Saved: {filename}.{{{', '.join(formats)}}}")
