#!/usr/bin/env python3
"""
Plotting Module for hybrid hadron-quark ZL + vMIT EOS
===================================================
Main function: plot_zlvmit_results()
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field

# ==============================================================================
#                           CONFIGURATION DEFAULTS
# ==============================================================================

OUTPUT_DIR = Path("/Users/mircoguerrini/Desktop/Research/Python_codes/output/zlvmit_hybrid_outputs")
N0 = 0.16  # Saturation density in fm^-3

# Standard colors (RGB tuples matching Mathematica)
STANDARD_RED = (0.8, 0.25, 0.33)
STANDARD_GREEN = (0.24, 0.6, 0.44)
STANDARD_BLUE = (0.24, 0.6, 0.8)
STANDARD_GRAY = (0.4, 0.4, 0.4)
STANDARD_CYAN = (0.1, 0.6, 0.6)
STANDARD_MAGENTA = (0.6, 0.24, 0.6)
STANDARD_YELLOW = (0.95, 0.75, 0.1)
STANDARD_BROWN = (0.6, 0.4, 0.2)
STANDARD_ORANGE = (0.9, 0.4, 0.0)
STANDARD_PINK = (0.9, 0.4, 0.6)
STANDARD_PURPLE = (0.5, 0.35, 0.65)

# Default line styles per eta (basic matplotlib colors)
DEFAULT_ETA_STYLES = {
    0.00: {'color': 'red',    'linestyle': '-',  'linewidth': 2.0, 'label': r'$\eta = 0$'},
    0.10: {'color': 'orange', 'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 0.1$'},
    0.30: {'color': 'green',  'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 0.3$'},
    0.60: {'color': 'blue',   'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 0.6$'},
    1.00: {'color': 'black',  'linestyle': '-',  'linewidth': 2.5, 'label': r'$\eta = 1$'},
}

# Standard line styles per eta (custom RGB colors matching Mathematica)
STANDARD_ETA_STYLES = {
    0.00: {'color': STANDARD_RED,    'linestyle': '-',  'linewidth': 2.0, 'label': r'$\eta = 0$'},
    0.10: {'color': STANDARD_ORANGE, 'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 0.1$'},
    0.30: {'color': STANDARD_GREEN,  'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 0.3$'},
    0.60: {'color': STANDARD_BLUE,   'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 0.6$'},
    1.00: {'color': STANDARD_GRAY,   'linestyle': '-',  'linewidth': 2.5, 'label': r'$\eta = 1$'},
}


# ==============================================================================
#                           PANEL CONFIGURATION
# ==============================================================================

@dataclass
class PanelConfig:
    """Configuration for a single plot panel."""
    quantity: str                     # Column name: 'chi', 'P_total', 'e_total', 'f_total', etc.
    mode: str = 'beta'                # 'beta' (beta-equilibrium) or 'yc_fixed' (fixed Y_C)
    T: Optional[float] = None         # Temperature [MeV] (filter data at this T)
    YC: Optional[float] = None        # Fixed Y_C value (only for mode='yc_fixed')
    eta_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.6, 1.0])
    B: int = 180                      # B^(1/4) in MeV
    a: float = 0.2                    # a parameter in fm^2
    show_boundaries: bool = False     # Overlay phase boundaries as vertical bands
    boundaries_file: Optional[str] = None  # Custom boundaries file path
    ylabel: Optional[str] = None      # Custom y-axis label
    ylim: Optional[Tuple[float, float]] = None  # Y-axis limits
    xlim: Tuple[float, float] = (0, 12)  # X-axis limits (n_B/n_0)
    normalize_nB: bool = True         # If True, plot n_B/n_0; otherwise n_B [fm^-3]
    title: Optional[str] = None       # Panel title (auto-generated if None)


# ==============================================================================
#                           DATA LOADING
# ==============================================================================

def _get_table_filename(eta: float, mode: str, YC: Optional[float], 
                        B: int, a: float, output_dir: Path) -> Path:
    """Generate the table filename based on mode and parameters."""
    if mode == 'beta':
        return output_dir / f"table_hybrid_eta{eta:.2f}_B{B}_a{a}_complete.dat"
    else:  # yc_fixed
        return output_dir / f"table_hybrid_eta{eta:.2f}_YC{YC:.2f}_B{B}_a{a}_complete.dat"


def _get_boundaries_filename(eta: float, mode: str, YC: Optional[float],
                             B: int, a: float, output_dir: Path) -> Path:
    """Generate the boundaries filename."""
    if mode == 'beta':
        return output_dir / f"boundaries_B{B}_a{a}.dat"
    else:
        return output_dir / f"boundaries_YC{YC:.2f}_B{B}_a{a}.dat"


def _load_table_data(filepath: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load data file and return as dictionary."""
    if not filepath.exists():
        return None
    
    # Read header to get column names
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and 'n_B' in line:
                header = line.lstrip('# ').split()
                break
    
    if header is None:
        return None
    
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    result = {name: data[:, i] for i, name in enumerate(header)}
    
    # Calculate derived quantities
    if 'n_B' in result:
        mask = result['n_B'] > 1e-10
        if 's_total' in result:
            result['s_per_nB'] = np.zeros_like(result['s_total'])
            result['s_per_nB'][mask] = result['s_total'][mask] / result['n_B'][mask]
        
        if 'e_total' in result:
            result['e_per_nB'] = np.zeros_like(result['e_total'])
            result['e_per_nB'][mask] = result['e_total'][mask] / result['n_B'][mask]
            
        if 'f_total' in result:
            result['f_per_nB'] = np.zeros_like(result['f_total'])
            result['f_per_nB'][mask] = result['f_total'][mask] / result['n_B'][mask]

    return result


def _load_boundaries(filepath: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load phase boundaries file with header parsing."""
    if not filepath.exists():
        return None
    
    # Identify column indices from header
    col_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'Columns:' in line:
                # e.g. # Columns: eta T n_B_onset n_B_offset ...
                # normalize spaces
                parts = line.split('Columns:')[1].strip().replace(',', ' ').split()
                for idx, col in enumerate(parts):
                    col_map[col] = idx
                break
    
    # Fallback if no header found or legacy format
    if not col_map:
        # Assume legacy beta-eq format: T, n_on, n_off
        col_map = {'T': 0, 'n_onset': 1, 'n_offset': 2, 'converged': 3}

    try:
        data = np.loadtxt(filepath)
    except:
        return None

    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    if data.size == 0:
        return None
    
    res = {}
    
    # Map file columns to standardized keys
    key_mapping = {
        'n_B_onset': 'n_onset',
        'n_B_offset': 'n_offset',
        'n_onset': 'n_onset', 
        'n_offset': 'n_offset',
        'T': 'T',
        'eta': 'eta',
        'conv': 'converged',
        'converged': 'converged'
    }
    
    for file_col, idx in col_map.items():
        std_key = key_mapping.get(file_col, file_col)
        # Handle cases where column might not exist in data if header is wrong
        if idx < data.shape[1]:
            res[std_key] = data[:, idx]
            
    # Legacy fallback for conv if not found
    if 'converged' not in res:
        res['converged'] = np.ones(len(res.get('T', [])), dtype=bool)
        
    return res


def _filter_by_temperature(data: Dict[str, np.ndarray], T_target: float, 
                           tol: float = 0.1) -> Dict[str, np.ndarray]:
    """Filter data for a specific temperature."""
    if 'T' not in data:
        return data
    
    T = data['T']
    unique_T = np.unique(T)
    idx_closest = np.argmin(np.abs(unique_T - T_target))
    T_actual = unique_T[idx_closest]
    
    mask = np.abs(T - T_actual) < tol
    if 'converged' in data:
        mask &= data['converged'] == 1
    
    result = {k: v[mask] for k, v in data.items()}
    
    # Sort by n_B
    if 'n_B' in result and len(result['n_B']) > 0:
        sort_idx = np.argsort(result['n_B'])
        result = {k: v[sort_idx] for k, v in result.items()}
    
    return result


# ==============================================================================
#                           PLOTTING HELPERS
# ==============================================================================

def _setup_style():
    """Configure matplotlib"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,  # No box around legend
        'legend.framealpha': 0.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


def _get_ylabel(quantity: str) -> str:
    """Get default y-axis label for a quantity."""
    labels = {
        'chi': r'$\chi$',
        'P_total': r'$P$ [MeV/fm$^3$]',
        'e_total': r'$\varepsilon/n_B$ [MeV]',
        'f_total': r'$f/n_B$ [MeV]',
        'e_per_nB': r'$\varepsilon/n_B$ [MeV]',
        'f_per_nB': r'$f/n_B$ [MeV]',
        's_total': r'$s$ [fm$^{-3}$]',
        's_per_nB': r'$s/n_B$',
        'n_B': r'$n_B$ [fm$^{-3}$]',
    }
    return labels.get(quantity, quantity)


def _plot_single_panel(ax, config: PanelConfig, output_dir: Path = OUTPUT_DIR):
    """Plot a single panel based on its configuration."""
    
    for eta in config.eta_values:
        style = STANDARD_ETA_STYLES.get(eta, {
            'color': 'gray', 'linestyle': '-', 'linewidth': 1.5, 
            'label': f'η={eta}'
        })
        
        # Load data - try both complete and primary files and merge
        filepath_complete = _get_table_filename(eta, config.mode, config.YC, 
                                                config.B, config.a, output_dir)
        filepath_primary = str(filepath_complete).replace('_complete.dat', '_primary.dat')
        filepath_primary = Path(filepath_primary)
        
        data = {}
        
        # Load complete file first
        data_complete = _load_table_data(filepath_complete)
        if data_complete:
            data.update(data_complete)
        
        # Load primary file and merge (primary has chi, etc.)
        data_primary = _load_table_data(filepath_primary)
        if data_primary:
            for key, val in data_primary.items():
                if key not in data:
                    data[key] = val
        
        if not data:
            continue
        
        # Filter by temperature if specified
        if config.T is not None:
            data = _filter_by_temperature(data, config.T)
        
        if config.quantity not in data or len(data.get(config.quantity, [])) == 0:
            continue
        
        # Get x and y values
        n_B = data['n_B']
        x = n_B / N0 if config.normalize_nB else n_B
        
        # Handle special quantities
        if config.quantity == 's_per_nB' and 's_total' in data:
            y = data['s_total'] / n_B
        else:
            y = data[config.quantity]
        
        ax.plot(x, y, color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], label=style['label'])
        
        # Add phase boundary shading if requested
        if config.show_boundaries:
            bounds_file = (Path(config.boundaries_file) if config.boundaries_file 
                          else _get_boundaries_filename(eta, config.mode, config.YC,
                                                        config.B, config.a, output_dir))
            bounds = _load_boundaries(bounds_file)
            if bounds is not None and config.T is not None:
                idx = np.argmin(np.abs(bounds['T'] - config.T))
                if bounds['converged'][idx] == 1:
                    n_onset = bounds['n_onset'][idx] / N0 if config.normalize_nB else bounds['n_onset'][idx]
                    n_offset = bounds['n_offset'][idx] / N0 if config.normalize_nB else bounds['n_offset'][idx]
                    ax.axvspan(n_onset, n_offset, alpha=0.15, color=style['color'])
    
    # Labels and limits
    xlabel = r'$n_B/n_0$' if config.normalize_nB else r'$n_B$ [fm$^{-3}$]'
    ylabel = config.ylabel if config.ylabel else _get_ylabel(config.quantity)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(config.xlim)
    if config.ylim:
        ax.set_ylim(config.ylim)
    
    ax.legend(loc='best')
    
    # Title
    if config.title:
        ax.set_title(config.title)
    else:
        mode_str = 'β-eq.' if config.mode == 'beta' else f'$Y_C = {config.YC:.2f}$'
        T_str = f'$T = {config.T:.0f}$ MeV' if config.T else ''
        ax.set_title(f'{mode_str}, {T_str}' if T_str else mode_str)


# ==============================================================================
#                           MAIN FUNCTION
# ==============================================================================

def plot_zlvmit_results(
    panels: List[PanelConfig],
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = None,
    show_panel_labels: bool = True,
    panel_label_style: str = 'abc',  # 'abc' for a), b), c) or 'num' for 1), 2), 3)
    suptitle: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    output_dir: Path = OUTPUT_DIR,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a flexible multi-panel plot for EOS results.
    
    Parameters
    ----------
    panels : List[PanelConfig]
        List of panel configurations. Each PanelConfig specifies:
        - quantity: str - Column to plot ('chi', 'P_total', 'e_total', 'f_total', etc.)
        - mode: str - 'beta' for beta-equilibrium, 'yc_fixed' for fixed Y_C
        - T: float - Temperature to filter [MeV]
        - YC: float - Fixed Y_C value (required if mode='yc_fixed')
        - eta_values: List[float] - List of eta values to include
        - B, a: int, float - Model parameters
        - show_boundaries: bool - Overlay phase boundaries as shaded regions
        - boundaries_file: str - Custom path to boundaries file
        - ylabel, ylim, xlim, title: customization options
    
    nrows, ncols : int
        Number of rows and columns in the panel grid.
    
    figsize : Tuple[float, float], optional
        Figure size in inches. Default: (6*ncols, 5*nrows)
    
    show_panel_labels : bool
        If True, add labels (a), (b), (c), ... to panels.
    
    panel_label_style : str
        'abc' for a), b), c) or 'num' for 1), 2), 3)
    
    suptitle : str, optional
        Super title for the figure.
    
    save_path : str or Path, optional
        Path to save the figure (supports .pdf, .png).
    
    output_dir : Path
        Directory containing the data files.
    
    show : bool
        If True, display the figure.
    
    Returns
    -------
    fig, axes : Figure and array of axes.
    
    Examples
    --------
    # Simple single panel
    >>> panels = [PanelConfig(quantity='chi', T=50)]
    >>> plot_zlvmit_results(panels)
    
    # 2x2 grid with different quantities
    >>> panels = [
    ...     PanelConfig(quantity='chi', T=50),
    ...     PanelConfig(quantity='P_total', T=50),
    ...     PanelConfig(quantity='chi', mode='yc_fixed', YC=0.4, T=50),
    ...     PanelConfig(quantity='P_total', mode='yc_fixed', YC=0.4, T=50),
    ... ]
    >>> plot_zlvmit_results(panels, nrows=2, ncols=2, show_panel_labels=True)
    
    # Compare same quantity at different T
    >>> panels = [PanelConfig(quantity='chi', T=t) for t in [10, 30, 50, 80]]
    >>> plot_zlvmit_results(panels, nrows=2, ncols=2)
    """
    _setup_style()
    
    # Default figure size
    if figsize is None:
        figsize = (3.5 * ncols, 3.3 * nrows)  # Nearly square panels for 2-column papers
    
    # Create figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()
    
    # Panel labels
    if panel_label_style == 'abc':
        labels = 'abcdefghijklmnopqrstuvwxyz'
    else:
        labels = [str(i+1) for i in range(26)]
    
    # Plot each panel
    for i, config in enumerate(panels):
        if i >= len(axes_flat):
            break
        
        ax = axes_flat[i]
        _plot_single_panel(ax, config, output_dir)
        
        # Add panel label
        if show_panel_labels and i < len(labels):
            label_text = f'({labels[i]})'
            ax.text(0.02, 0.96, label_text, transform=ax.transAxes,
                    fontsize=11, fontweight='normal',
                    verticalalignment='top', horizontalalignment='left')
    
    # Hide unused axes
    for i in range(len(panels), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Super title
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='normal', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        # Also save PNG if saving PDF
        if save_path.suffix == '.pdf':
            png_path = save_path.with_suffix('.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_phase_boundaries(
    eta_values: List[float] = [0.0, 0.1, 0.3, 0.6, 1.0],
    mode: str = 'beta',
    YC: Optional[float] = None,
    B: int = 180,
    a: float = 0.2,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 12),
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    output_dir: Path = OUTPUT_DIR,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot phase boundaries in the (nB, T) plane.
    
    Parameters
    ----------
    eta_values : List[float]
        List of eta values to include.
    mode : str
        'beta' for beta-equilibrium boundaries, 'yc_fixed' for fixed Y_C.
    YC : float, optional
        Fixed Y_C value (required if mode='yc_fixed').
    B, a : int, float
        Model parameters.
    normalize_nB : bool
        If True, plot nB/n0; otherwise nB [fm^-3].
    xlim, ylim : Tuple
        Axis limits.
    figsize : Tuple
        Figure size.
    title : str, optional
        Plot title.
    save_path : Path, optional
        Path to save figure.
    output_dir : Path
        Directory containing boundary files.
    show : bool
        If True, display the figure.
    
    Returns
    -------
    fig, ax : Figure and axis.
    """
    _setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for eta in eta_values:
        style = DEFAULT_ETA_STYLES.get(eta, {
            'color': 'gray', 'linestyle': '-', 'linewidth': 1.5,
            'label': f'η={eta}'
        })
        
        # Construct boundaries filename
        if mode == 'beta':
            fname = output_dir / f"phase_boundaries_eta{eta:.2f}_B{B}_a{a}.dat"
        else:
            fname = output_dir / f"boundaries_YC{YC:.2f}_B{B}_a{a}.dat"
        
        if not fname.exists():
            print(f"  [WARNING] Boundaries file not found: {fname.name}")
            continue
        
        # Load data
        data = np.loadtxt(fname)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        T = data[:, 0]
        n_onset = data[:, 1]
        n_offset = data[:, 2]
        conv = data[:, 3] if data.shape[1] > 3 else np.ones(len(data))
        
        # Normalize
        if normalize_nB:
            n_onset = n_onset / N0
            n_offset = n_offset / N0
        
        mask = conv == 1
        
        # Plot onset and offset
        ax.plot(n_onset[mask], T[mask], color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'],
                label=style['label'])
        ax.plot(n_offset[mask], T[mask], color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'])
        
        # Mark non-converged points
        mask_nc = ~mask
        if np.any(mask_nc):
            ax.scatter(n_onset[mask_nc], T[mask_nc], color=style['color'],
                       marker='x', s=20, alpha=0.5)
            ax.scatter(n_offset[mask_nc], T[mask_nc], color=style['color'],
                       marker='x', s=20, alpha=0.5)
    
    # Labels
    xlabel = r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$T$ [MeV]')
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='best')
    
    if title:
        ax.set_title(title)
    else:
        mode_str = r'$\beta$-equilibrium' if mode == 'beta' else f'$Y_C = {YC:.2f}$'
        ax.set_title(f'Phase Boundaries ({mode_str})')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        if save_path.suffix == '.pdf':
            png_path = save_path.with_suffix('.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_composition(
    eta: float = 0.0,
    T: float = 50.0,
    mode: str = 'beta',
    YC: Optional[float] = None,
    B: int = 180,
    a: float = 0.2,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 12),
    ylim: Tuple[float, float] = (-0.05, 1.05),
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    output_dir: Path = OUTPUT_DIR,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot particle composition (Y_i fractions) vs nB at fixed eta and T.
    
    Shows Y_p, Y_n, Y_u, Y_d, Y_s, Y_e with gray shading in mixed phase region.
    
    Parameters
    ----------
    eta : float
        eta parameter value.
    T : float
        Temperature [MeV].
    mode : str
        'beta' for beta-equilibrium, 'yc_fixed' for fixed Y_C.
    YC : float, optional
        Fixed Y_C value (required if mode='yc_fixed').
    B, a : int, float
        Model parameters.
    normalize_nB : bool
        If True, plot nB/n0.
    xlim, ylim : Tuple
        Axis limits.
    figsize : Tuple
        Figure size.
    title : str, optional
        Plot title.
    save_path : Path, optional
        Path to save figure.
    output_dir : Path
        Directory containing data files.
    show : bool
        If True, display the figure.
    
    Returns
    -------
    fig, ax : Figure and axis.
    """
    _setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Load data (merge complete and primary)
    filepath_complete = _get_table_filename(eta, mode, YC, B, a, output_dir)
    filepath_primary = Path(str(filepath_complete).replace('_complete.dat', '_primary.dat'))
    
    data = {}
    data_complete = _load_table_data(filepath_complete)
    if data_complete:
        data.update(data_complete)
    data_primary = _load_table_data(filepath_primary)
    if data_primary:
        for key, val in data_primary.items():
            if key not in data:
                data[key] = val
    
    if not data:
        print(f"[ERROR] No data found for eta={eta}")
        return fig, ax
    
    # Filter by temperature
    data = _filter_by_temperature(data, T)
    if len(data.get('n_B', [])) == 0:
        print(f"[ERROR] No data at T={T} MeV")
        return fig, ax
    
    # Get x values
    n_B = data['n_B']
    x = n_B / N0 if normalize_nB else n_B
    
    # Add gray shading for mixed phase
    if mode == 'beta':
        bounds_file = output_dir / f"phase_boundaries_eta{eta:.2f}_B{B}_a{a}.dat"
    else:
        bounds_file = output_dir / f"boundaries_YC{YC:.2f}_B{B}_a{a}.dat"
    
    bounds = _load_boundaries(bounds_file)
    if bounds is not None:
        idx = np.argmin(np.abs(bounds['T'] - T))
        if bounds['converged'][idx] == 1:
            n_onset = bounds['n_onset'][idx] / N0 if normalize_nB else bounds['n_onset'][idx]
            n_offset = bounds['n_offset'][idx] / N0 if normalize_nB else bounds['n_offset'][idx]
            ax.axvspan(n_onset, n_offset, alpha=0.2, color='gray', label='Mixed phase')
    
    # Define species to plot
    species = [
        ('Y_p_tot', r'$Y_p$', 'blue', '-', 2.0),
        ('Y_n_tot', r'$Y_n$', 'red', '-', 2.0),
        ('Y_u_tot', r'$Y_u$', 'cyan', '--', 1.5),
        ('Y_d_tot', r'$Y_d$', 'orange', '--', 1.5),
        ('Y_s_tot', r'$Y_s$', 'green', '--', 1.5),
        ('Y_e_tot', r'$Y_e$', 'purple', ':', 1.5),
    ]
    
    for col_name, label, color, ls, lw in species:
        if col_name in data and len(data[col_name]) > 0:
            ax.plot(x, data[col_name], color=color, linestyle=ls, 
                    linewidth=lw, label=label)
    
    # Labels
    xlabel = r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$Y_i$')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc='best', ncol=2)
    
    if title:
        ax.set_title(title)
    else:
        mode_str = r'$\beta$-eq.' if mode == 'beta' else f'$Y_C = {YC:.2f}$'
        ax.set_title(f'Composition at $\\eta = {eta}$, $T = {T:.0f}$ MeV ({mode_str})')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        if save_path.suffix == '.pdf':
            png_path = save_path.with_suffix('.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")
    
    if show:
        plt.show()
    
    return fig, ax


# ==============================================================================
#                           ISENTROPE PLOTTING
# ==============================================================================

def plot_isentropes(
    S_values: List[float] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    eta_values: List[float] = [0.0, 1.0],
    mode: str = 'beta',
    YC: Optional[float] = None,
    B: int = 180,
    a: float = 0.2,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 12),
    ylim: Tuple[float, float] = None,
    figsize: Tuple[float, float] = (10, 7),
    show_boundaries: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    output_dir: Path = OUTPUT_DIR,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot isentropes T(n_B) at fixed entropy per baryon S = s/n_B.
    
    Shows lines of constant entropy in the (n_B, T) plane, with optional
    phase boundaries overlay.
    
    Parameters
    ----------
    S_values : List[float]
        Entropy per baryon values to plot (in units of k_B).
    eta_values : List[float]
        eta values for phase boundaries (if show_boundaries=True).
    mode : str
        'beta' for beta-equilibrium, 'yc_fixed' for fixed Y_C.
    YC : float, optional
        Fixed Y_C value (required if mode='yc_fixed').
    B, a : int, float
        Model parameters.
    normalize_nB : bool
        If True, plot nB/n0; otherwise nB [fm^-3].
    xlim, ylim : Tuple
        Axis limits.
    figsize : Tuple
        Figure size.
    show_boundaries : bool
        If True, overlay phase boundaries.
    title : str, optional
        Plot title.
    save_path : Path, optional
        Path to save figure.
    output_dir : Path
        Directory containing data files.
    show : bool
        If True, display the figure.
    
    Returns
    -------
    fig, ax : Figure and axis.
    """
    from zlvmit_isentropic import create_entropy_interpolator, compute_isentropes
    
    _setup_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colormap for S values
    cmap = plt.cm.coolwarm
    colors = [cmap(i / (len(S_values) - 1)) if len(S_values) > 1 else cmap(0.5) 
              for i in range(len(S_values))]
    
    # Load data from the first available eta to get the grid
    for eta in [0.0, 0.1, 0.3, 0.6, 1.0]:
        filepath_complete = _get_table_filename(eta, mode, YC, B, a, output_dir)
        data = _load_table_data(filepath_complete)
        if data is not None:
            break
    
    if data is None:
        print("[ERROR] No table data found for isentrope computation")
        return fig, ax
    
    # Get unique grid values
    n_B_values = np.unique(data['n_B'])
    T_values = np.unique(data['T'])
    
    # Create interpolator
    interpolator = create_entropy_interpolator(data, n_B_values, T_values)
    
    # Compute isentropes
    isentropes = compute_isentropes(
        n_B_values, 
        S_values=np.array(S_values),
        interpolator=interpolator
    )
    
    # X values
    x = n_B_values / N0 if normalize_nB else n_B_values
    
    # Plot isentropes
    for i, S in enumerate(S_values):
        T_arr = isentropes[S]
        valid = ~np.isnan(T_arr)
        if np.sum(valid) > 0:
            ax.plot(x[valid], T_arr[valid], color=colors[i], linewidth=2.0,
                    label=f'$S = {S:.1f}$')
    
    # Plot phase boundaries if requested
    if show_boundaries:
        for eta in eta_values:
            style = DEFAULT_ETA_STYLES.get(eta, {
                'color': 'gray', 'linestyle': ':', 'linewidth': 1.5
            })
            
            if mode == 'beta':
                bounds_file = output_dir / f"phase_boundaries_eta{eta:.2f}_B{B}_a{a}.dat"
            else:
                bounds_file = output_dir / f"boundaries_YC{YC:.2f}_B{B}_a{a}.dat"
            
            bounds = _load_boundaries(bounds_file)
            if bounds is not None:
                mask = bounds.get('converged', np.ones(len(bounds['T']))) == 1
                n_on = bounds['n_onset'][mask] / N0 if normalize_nB else bounds['n_onset'][mask]
                n_off = bounds['n_offset'][mask] / N0 if normalize_nB else bounds['n_offset'][mask]
                T_b = bounds['T'][mask]
                
                ax.plot(n_on, T_b, color=style['color'], linestyle='--', 
                        linewidth=1.5, alpha=0.7)
                ax.plot(n_off, T_b, color=style['color'], linestyle='--',
                        linewidth=1.5, alpha=0.7)
    
    # Labels
    xlabel = r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$T$ [MeV]')
    ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    ax.legend(loc='best', title='Entropy per baryon')
    
    if title:
        ax.set_title(title)
    else:
        mode_str = r'$\beta$-equilibrium' if mode == 'beta' else f'$Y_C = {YC:.2f}$'
        ax.set_title(f'Isentropes ({mode_str})')
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path = Path(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        if save_path.suffix == '.pdf':
            png_path = save_path.with_suffix('.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {png_path}")
    
    if show:
        plt.show()
    
    return fig, ax


# ==============================================================================
#                           EXAMPLE USAGE
# ==============================================================================


if __name__ == "__main__":
    _setup_style()
    
    # Label positioning helper
    LABEL_POS = {
        'top_left': (0.05, 0.90),
        'top_center': (0.5, 0.90),
        'top_right': (0.95, 0.90),
        'bottom_left': (0.05, 0.10),
    }
    
    def add_label(ax, text, pos='top_left'):
        x, y = LABEL_POS.get(pos, LABEL_POS['top_left'])
        ha = 'right' if 'right' in pos else 'left' if 'left' in pos else 'center'
        ax.text(x, y, text, transform=ax.transAxes, fontsize=11, fontweight='normal', ha=ha)

    # ==========================================================================
    # 1. PHASE BOUNDARIES - 4 panels: beta-eq, YC=0.1, YC=0.3, YC=0.5
    # ==========================================================================
    print("Plotting phase boundaries (4-panel)...")
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.6))
    axes = axes.flatten()
    labels = ['a', 'b', 'c', 'd']
    
    configs = [
        ('beta', None, r'$\beta$-eq.'),
        ('yc_fixed', 0.1, r'$Y_C = 0.1$'),
        ('yc_fixed', 0.3, r'$Y_C = 0.3$'),
        ('yc_fixed', 0.5, r'$Y_C = 0.5$'),
    ]
    
    for i, (ax, (mode, yc, title)) in enumerate(zip(axes, configs)):
        for eta in [0.0, 0.1, 0.3, 0.6, 1.0]:
            style = STANDARD_ETA_STYLES.get(eta, {'color': 'gray', 'linestyle': '-'})
            # Try B=180 first, fallback to B=165
            if mode == 'beta':
                candidates = [
                    OUTPUT_DIR / f"phase_boundaries_eta{eta:.2f}_B180_a0.2.dat",
                    OUTPUT_DIR / f"phase_boundaries_eta{eta:.2f}_B165_a0.2.dat"
                ]
            else:
                candidates = [
                    OUTPUT_DIR / f"boundaries_YC{yc:.2f}_B180_a0.2.dat",
                    OUTPUT_DIR / f"boundaries_YC{yc:.2f}_B165_a0.2.dat"
                ]
            
            for bounds_file in candidates:
                if bounds_file.exists():
                    bounds = _load_boundaries(bounds_file)
                    if bounds is not None:
                         # FILTER BY ETA for fixed YC files which contain all etas
                        if mode == 'yc_fixed' and 'eta' in bounds:
                            eta_col = bounds['eta']
                            eta_mask = np.abs(eta_col - eta) < 1e-3
                            if not np.any(eta_mask):
                                continue 
                            
                            filtered_bounds = {}
                            for k, v in bounds.items():
                                if len(v) == len(eta_col):
                                    filtered_bounds[k] = v[eta_mask]
                                else:
                                    filtered_bounds[k] = v
                            bounds = filtered_bounds

                        mask = bounds.get('converged', np.ones(len(bounds['T']))) == 1
                        n_on = bounds['n_onset'][mask] / N0
                        n_off = bounds['n_offset'][mask] / N0
                        T_b = bounds['T'][mask]
                        ax.plot(n_on, T_b, color=style['color'], linestyle='-', linewidth=1.5, label=style.get('label'))
                        ax.plot(n_off, T_b, color=style['color'], linestyle='-', linewidth=1.5)
                        break 
        
        ax.set_xlim(0.1, 12)
        ax.set_ylim(0, 120)
        ax.set_xlabel(r'$n_B/n_0$')
        ax.set_ylabel(r'$T$ [MeV]')
        ax.set_title(title)
        add_label(ax, f'({labels[i]})', pos='top_left')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'boundaries_4panel.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'boundaries_4panel.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'boundaries_4panel.pdf'}")
    plt.close()
    
    # ==========================================================================
    # 2. CHI vs nB - Beta equilibrium
    # ==========================================================================
    print("Plotting chi(nB) beta-equilibrium...")
    T_values_beta = [0.1, 20, 50, 70]
    panels_chi_beta = [
        PanelConfig(quantity='chi', mode='beta', T=T, title=f'$T = {T}$ MeV, $\\beta$-eq.', xlim=(0.1, 12))
        for T in T_values_beta
    ]
    plot_zlvmit_results(
        panels_chi_beta, nrows=2, ncols=2,
        suptitle=None,
        save_path=OUTPUT_DIR / 'chi_vs_nB_beta.pdf',
        show=False
    )
    
    # ==========================================================================
    # 3. P, e, f, s per nB - Beta equilibrium
    # ==========================================================================
    # Note: s_per_nB, e_per_nB, f_per_nB are calculated in _load_table_data
    quantities = ['P_total', 'e_per_nB', 'f_per_nB', 's_per_nB']
    
    for qty in quantities:
        print(f"Plotting {qty}(nB) beta-equilibrium...")
        panels = [
            PanelConfig(quantity=qty, mode='beta', T=T, title=f'$T = {T}$ MeV, $\\beta$-eq.', xlim=(0.1, 12))
            for T in T_values_beta
        ]
        plot_zlvmit_results(
            panels, nrows=2, ncols=2,
            suptitle=None,
            save_path=OUTPUT_DIR / f'{qty}_vs_nB_beta.pdf',
            show=False
        )

    # ==========================================================================
    # 3b. Thermodynamics - Fixed YC (4 panels: T=10/50 x YC=0.1/0.4)
    # ==========================================================================
    configs_thermo = [
        (10, 0.1, r'$T=10$ MeV, $Y_C=0.1$'),
        (50, 0.1, r'$T=50$ MeV, $Y_C=0.1$'),
        (10, 0.4, r'$T=10$ MeV, $Y_C=0.4$'),
        (50, 0.4, r'$T=50$ MeV, $Y_C=0.4$'),
    ]
    
    thermo_quantities = ['chi', 'P_total', 'e_per_nB', 'f_per_nB', 's_per_nB']
    
    for qty in thermo_quantities:
        print(f"Plotting {qty}(nB) fixed YC...")
        panels = [
            PanelConfig(quantity=qty, mode='yc_fixed', YC=yc, T=t, title=ttl, xlim=(0.1, 12))
            for t, yc, ttl in configs_thermo
        ]
        plot_zlvmit_results(
            panels, nrows=2, ncols=2,
            suptitle=None,
            save_path=OUTPUT_DIR / f'{qty}_vs_nB_fixedYC.pdf',
            show=False
        )

    
    # ==========================================================================
    # 4. COMPOSITION Y_i - Beta equilibrium (6 panels)
    # ==========================================================================
    print("Plotting composition (beta-eq, 6-panel)...")
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    eta_T_combos = [(0.0, 10), (0.0, 50), (0.6, 10), (0.6, 50), (1.0, 10), (1.0, 50)]
    
    for idx, (ax, (eta, T)) in enumerate(zip(axes.flatten(), eta_T_combos)):
        filepath = _get_table_filename(eta, 'beta', None, 165, 0.2, OUTPUT_DIR)
        data = _load_table_data(filepath)
        if data is not None:
            data_T = _filter_by_temperature(data, T)
            if len(data_T.get('n_B', [])) > 0:
                x = data_T['n_B'] / N0
                
                # Shading
                if 'chi' in data_T:
                    plot_phase_shading(ax, x, data_T['chi'])

                species = [('Y_p_tot', r'$Y_p$', STANDARD_RED), ('Y_n_tot', r'$Y_n$', STANDARD_BLUE),
                          ('Y_u_tot', r'$Y_u$', STANDARD_CYAN), ('Y_d_tot', r'$Y_d$', STANDARD_ORANGE),
                          ('Y_s_tot', r'$Y_s$', STANDARD_GREEN), ('Y_e_tot', r'$Y_e$', STANDARD_PURPLE)]
                for col, lbl, clr in species:
                    if col in data_T:
                        ax.plot(x, data_T[col], color=clr, linewidth=1.5, label=lbl)
        else:
             print(f"  [WARNING] File not found: {filepath}")

        ax.set_xlim(0.1, 12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$n_B/n_0$')
        ax.set_ylabel(r'$Y_i$')
        ax.set_title(f'$\\eta={eta}$, $T={T}$ MeV, $\\beta$-eq.')
        add_label(ax, f'({labels[idx]})', pos='top_left')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'composition_beta_6panel.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'composition_beta_6panel.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'composition_beta_6panel.pdf'}")
    plt.close()
    
    # ==========================================================================
    # 5. COMPOSITION Y_i - Fixed YC=0.1
    # ==========================================================================
    print("Plotting composition (YC=0.1, 6-panel)...")
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))
    for idx, (ax, (eta, T)) in enumerate(zip(axes.flatten(), eta_T_combos)):
        filepath = _get_table_filename(eta, 'yc_fixed', 0.1, 165, 0.2, OUTPUT_DIR)
        data = _load_table_data(filepath)
        if data is not None:
            data_T = _filter_by_temperature(data, T)
            if len(data_T.get('n_B', [])) > 0:
                x = data_T['n_B'] / N0
                
                 # Shading
                if 'chi' in data_T:
                    plot_phase_shading(ax, x, data_T['chi'])
                            
                species = [('Y_p_tot', r'$Y_p$', STANDARD_RED), ('Y_n_tot', r'$Y_n$', STANDARD_BLUE),
                          ('Y_u_tot', r'$Y_u$', STANDARD_CYAN), ('Y_d_tot', r'$Y_d$', STANDARD_ORANGE),
                          ('Y_s_tot', r'$Y_s$', STANDARD_GREEN)]
                for col, lbl, clr in species:
                    if col in data_T:
                        ax.plot(x, data_T[col], color=clr, linewidth=1.5, label=lbl)

        ax.set_xlim(0.1, 12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$n_B/n_0$')
        ax.set_ylabel(r'$Y_i$')
        ax.set_title(f'$\\eta={eta}$, $T={T}$ MeV, $Y_C=0.1$')
        add_label(ax, f'({labels[idx]})', pos='top_left')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'composition_YC0.1_6panel.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'composition_YC0.1_6panel.pdf'}")
    plt.close()
    
    # ==========================================================================
    # 6. COMPOSITION Y_i - Fixed YC=0.4
    # ==========================================================================
    print("Plotting composition (YC=0.4, 6-panel)...")
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))
    for idx, (ax, (eta, T)) in enumerate(zip(axes.flatten(), eta_T_combos)):
        filepath = _get_table_filename(eta, 'yc_fixed', 0.4, 165, 0.2, OUTPUT_DIR)
        data = _load_table_data(filepath)
        if data is not None:
            data_T = _filter_by_temperature(data, T)
            if len(data_T.get('n_B', [])) > 0:
                x = data_T['n_B'] / N0
                
                # Shading
                if 'chi' in data_T:
                    plot_phase_shading(ax, x, data_T['chi'])
                            
                species = [('Y_p_tot', r'$Y_p$', STANDARD_RED), ('Y_n_tot', r'$Y_n$', STANDARD_BLUE),
                          ('Y_u_tot', r'$Y_u$', STANDARD_CYAN), ('Y_d_tot', r'$Y_d$', STANDARD_ORANGE),
                          ('Y_s_tot', r'$Y_s$', STANDARD_GREEN)]
                for col, lbl, clr in species:
                    if col in data_T:
                        ax.plot(x, data_T[col], color=clr, linewidth=1.5, label=lbl)

        ax.set_xlim(0.1, 12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$n_B/n_0$')
        ax.set_ylabel(r'$Y_i$')
        ax.set_title(f'$\\eta={eta}$, $T={T}$ MeV, $Y_C=0.4$')
        add_label(ax, f'({labels[idx]})', pos='top_left')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'composition_YC0.4_6panel.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'composition_YC0.4_6panel.pdf'}")
    plt.close()
    
    # ==========================================================================
    # 7. ISENTROPES - Fixed YC (4 panels: YC=0.1 S=1, YC=0.4 S=1, YC=0.1 S=2, YC=0.4 S=2)
    # ==========================================================================
    print("Plotting isentropes (fixed YC, 4-panel)...")
    from zlvmit_isentropic import create_entropy_interpolator, compute_isentropes
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.6))
    axes = axes.flatten()
    iso_labels = ['a', 'b', 'c', 'd']
    
    iso_configs = [
        (0.1, 1.0, r'$Y_C=0.1$, $S=1$'),
        (0.4, 1.0, r'$Y_C=0.4$, $S=1$'),
        (0.1, 2.0, r'$Y_C=0.1$, $S=2$'),
        (0.4, 2.0, r'$Y_C=0.4$, $S=2$'),
    ]
    
    for idx, (ax, (yc, S_target, title)) in enumerate(zip(axes, iso_configs)):
        for eta in [0.0, 0.1, 0.3, 0.6, 1.0]:
            style = STANDARD_ETA_STYLES.get(eta, {'color': 'gray'})
            filepath = _get_table_filename(eta, 'yc_fixed', yc, 165, 0.2, OUTPUT_DIR)
            data = _load_table_data(filepath)
            if data is not None:
                n_B = data['n_B']
                T_all = data['T']
                
                # We need uniques for interpolation
                n_B_values = np.unique(n_B)
                T_values = np.unique(T_all)
                
                try:
                    interp = create_entropy_interpolator(data, n_B_values, T_values)
                    isentropes = compute_isentropes(n_B_values, np.array([S_target]), interp)
                    T_arr = isentropes[S_target]
                    valid = ~np.isnan(T_arr)
                    x = n_B_values / N0
                    if np.sum(valid) > 0:
                        ax.plot(x[valid], T_arr[valid], color=style['color'], linewidth=1.5, label=style.get('label'))
                except:
                    pass
            
            # Boundaries - try B=180 then B=165
            candidates = [
                OUTPUT_DIR / f"boundaries_YC{yc:.2f}_B180_a0.2.dat",
                OUTPUT_DIR / f"boundaries_YC{yc:.2f}_B165_a0.2.dat"
            ]
            for bounds_file in candidates:
                if bounds_file.exists():
                    bounds = _load_boundaries(bounds_file)
                    if bounds is not None:
                        # FILTER BY ETA for fixed YC files
                        if 'eta' in bounds:
                            eta_col = bounds['eta']
                            eta_mask = np.abs(eta_col - eta) < 1e-3
                            if not np.any(eta_mask):
                                continue # Eta not in file
                            
                            filtered_bounds = {}
                            for k, v in bounds.items():
                                if len(v) == len(eta_col):
                                    filtered_bounds[k] = v[eta_mask]
                                else:
                                    filtered_bounds[k] = v
                            bounds = filtered_bounds

                        mask = bounds.get('converged', np.ones(len(bounds['T']))) == 1
                        n_on = bounds['n_onset'][mask] / N0
                        n_off = bounds['n_offset'][mask] / N0
                        T_b = bounds['T'][mask]
                        ax.plot(n_on, T_b, color=style['color'], linestyle='--', linewidth=1.2, alpha=0.9)
                        ax.plot(n_off, T_b, color=style['color'], linestyle='--', linewidth=1.2, alpha=0.9)
                        break
        
        ax.set_xlim(0.1, 12)
        ax.set_ylim(0, 120)
        ax.set_xlabel(r'$n_B/n_0$')
        ax.set_ylabel(r'$T$ [MeV]')
        ax.set_title(title)
        add_label(ax, f'({iso_labels[idx]})', pos='top_left')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'isentropes_fixedYC_4panel.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'isentropes_fixedYC_4panel.pdf'}")
    plt.close()
    
    # ==========================================================================
    # 8. ISENTROPES - Beta eq (2 panels: S=1, S=2)
    # ==========================================================================
    print("Plotting isentropes (beta-eq, 2-panel)...")
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3))
    iso_labels_2 = ['a', 'b']
    
    for idx, (ax, S_target) in enumerate(zip(axes, [1.0, 2.0])):
        for eta in [0.0, 0.1, 0.3, 0.6, 1.0]:
            style = STANDARD_ETA_STYLES.get(eta, {'color': 'gray'})
            filepath = _get_table_filename(eta, 'beta', None, 165, 0.2, OUTPUT_DIR)
            data = _load_table_data(filepath)
            if data is not None:
                n_B_values = np.unique(data['n_B'])
                T_values = np.unique(data['T'])
                try:
                    interp = create_entropy_interpolator(data, n_B_values, T_values)
                    isentropes = compute_isentropes(n_B_values, np.array([S_target]), interp)
                    T_arr = isentropes[S_target]
                    valid = ~np.isnan(T_arr)
                    x = n_B_values / N0
                    if np.sum(valid) > 0:
                        ax.plot(x[valid], T_arr[valid], color=style['color'], linewidth=1.5, label=style.get('label'))
                except:
                    pass
            
            # Boundaries - try B=180 then B=165
            candidates = [
                OUTPUT_DIR / f"phase_boundaries_eta{eta:.2f}_B180_a0.2.dat",
                OUTPUT_DIR / f"phase_boundaries_eta{eta:.2f}_B165_a0.2.dat"
            ]
            for bounds_file in candidates:
                if bounds_file.exists():
                    bounds = _load_boundaries(bounds_file)
                    if bounds is not None:
                        mask = bounds.get('converged', np.ones(len(bounds['T']))) == 1
                        n_on = bounds['n_onset'][mask] / N0
                        n_off = bounds['n_offset'][mask] / N0
                        T_b = bounds['T'][mask]
                        ax.plot(n_on, T_b, color=style['color'], linestyle='--', linewidth=0.8, alpha=0.5)
                        ax.plot(n_off, T_b, color=style['color'], linestyle='--', linewidth=0.8, alpha=0.5)
                        break

        ax.set_xlim(0.1, 12)
        ax.set_ylim(0, 120)
        ax.set_xlabel(r'$n_B/n_0$')
        ax.set_ylabel(r'$T$ [MeV]')
        ax.set_title(f'$S={S_target:.0f}$, $\\beta$-eq.')
        add_label(ax, f'({iso_labels_2[idx]})', pos='top_left')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'isentropes_beta_2panel.pdf', dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'isentropes_beta_2panel.pdf'}")
    plt.close()
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


