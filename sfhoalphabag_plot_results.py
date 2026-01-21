#!/usr/bin/env python3
"""
Plotting Module for hybrid hadron-quark SFHo + AlphaBag EOS
============================================================
Main functions: plot_chi_vs_nB(), plot_pressure(), plot_phase_boundaries()
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ==============================================================================
#                           CONFIGURATION
# ==============================================================================

OUTPUT_DIR = Path("/Users/mircoguerrini/Desktop/Research/Python_codes/output/sfhoalphabag_hybrid_outputs")
PLOT_DIR = OUTPUT_DIR / "plots"
N0 = 0.1583  # Saturation density in fm^-3 (SFHo)

# Colors for eta values
ETA_STYLES = {
    0.00: {'color': 'red',   'linestyle': '-',  'linewidth': 2.0, 'label': r'$\eta = 0$ (Gibbs)'},
    1.00: {'color': 'black', 'linestyle': '--', 'linewidth': 2.0, 'label': r'$\eta = 1$ (Maxwell)'},
}

# Colors for T values
T_COLORS = plt.cm.viridis(np.linspace(0.2, 0.9, 8))

# ==============================================================================
#                           DATA LOADING
# ==============================================================================

def get_table_filename(eta: float, Y_C: float, B: float = 180.0, a: float = 0.1571) -> Path:
    """Generate the table filename based on parameters."""
    return OUTPUT_DIR / f"table_hybrid_eta{eta:.2f}_YC{Y_C:.2f}_B{int(B)}_a{a:.4f}.dat"

def get_boundaries_filename(Y_C: float, B: float = 180.0, a: float = 0.1571) -> Path:
    """Generate the boundaries filename."""
    return OUTPUT_DIR / "boundaries" / f"boundaries_YC{Y_C:.2f}_B{int(B)}_a{a:.4f}.dat"

def load_table(filepath: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load data file and return as dictionary."""
    if not filepath.exists():
        print(f"  [WARNING] File not found: {filepath}")
        return None
    
    # Read header to get column names
    header = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') and 'n_B' in line and 'T' in line:
                header = line.lstrip('# ').split()
                break
    
    if header is None:
        print(f"  [WARNING] No header found in: {filepath}")
        return None
    
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    result = {name: data[:, i] for i, name in enumerate(header)}
    
    # Calculate derived quantities
    if 'n_B' in result:
        mask = result['n_B'] > 1e-10
        result['s_per_nB'] = np.zeros_like(result.get('s_total', result['n_B']))
        result['e_per_nB'] = np.zeros_like(result.get('e_total', result['n_B']))
        result['f_per_nB'] = np.zeros_like(result.get('f_total', result['n_B']))
        if 's_total' in result:
            result['s_per_nB'][mask] = result['s_total'][mask] / result['n_B'][mask]
        if 'e_total' in result:
            result['e_per_nB'][mask] = result['e_total'][mask] / result['n_B'][mask]
        if 'f_total' in result:
            result['f_per_nB'][mask] = result['f_total'][mask] / result['n_B'][mask]
    
    return result

def load_boundaries(filepath: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load phase boundaries file."""
    if not filepath.exists():
        print(f"  [WARNING] Boundaries file not found: {filepath}")
        return None
    
    # Parse header
    col_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            if 'Columns:' in line:
                parts = line.split('Columns:')[1].strip().replace(',', ' ').split()
                for idx, col in enumerate(parts):
                    col_map[col] = idx
                break
    
    if not col_map:
        col_map = {'eta': 0, 'T': 1, 'n_B_onset': 2, 'n_B_offset': 3, 'conv': 4}
    
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    res = {}
    key_mapping = {
        'n_B_onset': 'n_onset', 'n_B_offset': 'n_offset',
        'T': 'T', 'eta': 'eta', 'conv': 'converged'
    }
    for file_col, idx in col_map.items():
        std_key = key_mapping.get(file_col, file_col)
        if idx < data.shape[1]:
            res[std_key] = data[:, idx]
    
    return res

def filter_temperature(data: Dict[str, np.ndarray], T_target: float, tol: float = 0.5) -> Dict[str, np.ndarray]:
    """Filter data for a specific temperature."""
    if 'T' not in data:
        return data
    T = data['T']
    mask = np.abs(T - T_target) < tol
    result = {k: v[mask] for k, v in data.items()}
    if 'n_B' in result and len(result['n_B']) > 0:
        sort_idx = np.argsort(result['n_B'])
        result = {k: v[sort_idx] for k, v in result.items()}
    return result

# ==============================================================================
#                           PLOTTING SETUP
# ==============================================================================

def setup_style():
    """Configure matplotlib."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'legend.frameon': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })

# ==============================================================================
#                           MAIN PLOTTING FUNCTIONS
# ==============================================================================

def plot_chi_vs_nB(
    Y_C: float = 0.5,
    T_values: List[float] = [10.0, 30.0, 50.0, 70.0],
    eta_values: List[float] = [0.0, 1.0],
    B: float = 180.0,
    a: float = 0.1571,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 12),
    save: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot chi (quark fraction) vs nB for different T and eta.
    """
    setup_style()
    PLOT_DIR.mkdir(exist_ok=True)
    
    nrows = len(T_values)
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for col, eta in enumerate(eta_values):
        filepath = get_table_filename(eta, Y_C, B, a)
        data = load_table(filepath)
        
        if data is None:
            continue
        
        for row, T in enumerate(T_values):
            ax = axes[row, col]
            data_T = filter_temperature(data, T)
            
            if 'chi' not in data_T or len(data_T.get('chi', [])) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            n_B = data_T['n_B']
            x = n_B / N0 if normalize_nB else n_B
            chi = data_T['chi']
            
            ax.plot(x, chi, color=ETA_STYLES[eta]['color'], 
                   linewidth=ETA_STYLES[eta]['linewidth'])
            
            ax.set_xlim(xlim)
            ax.set_ylim(-0.05, 1.05)
            
            if row == nrows - 1:
                ax.set_xlabel(r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]')
            if col == 0:
                ax.set_ylabel(r'$\chi$')
            
            ax.set_title(f'T = {T:.0f} MeV, {ETA_STYLES[eta]["label"]}')
    
    plt.tight_layout()
    
    if save:
        save_path = PLOT_DIR / f"chi_vs_nB_YC{Y_C:.2f}.pdf"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_pressure_vs_nB(
    Y_C: float = 0.5,
    T_values: List[float] = [10.0, 50.0, 100.0],
    eta_values: List[float] = [0.0, 1.0],
    B: float = 180.0,
    a: float = 0.1571,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 12),
    save: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot total pressure vs nB for different T and eta.
    
    Shows both eta=0 (Gibbs) and eta=1 (Maxwell) on same panel for comparison.
    """
    setup_style()
    PLOT_DIR.mkdir(exist_ok=True)
    
    nrows = (len(T_values) + 1) // 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), sharey=False)
    axes_flat = axes.flatten() if nrows > 1 else [axes[0], axes[1]] if ncols > 1 else [axes]
    
    for i, T in enumerate(T_values):
        if i >= len(axes_flat):
            break
        ax = axes_flat[i]
        
        for eta in eta_values:
            filepath = get_table_filename(eta, Y_C, B, a)
            data = load_table(filepath)
            
            if data is None:
                continue
            
            data_T = filter_temperature(data, T)
            
            if 'P_total' not in data_T or len(data_T['P_total']) == 0:
                continue
            
            n_B = data_T['n_B']
            x = n_B / N0 if normalize_nB else n_B
            P = data_T['P_total']
            
            ax.plot(x, P, color=ETA_STYLES[eta]['color'], 
                   linestyle=ETA_STYLES[eta]['linestyle'],
                   linewidth=ETA_STYLES[eta]['linewidth'],
                   label=ETA_STYLES[eta]['label'])
        
        ax.set_xlim(xlim)
        ax.set_xlabel(r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]')
        ax.set_ylabel(r'$P$ [MeV/fm$^3$]')
        ax.set_title(f'$Y_C = {Y_C:.2f}$, T = {T:.0f} MeV')
        ax.legend(loc='upper left')
    
    # Hide extra axes
    for i in range(len(T_values), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        save_path = PLOT_DIR / f"pressure_vs_nB_YC{Y_C:.2f}.pdf"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_phase_boundaries(
    Y_C: float = 0.5,
    eta_values: List[float] = [0.0, 1.0],
    B: float = 180.0,
    a: float = 0.1571,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 20),
    ylim: Tuple[float, float] = (0, 105),
    save: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot phase boundaries in the (nB/n0, T) plane.
    """
    setup_style()
    PLOT_DIR.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bounds_file = get_boundaries_filename(Y_C, B, a)
    bounds = load_boundaries(bounds_file)
    
    if bounds is None:
        ax.text(0.5, 0.5, 'No boundaries data', ha='center', va='center', transform=ax.transAxes)
    else:
        for eta in eta_values:
            style = ETA_STYLES.get(eta, {
                'color': 'gray', 'linestyle': '-', 'linewidth': 1.5, 'label': f'η={eta}'
            })
            
            # Filter for this eta
            if 'eta' in bounds:
                mask = np.abs(bounds['eta'] - eta) < 0.01
            else:
                mask = np.ones(len(bounds['T']), dtype=bool)
            
            if 'converged' in bounds:
                mask &= bounds['converged'] == 1
            
            T = bounds['T'][mask]
            n_onset = bounds['n_onset'][mask]
            n_offset = bounds['n_offset'][mask]
            
            if normalize_nB:
                n_onset = n_onset / N0
                n_offset = n_offset / N0
            
            # Sort by T
            sort_idx = np.argsort(T)
            T = T[sort_idx]
            n_onset = n_onset[sort_idx]
            n_offset = n_offset[sort_idx]
            
            ax.plot(n_onset, T, color=style['color'], linestyle=style['linestyle'],
                   linewidth=style['linewidth'], label=style['label'] + ' onset')
            ax.plot(n_offset, T, color=style['color'], linestyle=style['linestyle'],
                   linewidth=style['linewidth'])
            ax.fill_betweenx(T, n_onset, n_offset, color=style['color'], alpha=0.1)
    
    ax.set_xlabel(r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]')
    ax.set_ylabel(r'$T$ [MeV]')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(f'Phase Boundaries ($Y_C = {Y_C:.2f}$)')
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save:
        save_path = PLOT_DIR / f"phase_boundaries_YC{Y_C:.2f}.pdf"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_4panel_thermo(
    Y_C: float = 0.5,
    T: float = 50.0,
    eta_values: List[float] = [0.0, 1.0],
    B: float = 180.0,
    a: float = 0.1571,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = (0, 12),
    save: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot 4-panel thermodynamics: chi, P, e/n_B, f/n_B.
    """
    setup_style()
    PLOT_DIR.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    quantities = [
        ('chi', r'$\chi$'),
        ('P_total', r'$P$ [MeV/fm$^3$]'),
        ('e_per_nB', r'$\varepsilon/n_B$ [MeV]'),
        ('f_per_nB', r'$f/n_B$ [MeV]'),
    ]
    
    for eta in eta_values:
        filepath = get_table_filename(eta, Y_C, B, a)
        data = load_table(filepath)
        
        if data is None:
            continue
        
        data_T = filter_temperature(data, T)
        
        if len(data_T.get('n_B', [])) == 0:
            continue
        
        n_B = data_T['n_B']
        x = n_B / N0 if normalize_nB else n_B
        
        for idx, (qty, ylabel) in enumerate(quantities):
            ax = axes.flatten()[idx]
            
            if qty not in data_T:
                continue
            
            y = data_T[qty]
            ax.plot(x, y, color=ETA_STYLES[eta]['color'],
                   linestyle=ETA_STYLES[eta]['linestyle'],
                   linewidth=ETA_STYLES[eta]['linewidth'],
                   label=ETA_STYLES[eta]['label'])
    
    # Labels and limits
    for idx, (qty, ylabel) in enumerate(quantities):
        ax = axes.flatten()[idx]
        ax.set_xlim(xlim)
        ax.set_xlabel(r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best')
        if qty == 'chi':
            ax.set_ylim(-0.05, 1.05)
    
    fig.suptitle(f'$Y_C = {Y_C:.2f}$, T = {T:.0f} MeV', fontsize=14)
    plt.tight_layout()
    
    if save:
        save_path = PLOT_DIR / f"4panel_thermo_YC{Y_C:.2f}_T{T:.0f}.pdf"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


def plot_pressure_discontinuity(
    Y_C: float = 0.5,
    T_values: List[float] = [10.0, 50.0, 100.0],
    eta: float = 1.0,
    B: float = 180.0,
    a: float = 0.1571,
    normalize_nB: bool = True,
    xlim: Tuple[float, float] = None,
    save: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot pressure vs nB zoomed at phase boundaries to check discontinuities.
    Shows pressure jump at onset and offset of mixed phase.
    """
    setup_style()
    PLOT_DIR.mkdir(exist_ok=True)
    
    # Load boundaries to get onset/offset
    bounds_file = get_boundaries_filename(Y_C, B, a)
    bounds = load_boundaries(bounds_file)
    
    ncols = len(T_values)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    
    filepath = get_table_filename(eta, Y_C, B, a)
    data = load_table(filepath)
    
    if data is None:
        return fig, axes
    
    for i, T in enumerate(T_values):
        ax = axes[i]
        data_T = filter_temperature(data, T)
        
        if len(data_T.get('n_B', [])) == 0:
            continue
        
        n_B = data_T['n_B']
        x = n_B / N0 if normalize_nB else n_B
        P = data_T['P_total']
        chi = data_T.get('chi', np.zeros_like(P))
        
        # Color by phase
        colors = np.where(chi < 0.01, 'blue', np.where(chi > 0.99, 'green', 'purple'))
        
        ax.scatter(x, P, c=colors, s=10, alpha=0.7)
        ax.plot(x, P, 'k-', linewidth=0.5, alpha=0.5)
        
        # Mark boundaries
        if bounds is not None:
            mask_eta = np.abs(bounds.get('eta', np.zeros(len(bounds['T']))) - eta) < 0.01
            mask_T = np.abs(bounds['T'] - T) < 1.0
            mask = mask_eta & mask_T
            if np.any(mask):
                n_onset = bounds['n_onset'][mask][0]
                n_offset = bounds['n_offset'][mask][0]
                if normalize_nB:
                    n_onset, n_offset = n_onset / N0, n_offset / N0
                ax.axvline(n_onset, color='red', linestyle='--', linewidth=1, label='onset')
                ax.axvline(n_offset, color='red', linestyle=':', linewidth=1, label='offset')
        
        ax.set_xlabel(r'$n_B/n_0$' if normalize_nB else r'$n_B$ [fm$^{-3}$]')
        if i == 0:
            ax.set_ylabel(r'$P$ [MeV/fm$^3$]')
        ax.set_title(f'T = {T:.0f} MeV')
        
        if xlim:
            ax.set_xlim(xlim)
    
    fig.suptitle(f'Pressure at Phase Transitions ($Y_C = {Y_C:.2f}$, {ETA_STYLES.get(eta, {}).get("label", f"η={eta}")})', fontsize=12)
    plt.tight_layout()
    
    if save:
        save_path = PLOT_DIR / f"pressure_discontinuity_YC{Y_C:.2f}_eta{eta:.2f}.pdf"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    
    return fig, axes


# ==============================================================================
#                           MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SFHo+AlphaBag Hybrid EOS Plotting")
    print("=" * 60)
    
    # Create plot directory
    PLOT_DIR.mkdir(exist_ok=True)
    
    # Generate all plots for Y_C = 0.5
    Y_C = 0.5
    
    print(f"\nGenerating plots for Y_C = {Y_C}...")
    
    # 1. Phase boundaries
    print("  [1/5] Phase boundaries...")
    plot_phase_boundaries(Y_C=Y_C, save=True, show=False)
    
    # 2. Chi vs nB
    print("  [2/5] Chi vs nB...")
    plot_chi_vs_nB(Y_C=Y_C, T_values=[10.0, 50.0, 70.0, 100.0], save=True, show=False)
    
    # 3. Pressure vs nB
    print("  [3/5] Pressure vs nB...")
    plot_pressure_vs_nB(Y_C=Y_C, T_values=[10.0, 50.0, 100.0], save=True, show=False)
    
    # 4. 4-panel thermo for T=50 MeV
    print("  [4/5] 4-panel thermodynamics (T=50 MeV)...")
    plot_4panel_thermo(Y_C=Y_C, T=50.0, save=True, show=False)
    
    # 5. Pressure discontinuity check (Maxwell)
    print("  [5/5] Pressure discontinuity (Maxwell)...")
    plot_pressure_discontinuity(Y_C=Y_C, T_values=[10.0, 50.0, 100.0], eta=1.0, save=True, show=False)
    
    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("=" * 60)
