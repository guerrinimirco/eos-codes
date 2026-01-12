"""
compare_with_compose.py
========================
Compare my Python EOS tables with CompOSE SFHo/SFHoY tables at fixed YC.

Reads CompOSE files (eos.nb, eos.t, eos.yq, eos.thermo, eos.compo) and computes
tables using my Python implementation, then generates comparison plots.

CompOSE file format:
- eos.nb: baryon densities [fm^-3], columns: [1, N_points, n_B_1, n_B_2, ...]
- eos.t: temperatures [MeV], columns: [1, N_points, T_1, T_2, ...]
- eos.yq: charge fractions, columns: [1, N_points, Y_C_1, Y_C_2, ...]
- eos.thermo: [i_T, i_nB, i_Yq, P/nB, s/nB, μB/mN-1, μC/mN, μL/mN, f/mN/nB-1, ε/mN/nB-1, flag]
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os

# My modules
from sfho_eos import SFHoEOS
from sfho_compute_tables import compute_table, TableSettings, results_to_arrays


#==============================================================================
# COMPOSE FILE READER
#==============================================================================
@dataclass
class CompOSEData:
    """Container for CompOSE EOS data."""
    name: str
    m_N: float  # nucleon mass used in CompOSE (from eos.thermo header)
    
    # Grid arrays
    n_B_values: np.ndarray  # fm^-3
    T_values: np.ndarray    # MeV
    Y_C_values: np.ndarray  # charge fraction
    
    # 3D arrays indexed by (i_T, i_nB, i_Yq)
    # Physical quantities (in physical units, not normalized)
    P: np.ndarray       # MeV/fm^3
    s: np.ndarray       # per baryon (entropy per baryon)
    e: np.ndarray       # MeV/fm^3
    f: np.ndarray       # MeV/fm^3 (free energy density)
    mu_B: np.ndarray    # MeV
    mu_C: np.ndarray    # MeV (charge chemical potential)
    mu_L: np.ndarray    # MeV (lepton chemical potential)


def read_compose_data(compose_dir: str, name: str = "CompOSE") -> CompOSEData:
    """
    Read CompOSE EOS files from a directory.
    
    Args:
        compose_dir: Path to directory containing eos.nb, eos.t, eos.yq, eos.thermo
        name: Name label for this dataset
    
    Returns:
        CompOSEData object with all thermodynamic quantities
    """
    # Read grid files
    # Each file: line 1 = "1", line 2 = "N_points", lines 3+ = values
    
    def read_grid_file(filename):
        with open(os.path.join(compose_dir, filename), 'r') as f:
            lines = f.readlines()
        # Skip first two lines (1, N_points), then parse values
        values = []
        for line in lines[2:]:
            line = line.strip()
            if line:
                values.append(float(line))
        return np.array(values)
    
    n_B_values = read_grid_file("eos.nb")
    T_values = read_grid_file("eos.t")
    Y_C_values = read_grid_file("eos.yq")
    
    n_nB = len(n_B_values)
    n_T = len(T_values)
    n_Yq = len(Y_C_values)
    
    print(f"Reading CompOSE data from {compose_dir}")
    print(f"  n_B: {n_nB} points, range [{n_B_values.min():.2e}, {n_B_values.max():.2e}] fm^-3")
    print(f"  T:   {n_T} points, range [{T_values.min():.2f}, {T_values.max():.2f}] MeV")
    print(f"  Y_C: {n_Yq} points, range [{Y_C_values.min():.3f}, {Y_C_values.max():.3f}]")
    
    # Read thermo file
    # Header line: m_N_neutron m_N_proton flag
    # Data lines: i_T i_nB i_Yq P/nB s/nB muB/mN-1 muC/mN muL/mN f/mN/nB-1 e/mN/nB-1 flag
    thermo_path = os.path.join(compose_dir, "eos.thermo")
    
    with open(thermo_path, 'r') as f:
        header = f.readline().split()
        m_N_neutron = float(header[0])
        m_N_proton = float(header[1])
        m_N = (m_N_neutron + m_N_proton) / 2  # average
        
        # Read all data lines
        data = []
        for line in f:
            parts = line.split()
            if len(parts) >= 10:
                data.append([float(x) for x in parts])
    
    data = np.array(data)
    print(f"  m_N (neutron) = {m_N_neutron:.5f} MeV (proton = {m_N_proton:.5f})")
    print(f"  Thermo data: {len(data)} points")
    
    # Initialize 3D arrays with shape (n_T, n_nB, n_Yq)
    P = np.full((n_T, n_nB, n_Yq), np.nan)
    s = np.full((n_T, n_nB, n_Yq), np.nan)
    e = np.full((n_T, n_nB, n_Yq), np.nan)
    f = np.full((n_T, n_nB, n_Yq), np.nan)
    mu_B = np.full((n_T, n_nB, n_Yq), np.nan)
    mu_C = np.full((n_T, n_nB, n_Yq), np.nan)
    mu_L = np.full((n_T, n_nB, n_Yq), np.nan)
    
    # Parse data and fill arrays
    # Columns: i_T(0), i_nB(1), i_Yq(2), P/nB(3), s/nB(4), muB/mN-1(5), muC/mN(6), muL/mN(7), f/mN/nB-1(8), e/mN/nB-1(9)
    # NOTE: CompOSE uses m_n (neutron mass) for Q3-Q7 normalization, not average
    for row in data:
        i_T = int(row[0]) - 1   # Convert to 0-indexed
        i_nB = int(row[1]) - 1
        i_Yq = int(row[2]) - 1
        
        if 0 <= i_T < n_T and 0 <= i_nB < n_nB and 0 <= i_Yq < n_Yq:
            n_B = n_B_values[i_nB]
            
            # Convert from normalized to physical units using m_N_neutron (per CompOSE guide)
            P[i_T, i_nB, i_Yq] = row[3] * n_B                          # P/n_B * n_B = P [MeV/fm^3]
            s[i_T, i_nB, i_Yq] = row[4]                                # s per baryon (dimensionless)
            mu_B[i_T, i_nB, i_Yq] = (row[5] + 1) * m_N_neutron         # (μB/mN - 1 + 1) * mN = μB [MeV]
            mu_C[i_T, i_nB, i_Yq] = row[6] * m_N_neutron               # μC/mN * mN = μC [MeV]
            mu_L[i_T, i_nB, i_Yq] = row[7] * m_N_neutron               # μL/mN * mN = μL [MeV]
            f[i_T, i_nB, i_Yq] = (row[8] + 1) * m_N_neutron * n_B      # (f/mN/nB - 1 + 1) * mN * nB = f [MeV/fm^3]
            e[i_T, i_nB, i_Yq] = (row[9] + 1) * m_N_neutron * n_B      # (e/mN/nB - 1 + 1) * mN * nB = e [MeV/fm^3]
    
    return CompOSEData(
        name=name,
        m_N=m_N_neutron,  # Store neutron mass as the reference
        n_B_values=n_B_values,
        T_values=T_values,
        Y_C_values=Y_C_values,
        P=P, s=s, e=e, f=f, mu_B=mu_B, mu_C=mu_C, mu_L=mu_L
    )


def get_compose_slice(compose: CompOSEData, Y_C: float, T: float, 
                      interpolate_T: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract a 1D slice from CompOSE data at fixed Y_C and T.
    
    Args:
        compose: CompOSE data object
        Y_C: Target charge fraction  
        T: Target temperature [MeV]
        interpolate_T: If True, linearly interpolate between T grid points.
                       If False, use nearest neighbor (old behavior).
    
    Returns dict with n_B, P, e, s, mu_B, mu_C arrays.
    """
    # Find nearest Y_C index (no interpolation in Y_C)
    i_Yq = np.argmin(np.abs(compose.Y_C_values - Y_C))
    actual_Y_C = compose.Y_C_values[i_Yq]
    
    T_grid = compose.T_values
    
    if interpolate_T and T >= T_grid.min() and T <= T_grid.max():
        # Find bracketing indices for interpolation
        i_T_low = np.searchsorted(T_grid, T, side='right') - 1
        i_T_low = max(0, min(i_T_low, len(T_grid) - 2))
        i_T_high = i_T_low + 1
        
        T_low = T_grid[i_T_low]
        T_high = T_grid[i_T_high]
        
        # Check if we're very close to a grid point (avoid interpolation noise)
        if np.abs(T - T_low) < 0.01:  # Within 0.01 MeV
            i_T = i_T_low
            use_interp = False
        elif np.abs(T - T_high) < 0.01:
            i_T = i_T_high
            use_interp = False
        else:
            use_interp = True
            # Linear interpolation weight
            w = (T - T_low) / (T_high - T_low)
            
        if use_interp:
            print(f"  Requested: T={T:.2f} MeV, Y_C={Y_C:.3f}")
            print(f"  Interpolating between T={T_low:.2f} and T={T_high:.2f} MeV (w={w:.3f})")
            
            # Interpolate all quantities
            def interp_quantity(arr_3d):
                """Interpolate 3D array along T axis at fixed i_Yq."""
                low = arr_3d[i_T_low, :, i_Yq]
                high = arr_3d[i_T_high, :, i_Yq]
                return (1 - w) * low + w * high
            
            return {
                'n_B': compose.n_B_values,
                'P': interp_quantity(compose.P),
                'e': interp_quantity(compose.e),
                's': interp_quantity(compose.s),
                'mu_B': interp_quantity(compose.mu_B),
                'mu_C': interp_quantity(compose.mu_C),
                'mu_L': interp_quantity(compose.mu_L),
                'f': interp_quantity(compose.f),
                'T': T,  # Return the requested T
                'Y_C': actual_Y_C,
            }
    
    # Fallback: nearest neighbor
    i_T = np.argmin(np.abs(T_grid - T))
    actual_T = T_grid[i_T]
    
    print(f"  Requested: T={T:.2f} MeV, Y_C={Y_C:.3f}")
    print(f"  Using:     T={actual_T:.2f} MeV, Y_C={actual_Y_C:.3f}")
    
    return {
        'n_B': compose.n_B_values,
        'P': compose.P[i_T, :, i_Yq],
        'e': compose.e[i_T, :, i_Yq],
        's': compose.s[i_T, :, i_Yq],
        'mu_B': compose.mu_B[i_T, :, i_Yq],
        'mu_C': compose.mu_C[i_T, :, i_Yq],
        'mu_L': compose.mu_L[i_T, :, i_Yq],
        'f': compose.f[i_T, :, i_Yq],
        'T': actual_T,
        'Y_C': actual_Y_C,
    }


#==============================================================================
# LOAD PRE-COMPUTED MY TABLES
#==============================================================================
# Path to pre-computed tables directory
MY_TABLES_DIR = "/Users/mircoguerrini/Desktop/Research/Python_codes/sfho_tables_output"

# Mapping from (parametrization, particle_content) to filename
MY_TABLE_FILES = {
    ('sfho', 'nucleons'): 'eos_sfho_nucleons_fixed_YC_neutral_no_muons.dat',
    ('sfhoy', 'nucleons_hyperons'): 'eos_sfhoy_nucleons_hyperons_fixed_YC_neutral_no_muons.dat',
}


def load_my_table_file(parametrization: str, particle_content: str) -> Dict[str, np.ndarray]:
    """
    Load pre-computed EOS table from file.
    
    Returns:
        Dict with column names as keys and numpy arrays as values
    """
    key = (parametrization, particle_content)
    if key not in MY_TABLE_FILES:
        raise ValueError(f"No pre-computed table for {parametrization}, {particle_content}")
    
    filepath = os.path.join(MY_TABLES_DIR, MY_TABLE_FILES[key])
    print(f"\nLoading my table from: {filepath}")
    
    # Read header to get column names
    with open(filepath, 'r') as f:
        f.readline()  # Skip first comment line
        f.readline()  # Skip second comment line
        header_line = f.readline()
    
    # Parse column names from header (strip # and whitespace)
    columns = header_line.strip().lstrip('#').split()
    
    # Load data
    data = np.loadtxt(filepath, comments='#')
    
    # Create dict with column names
    result = {}
    for i, col in enumerate(columns):
        result[col] = data[:, i]
    
    print(f"  Loaded {len(data)} points")
    print(f"  n_B range: [{result['n_B'].min():.4e}, {result['n_B'].max():.4e}] fm^-3")
    print(f"  T range: [{result['T'].min():.1f}, {result['T'].max():.1f}] MeV")
    print(f"  Y_C range: [{result['Y_C'].min():.3f}, {result['Y_C'].max():.3f}]")
    
    return result


def get_my_table_slice(full_data: Dict[str, np.ndarray], 
                       T: float, Y_C: float) -> Dict[str, np.ndarray]:
    """
    Extract a slice at fixed T and Y_C from the full loaded table.
    
    Args:
        full_data: Full table loaded by load_my_table_file
        T: Temperature [MeV]
        Y_C: Charge fraction
    
    Returns:
        Dict with n_B, P_total, e_total, s_total, mu_B, mu_C arrays for matching points
    """
    # Find points matching T and Y_C (with tolerance)
    T_tol = 0.5  # MeV tolerance
    Y_C_tol = 0.01  # fraction tolerance
    
    mask = (np.abs(full_data['T'] - T) < T_tol) & (np.abs(full_data['Y_C'] - Y_C) < Y_C_tol)
    
    if np.sum(mask) == 0:
        print(f"  WARNING: No data found for T={T:.1f} MeV, Y_C={Y_C:.3f}")
        # Find closest available
        unique_T = np.unique(full_data['T'])
        unique_Y_C = np.unique(full_data['Y_C'])
        closest_T = unique_T[np.argmin(np.abs(unique_T - T))]
        closest_Y_C = unique_Y_C[np.argmin(np.abs(unique_Y_C - Y_C))]
        print(f"  Using closest: T={closest_T:.1f} MeV, Y_C={closest_Y_C:.3f}")
        mask = (np.abs(full_data['T'] - closest_T) < T_tol) & (np.abs(full_data['Y_C'] - closest_Y_C) < Y_C_tol)
    
    # Extract data
    result = {}
    for key in full_data:
        result[key] = full_data[key][mask]
    
    # IMPORTANT: Sort all arrays by n_B to ensure proper line plotting
    if 'n_B' in result and len(result['n_B']) > 0:
        sort_idx = np.argsort(result['n_B'])
        for key in result:
            result[key] = result[key][sort_idx]
    
    # Add converged column (all True for loaded data)
    result['converged'] = np.ones(len(result['n_B']), dtype=bool)
    
    # mu_C = -mu_Q in my convention (if not already present)
    if 'mu_C' not in result and 'mu_Q' in result:
        result['mu_C'] = -result['mu_Q']
    
    print(f"  Extracted {len(result['n_B'])} points for T={T:.1f} MeV, Y_C={Y_C:.3f}")
    
    return result


# Keep old function for backwards compatibility (but recommend using new one)
def compute_my_table(parametrization: str, particle_content: str,
                     n_B_values: np.ndarray, T: float, Y_C: float,
                     include_muons: bool = False) -> Dict[str, np.ndarray]:
    """
    DEPRECATED: Use load_my_table_file + get_my_table_slice instead.
    
    Compute EOS table using my Python implementation.
    """
    print("WARNING: Computing table on-the-fly. Consider using pre-computed tables.")
    
    settings = TableSettings(
        parametrization=parametrization,
        particle_content=particle_content,
        equilibrium='fixed_yc_neutral',
        n_B_values=n_B_values,
        T_values=[T],
        Y_C_values=Y_C,
        include_muons=include_muons,
        print_results=False,
        print_timing=True,
        save_to_file=False,
    )
    
    print(f"\nComputing my table: {parametrization}, {particle_content}")
    print(f"  T = {T} MeV, Y_C = {Y_C}")
    print(f"  n_B range: [{n_B_values.min():.4e}, {n_B_values.max():.4e}] fm^-3")
    
    results = compute_table(settings)
    key = list(results.keys())[0]
    data = results_to_arrays(results[key])
    
    # mu_C = -mu_Q in my convention
    data['mu_C'] = -data['mu_Q']
    
    return data


#==============================================================================
# PLOTTING FUNCTIONS (using general_plotting_info)
#==============================================================================
from general_plotting_info import set_global_style, FONTS, STYLE, COLORS, LABELS, apply_style

# Nuclear saturation density
N_SAT = 0.158  # fm^-3


def plot_comparison(compose_data: Dict, my_data: Dict, 
                    title_prefix: str = "",
                    save_path: Optional[str] = None,
                    n_B_range: Optional[Tuple[float, float]] = None):
    """
    Create comparison plots for thermodynamic quantities.
    
    Plots: P, e, s, mu_B, mu_C vs n_B (linear scale)
    
    Args:
        compose_data: CompOSE data slice
        my_data: My code data slice
        title_prefix: Prefix for plot title
        save_path: Path to save figure (optional)
        n_B_range: (n_B_min, n_B_max) in fm^-3, default (0.1*n_sat, 10*n_sat)
    """
    set_global_style()
    
    # Default density range: 0.1*n_sat to 10*n_sat
    if n_B_range is None:
        n_B_range = (0.1 * N_SAT, 10.0 * N_SAT)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=STYLE['dpi'])
    fig.suptitle(f"{title_prefix}T = {compose_data['T']:.1f} MeV, Y_C = {compose_data['Y_C']:.3f}", 
                 fontsize=FONTS['title'] + 2, fontweight='bold')
    
    # Filter data to density range
    n_B_comp = compose_data['n_B']
    n_B_my = my_data['n_B']
    
    # Valid mask: within range and not NaN
    range_mask_comp = (n_B_comp >= n_B_range[0]) & (n_B_comp <= n_B_range[1])
    range_mask_my = (n_B_my >= n_B_range[0]) & (n_B_my <= n_B_range[1])
    
    valid_comp = range_mask_comp & ~np.isnan(compose_data['P'])
    valid_my = range_mask_my & my_data['converged'].astype(bool)
    
    n_B_comp_valid = n_B_comp[valid_comp]
    n_B_my_valid = n_B_my[valid_my]
    
    # Extract mu_C with sign convention fix
    # CompOSE: mu_C = mu_Q (negative for electrons)
    # My code: mu_C = -mu_Q = mu_e (positive)
    # To compare: negate my mu_C to match CompOSE convention
    my_mu_C = my_data['mu_C'][valid_my]
    comp_mu_C = compose_data['mu_C'][valid_comp]
    
    # Negate my mu_C to match CompOSE sign convention (mu_C = mu_Q, not mu_e)
    my_mu_C_corrected = -my_mu_C
    
    # Quantities to compare: (comp_key, my_data_array, ylabel)
    quantities = [
        ('P', my_data['P_total'][valid_my], r'$P$ [MeV fm$^{-3}$]'),
        ('e', my_data['e_total'][valid_my], r'$\varepsilon$ [MeV fm$^{-3}$]'),
        ('s', my_data['s_total'][valid_my], r'$s/n_B$'),
        ('mu_B', my_data['mu_B'][valid_my], r'$\mu_B$ [MeV]'),
        ('mu_C', my_mu_C_corrected, r'$\mu_C$ [MeV]'),
    ]
    
    colors = {'compose': COLORS['jel'], 'my': COLORS['reference']}
    
    for idx, (comp_key, my_arr, ylabel) in enumerate(quantities):
        ax = axes.flat[idx]
        
        # CompOSE data
        if comp_key == 'mu_C':
            y_comp = comp_mu_C
        else:
            y_comp = compose_data[comp_key][valid_comp]
        
        # Plot with linear x-axis (n_B in units of n_sat)
        ax.plot(n_B_comp_valid / N_SAT, y_comp, 'o-', color=colors['compose'], 
                label='CompOSE', markersize=4, linewidth=STYLE['linewidth'], alpha=0.8)
        ax.plot(n_B_my_valid / N_SAT, my_arr, 's--', color=colors['my'], 
                label='My code', markersize=4, linewidth=STYLE['linewidth'], alpha=0.8)
        
        ax.set_xlabel(r'$n_B / n_{\mathrm{sat}}$', fontsize=FONTS['label'])
        ax.set_ylabel(ylabel, fontsize=FONTS['label'])
        ax.legend(loc='best', fontsize=FONTS['legend'])
        ax.tick_params(labelsize=FONTS['tick'])
        ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_style'])
        
        # Set x-axis limits
        ax.set_xlim(n_B_range[0] / N_SAT, n_B_range[1] / N_SAT)
    
    # Sixth panel: relative difference in pressure
    ax = axes.flat[5]
    
    # Interpolate my data onto CompOSE grid for diff calculation
    if len(n_B_my_valid) > 2 and len(n_B_comp_valid) > 2:
        from scipy.interpolate import interp1d
        
        P_my_interp = interp1d(n_B_my_valid, my_data['P_total'][valid_my], 
                                kind='linear', bounds_error=False, fill_value=np.nan)
        P_my_on_comp = P_my_interp(n_B_comp_valid)
        
        P_comp_valid = compose_data['P'][valid_comp]
        
        # Avoid division by very small numbers
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(np.abs(P_comp_valid) > 1e-6,
                               (P_my_on_comp - P_comp_valid) / P_comp_valid * 100,
                               np.nan)
        
        ax.plot(n_B_comp_valid / N_SAT, rel_diff, 'k-', linewidth=STYLE['linewidth'])
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(r'$n_B / n_{\mathrm{sat}}$', fontsize=FONTS['label'])
        ax.set_ylabel(r'$\Delta P / P$ [%]', fontsize=FONTS['label'])
        
        max_diff = np.nanmax(np.abs(rel_diff))
        ax.set_title(f'(My - CompOSE)/CompOSE, max |ΔP| = {max_diff:.2f}%', 
                    fontsize=FONTS['label'])
        ax.tick_params(labelsize=FONTS['tick'])
        ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_style'])
        ax.set_xlim(n_B_range[0] / N_SAT, n_B_range[1] / N_SAT)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=STYLE['dpi'], bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()
    
    return fig


def add_neutrino_to_compose(compose_slice: Dict, T: float) -> Dict:
    """
    Add thermal neutrino contributions to CompOSE data.
    
    CompOSE tables typically don't include neutrinos. This adds thermal (μ=0)
    neutrino contributions to P and ε for a fairer comparison.
    
    For each neutrino flavor (νe, νμ, ντ), adds:
    - P_ν = (7/8) * (π²/45) * T⁴ / ℏc³  (per flavor, particle + antiparticle)
    - ε_ν = 3 * P_ν
    - s_ν = (4/T) * P_ν
    
    Args:
        compose_slice: CompOSE data slice from get_compose_slice
        T: Temperature in MeV
        
    Returns:
        New dict with corrected P, e, s
    """
    from general_thermodynamics_leptons import neutrino_thermo
    
    # Compute thermal neutrino contribution (3 flavors, μ=0)
    nu_thermo = neutrino_thermo(0.0, T, include_antiparticles=True)
    
    # Total for 3 flavors
    P_nu_total = 3 * nu_thermo.P
    e_nu_total = 3 * nu_thermo.e
    s_nu_total = 3 * nu_thermo.s
    
    # Create corrected copy
    corrected = dict(compose_slice)
    corrected['P'] = compose_slice['P'] + P_nu_total
    corrected['e'] = compose_slice['e'] + e_nu_total
    # s is per baryon, so need to convert: s_total/n_B
    n_B = compose_slice['n_B']
    s_per_baryon = compose_slice['s']  # Already per baryon
    # Add neutrino entropy per baryon
    corrected['s'] = s_per_baryon + s_nu_total / n_B
    
    return corrected


def plot_comparison_multi_T(
    compose_data,  # CompOSEData object
    my_full_data: Dict,  # Full loaded table
    T_values: List[float],
    Y_C: float,
    title: str = "",
    save_path: Optional[str] = None,
    n_B_range: Optional[Tuple[float, float]] = None,
    add_neutrinos: bool = True
):
    """
    Create comparison plots with multiple temperatures on the same axes.
    
    Args:
        compose_data: CompOSEData object (full dataset)
        my_full_data: Full my table dict from load_my_table_file
        T_values: List of temperatures to compare
        Y_C: Charge fraction
        title: Overall title
        save_path: Path to save figure
        n_B_range: (n_B_min, n_B_max) in fm^-3
        add_neutrinos: If True, add thermal neutrino contributions to CompOSE
    """
    set_global_style()
    
    # Default density range
    if n_B_range is None:
        n_B_range = (0.1 * N_SAT, 10.0 * N_SAT)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=STYLE['dpi'])
    fig.suptitle(f"{title}$Y_C$ = {Y_C:.2f}" + (" (with thermal $\\nu$)" if add_neutrinos else ""), 
                 fontsize=FONTS['title'] + 2, fontweight='bold')
    
    # Custom colors for temperatures: black, orange, red
    T_colors = {
        0.1: 'black',
        10.0: '#ff7f0e',  # Orange
        50.0: '#d62728',  # Red
    }
    # Fallback for other temperatures
    default_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#8c564b', '#e377c2']
    for i, T in enumerate(T_values):
        if T not in T_colors:
            T_colors[T] = default_colors[i % len(default_colors)]
    
    quantities = [
        ('P', 'P_total', r'$P$ [MeV fm$^{-3}$]'),
        ('e', 'e_total', r'$\varepsilon$ [MeV fm$^{-3}$]'),
        ('s', 's_total', r'$s/n_B$'),
        ('mu_B', 'mu_B', r'$\mu_B$ [MeV]'),
        ('mu_C', 'mu_C', r'$\mu_C$ [MeV]'),
    ]
    
    for T in T_values:
        color = T_colors[T]
        
        # Get data slices
        compose_slice = get_compose_slice(compose_data, Y_C, T)
        
        # Add neutrino contribution if requested
        if add_neutrinos:
            compose_slice = add_neutrino_to_compose(compose_slice, compose_slice['T'])
        
        my_slice = get_my_table_slice(my_full_data, T, Y_C)
        
        # Filter to range
        n_B_comp = compose_slice['n_B']
        n_B_my = my_slice['n_B']
        
        range_mask_comp = (n_B_comp >= n_B_range[0]) & (n_B_comp <= n_B_range[1])
        range_mask_my = (n_B_my >= n_B_range[0]) & (n_B_my <= n_B_range[1])
        
        valid_comp = range_mask_comp & ~np.isnan(compose_slice['P'])
        valid_my = range_mask_my & my_slice['converged'].astype(bool)
        
        n_B_comp_valid = n_B_comp[valid_comp]
        n_B_my_valid = n_B_my[valid_my]
        
        # Fix mu_C sign convention
        my_mu_C_corrected = -my_slice['mu_C'][valid_my]
        
        for idx, (comp_key, my_key, ylabel) in enumerate(quantities):
            ax = axes.flat[idx]
            
            # Get data
            if comp_key == 'mu_C':
                y_comp = compose_slice['mu_C'][valid_comp]
                y_my = my_mu_C_corrected
            elif comp_key == 's':
                # CompOSE has s/n_B (entropy per baryon)
                # My table has s_total (entropy density), need to divide by n_B
                y_comp = compose_slice['s'][valid_comp]
                y_my = my_slice['s_total'][valid_my] / n_B_my_valid  # s_total / n_B = s per baryon
            else:
                y_comp = compose_slice[comp_key][valid_comp]
                y_my = my_slice[my_key][valid_my]
            
            # Plot - CompOSE solid, My code dashed
            ax.plot(n_B_comp_valid / N_SAT, y_comp, '-', color=color, 
                    linewidth=STYLE['linewidth'], alpha=0.9)
            ax.plot(n_B_my_valid / N_SAT, y_my, '--', color=color, 
                    linewidth=STYLE['linewidth'], alpha=0.9)
            
            ax.set_xlabel(r'$n_B / n_{\mathrm{sat}}$', fontsize=FONTS['label'])
            ax.set_ylabel(ylabel, fontsize=FONTS['label'])
            ax.tick_params(labelsize=FONTS['tick'])
            ax.grid(True, alpha=STYLE['grid_alpha'], linestyle=STYLE['grid_style'])
            ax.set_xlim(n_B_range[0] / N_SAT, n_B_range[1] / N_SAT)
    
    # Sixth panel: Combined legend
    ax = axes.flat[5]
    ax.axis('off')
    
    # Temperature legend (colored lines)
    y_start = 0.85
    ax.text(0.5, y_start, 'Temperature:', transform=ax.transAxes, 
            fontsize=FONTS['label'], ha='center', fontweight='bold')
    for i, T in enumerate(T_values):
        y_pos = y_start - 0.12 * (i + 1)
        ax.plot([0.2, 0.4], [y_pos, y_pos], '-', color=T_colors[T], 
                linewidth=3, transform=ax.transAxes)
        ax.text(0.45, y_pos, f'$T$ = {T:.1f} MeV', transform=ax.transAxes,
                fontsize=FONTS['label'], va='center')
    
    # Model legend (line styles)
    y_model = y_start - 0.12 * (len(T_values) + 1.5)
    ax.text(0.5, y_model, 'Model:', transform=ax.transAxes,
            fontsize=FONTS['label'], ha='center', fontweight='bold')
    ax.plot([0.2, 0.4], [y_model - 0.1, y_model - 0.1], '-', color='gray', 
            linewidth=2, transform=ax.transAxes)
    ax.text(0.45, y_model - 0.1, 'CompOSE', transform=ax.transAxes,
            fontsize=FONTS['label'], va='center')
    ax.plot([0.2, 0.4], [y_model - 0.22, y_model - 0.22], '--', color='gray', 
            linewidth=2, transform=ax.transAxes)
    ax.text(0.45, y_model - 0.22, 'My code', transform=ax.transAxes,
            fontsize=FONTS['label'], va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=STYLE['dpi'], bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()
    
    return fig


#==============================================================================
# MAIN COMPARISON ROUTINE
#==============================================================================
def run_comparison(
    compose_dir: str,
    compose_name: str,
    my_parametrization: str,
    my_particle_content: str,
    T_values: List[float],
    Y_C_values: List[float],
    n_B_range: Optional[Tuple[float, float]] = None,
    n_B_points: int = 100,
    include_muons: bool = False,
    save_plots: bool = True,
    output_dir: str = "compose_comparison_plots",
    use_precomputed: bool = True
):
    """
    Run full comparison between CompOSE and my EOS tables.
    
    Args:
        compose_dir: Directory containing CompOSE files
        compose_name: Name for CompOSE dataset (e.g., 'SFHo')
        my_parametrization: 'sfho' or 'sfhoy' for my code
        my_particle_content: Particle content string
        T_values: List of temperatures to compare
        Y_C_values: List of charge fractions to compare
        n_B_range: Optional (min, max) density range (only used if not using precomputed)
        n_B_points: Number of density points (only used if not using precomputed)
        include_muons: Whether to include muons
        save_plots: Whether to save plots
        output_dir: Directory for saving plots
        use_precomputed: If True, load from pre-computed table files (MUCH faster)
    """
    print("=" * 70)
    print(f"COMPARISON: {compose_name} (CompOSE) vs {my_parametrization} (my code)")
    print("=" * 70)
    
    # Read CompOSE data
    compose = read_compose_data(compose_dir, compose_name)
    
    # Load my pre-computed table once (if using precomputed)
    my_full_data = None
    if use_precomputed:
        try:
            my_full_data = load_my_table_file(my_parametrization, my_particle_content)
        except (ValueError, FileNotFoundError) as e:
            print(f"WARNING: Could not load pre-computed table: {e}")
            print("Falling back to on-the-fly computation")
            use_precomputed = False
    
    # Determine density range (only for on-the-fly computation)
    if not use_precomputed:
        if n_B_range is None:
            n_B_min = max(compose.n_B_values.min(), 0.01)
            n_B_max = min(compose.n_B_values.max(), 1.0)
        else:
            n_B_min, n_B_max = n_B_range
        n_B_values = np.logspace(np.log10(n_B_min), np.log10(n_B_max), n_B_points)
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Compare for each (T, Y_C) combination
    for T in T_values:
        for Y_C in Y_C_values:
            print(f"\n{'='*50}")
            print(f"Comparing at T = {T} MeV, Y_C = {Y_C}")
            print("=" * 50)
            
            # Get CompOSE slice
            compose_slice = get_compose_slice(compose, Y_C, T)
            
            # Get my data (from precomputed or compute on-the-fly)
            if use_precomputed and my_full_data is not None:
                my_data = get_my_table_slice(my_full_data, T, Y_C)
            else:
                my_data = compute_my_table(
                    my_parametrization, my_particle_content,
                    n_B_values, T, Y_C, include_muons
                )
            
            # Generate plot
            title = f"{compose_name} vs {my_parametrization}: "
            if save_plots:
                save_path = os.path.join(
                    output_dir, 
                    f"compare_{compose_name}_vs_{my_parametrization}_T{T:.0f}_YC{Y_C:.2f}.png"
                )
            else:
                save_path = None
            
            plot_comparison(compose_slice, my_data, title, save_path)


#==============================================================================
# CONFIGURATION
#==============================================================================
# Paths to CompOSE data
COMPOSE_SFHO_DIR = "/Users/mircoguerrini/Desktop/Research/Compose/SFHO_Compose"
COMPOSE_SFHOY_DIR = "/Users/mircoguerrini/Desktop/Research/Compose/SFHOY_Compose"

# Comparison settings
COMPARISON_SETTINGS = {
    # SFHo comparison
    'sfho': {
        'compose_dir': COMPOSE_SFHO_DIR,
        'compose_name': 'SFHo',
        'my_parametrization': 'sfho',
        'my_particle_content': 'nucleons',  # SFHo CompOSE is nucleons only
    },
    # SFHoY comparison
    'sfhoy': {
        'compose_dir': COMPOSE_SFHOY_DIR,
        'compose_name': 'SFHoY',
        'my_parametrization': 'sfhoy',
        'my_particle_content': 'nucleons_hyperons',  # SFHoY has hyperons
    }
}

# Default temperatures and charge fractions to compare
DEFAULT_T_VALUES = [0.1, 10.0, 50.0]
DEFAULT_YC_VALUES = [0.1, 0.3, 0.5]


#==============================================================================
# MAIN
#==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CompOSE vs My EOS Comparison Tool")
    print("=" * 70 + "\n")
    
    # Run SFHo comparison
    print("\n\n" + "#" * 70)
    print("# SFHo COMPARISON")
    print("#" * 70)
    
    run_comparison(
        **COMPARISON_SETTINGS['sfho'],
        T_values=[100],
        Y_C_values=[0.1, 0.5],
        n_B_range=(0.5* N_SAT, 10* N_SAT),
        n_B_points=50,
        include_muons=False,
        save_plots=True,
        output_dir="compose_comparison_plots"
    )
    
    # Run SFHoY comparison
    print("\n\n" + "#" * 70)
    print("# SFHoY COMPARISON")
    print("#" * 70)
    
    run_comparison(
        **COMPARISON_SETTINGS['sfhoy'],
        T_values=[100],
        Y_C_values=[0.1, 0.5],
        n_B_range=(0.5* N_SAT, 10* N_SAT),
        n_B_points=50,
        include_muons=False,
        save_plots=True,
        output_dir="compose_comparison_plots"
    )
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
