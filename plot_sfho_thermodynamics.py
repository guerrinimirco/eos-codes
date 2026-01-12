"""
plot_sfho_thermodynamics.py
===========================
Script to plot thermodynamic quantities from SFHo and SFHoY EOS tables.

This script generates publication-quality plots of:
1. Thermodynamic quantities (P, ε, s/nB) vs nB/n0
2. Chemical potentials (μB, μC, μS, μQ) vs nB/n0  
3. Meson fields (σ, ω, ρ, φ) vs nB/n0
4. Particle fractions (Yi) for different particle species

Plots are created at multiple temperatures (T = 0.1, 10, 50, 100 MeV).

Usage:
    python plot_sfho_thermodynamics.py
    
Or import and call specific functions:
    from plot_sfho_thermodynamics import plot_thermodynamic_quantities, plot_particle_fractions
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import plotting configuration
from general_plotting_info import (
    set_global_style, setup_scientific_figure, add_panel_labels, apply_style,
    format_axes, STYLE, FONTS, LABELS, PARTICLE_COLORS, PARTICLE_STYLES,
    COLORS_SEQ, save_figure
)

# Import EOS modules for particle fraction computation
from sfho_eos import SFHoEOS
from sfho_parameters import get_sfhoy_fortin
from sfho_thermodynamics_hadrons import compute_hadron_thermo
from general_fermi_integrals import solve_fermi_jel
from general_particles import Proton, Neutron, Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM

# =============================================================================
# CONSTANTS
# =============================================================================
N0 = 0.16  # Nuclear saturation density [fm^-3]
TEMPERATURES = [0.1, 10.0, 50.0, 100.0]  # MeV - key temperatures for plots
Y_C_VALUES = [0.1, 0.3, 0.5]  # Charge fractions to consider

# Table column names
COLUMNS = ['n_B', 'Y_C', 'T', 'sigma', 'omega', 'rho', 'phi',
           'mu_B', 'mu_C', 'mu_S', 'mu_Q', 'mu_L',
           'P_total', 'e_total', 's_total', 'Y_S', 'Y_Q', 'Y_L', 'converged']


# =============================================================================
# DATA LOADING
# =============================================================================
def load_eos_table(filepath):
    """
    Load EOS table from file.
    
    Parameters:
        filepath: Path to .dat file
    
    Returns:
        pandas DataFrame with EOS data
    """
    data = pd.read_csv(filepath, sep=r'\s+', comment='#', names=COLUMNS)
    data['n_B_n0'] = data['n_B'] / N0  # Add normalized density column
    data['s_nB'] = data['s_total'] / data['n_B']  # Entropy per baryon
    return data


def filter_data(df, T=None, Y_C=None, converged_only=True):
    """Filter DataFrame by temperature, charge fraction, and convergence."""
    mask = pd.Series([True] * len(df))
    
    if T is not None:
        mask &= np.isclose(df['T'], T, rtol=0.01)
    if Y_C is not None:
        mask &= np.isclose(df['Y_C'], Y_C, rtol=0.01)
    if converged_only:
        mask &= df['converged'] == 1
    
    return df[mask].sort_values('n_B')


# =============================================================================
# PARTICLE FRACTION COMPUTATION
# =============================================================================
def compute_particle_fractions(n_B, T, sigma, omega, rho, phi, mu_B, mu_Q, mu_S,
                               params=None, include_hyperons=True):
    """
    Compute individual particle fractions from EOS solution.
    
    Given the meson fields and chemical potentials from the EOS solution,
    compute the number density and fraction Y_i = n_i / n_B for each species.
    
    Parameters:
        n_B, T, sigma, omega, rho, phi: EOS state
        mu_B, mu_Q, mu_S: Chemical potentials
        params: SFHoParams (default: SFHoY Fortin)
        include_hyperons: Whether to include hyperon species
    
    Returns:
        dict: {particle_name: Y_i fraction}
    """
    if params is None:
        params = get_sfhoy_fortin()
    
    fractions = {}
    
    # Define particles and their properties
    particles = [
        ('p', Proton, 1, 0, 0.5),     # name, particle, B, S, I3
        ('n', Neutron, 1, 0, -0.5),
    ]
    
    if include_hyperons:
        particles.extend([
            ('Lambda', Lambda, 1, 1, 0),  # S=+1 in our convention
            ('Sigma+', SigmaP, 1, 1, 1),
            ('Sigma0', Sigma0, 1, 1, 0),
            ('Sigma-', SigmaM, 1, 1, -1),
            ('Xi0', Xi0, 1, 2, 0.5),
            ('Xi-', XiM, 1, 2, -0.5),
        ])
    
    for name, particle, B, S, I3 in particles:
        # Get coupling ratios
        if name in ['p', 'n']:
            x_s, x_w, x_r, x_p = 1.0, 1.0, 1.0, 0.0
        elif name == 'Lambda':
            x_s = params.x_sigma_lambda
            x_w = params.x_omega_lambda
            x_r = 0.0  # Isoscalar
            x_p = params.x_phi_lambda
        elif name.startswith('Sigma'):
            x_s = params.x_sigma_sigma
            x_w = params.x_omega_sigma
            x_r = 2.0  # I=1
            x_p = params.x_phi_sigma
        elif name.startswith('Xi'):
            x_s = params.x_sigma_xi
            x_w = params.x_omega_xi
            x_r = 1.0  # I=1/2
            x_p = params.x_phi_xi
        else:
            x_s, x_w, x_r, x_p = 1.0, 1.0, 1.0, 0.0
        
        # Effective mass and chemical potential
        m_eff = particle.mass - params.g_sigma * x_s * sigma
        
        # Chemical potential: μ_i = B*μ_B + Q*μ_Q + S*μ_S
        Q = particle.charge
        mu_bare = B * mu_B + Q * mu_Q + S * mu_S
        
        # Vector field shifts
        mu_eff = mu_bare - (params.g_omega * x_w * omega 
                           + params.g_rho * x_r * I3 * rho
                           + params.g_phi * x_p * phi)
        
        # Skip if effective mass is invalid
        if m_eff <= 0:
            fractions[name] = 0.0
            continue
        
        # Compute number density using Fermi integral
        try:
            result = solve_fermi_jel(mu_eff, T, m_eff, particle.g_degen)
            n_i = result[0]  # Number density
            fractions[name] = max(0, n_i / n_B) if n_B > 0 else 0.0
        except:
            fractions[name] = 0.0
    
    # Electron fraction (from charge neutrality for fixed Y_C case)
    # Y_e = Y_C - Σ_i Y_i * Q_i (for hadrons)
    # But we have Y_Q in the table which should equal Y_C for neutral case
    
    return fractions


def compute_all_fractions_for_table(df, include_hyperons=True):
    """
    Compute particle fractions for each row in the table.
    
    Returns DataFrame with Y_p, Y_n, Y_Lambda, etc. columns added.
    """
    params = get_sfhoy_fortin()
    
    # Initialize fraction columns
    particle_names = ['p', 'n']
    if include_hyperons:
        particle_names.extend(['Lambda', 'Sigma+', 'Sigma0', 'Sigma-', 'Xi0', 'Xi-'])
    
    for name in particle_names:
        df[f'Y_{name}'] = 0.0
    
    # Compute fractions row by row
    for idx, row in df.iterrows():
        fracs = compute_particle_fractions(
            row['n_B'], row['T'], row['sigma'], row['omega'], 
            row['rho'], row['phi'], row['mu_B'], row['mu_Q'], row['mu_S'],
            params=params, include_hyperons=include_hyperons
        )
        for name, y in fracs.items():
            df.at[idx, f'Y_{name}'] = y
    
    return df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_thermodynamic_quantities(df_sfho, df_sfhoy, Y_C=0.5, 
                                  T_values=[0.1, 10.0, 50.0, 100.0],
                                  save_path=None):
    """
    Plot P, ε, s/nB vs nB/n0 in 2x2 grid.
    
    Each panel shows curves for SFHo (solid) and SFHoY (dashed) at one temperature.
    """
    set_global_style()
    fig, axes = setup_scientific_figure(2, 2, gray_background=True)
    axes = axes.flatten()
    
    for i, T in enumerate(T_values):
        ax = axes[i]
        
        # Filter data
        data_sfho = filter_data(df_sfho, T=T, Y_C=Y_C)
        data_sfhoy = filter_data(df_sfhoy, T=T, Y_C=Y_C)
        
        # Plot pressure
        ax.semilogy(data_sfho['n_B_n0'], data_sfho['P_total'], 
                   'b-', lw=STYLE['linewidth'], label='SFHo P')
        ax.semilogy(data_sfhoy['n_B_n0'], data_sfhoy['P_total'], 
                   'b--', lw=STYLE['linewidth'], label='SFHoY P')
        
        # Plot energy density
        ax.semilogy(data_sfho['n_B_n0'], data_sfho['e_total'], 
                   'r-', lw=STYLE['linewidth'], label=r'SFHo $\varepsilon$')
        ax.semilogy(data_sfhoy['n_B_n0'], data_sfhoy['e_total'], 
                   'r--', lw=STYLE['linewidth'], label=r'SFHoY $\varepsilon$')
        
        # Labels and styling
        ax.set_xlabel(LABELS['nB_n0'], fontsize=FONTS['label'])
        ax.set_ylabel(r'[MeV fm$^{-3}$]', fontsize=FONTS['label'])
        ax.set_title(f'$T = {T:.1f}$ MeV', fontsize=FONTS['title'])
        apply_style(ax, legend=(i==0))
        ax.set_xlim(0, 10)
    
    add_panel_labels(axes)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, axes


def plot_chemical_potentials(df_sfho, df_sfhoy, Y_C=0.5,
                            T_values=[0.1, 10.0, 50.0, 100.0],
                            save_path=None):
    """
    Plot μB, μC, μS, μQ vs nB/n0 in 2x2 grid.
    """
    set_global_style()
    fig, axes = setup_scientific_figure(2, 2, gray_background=True)
    axes = axes.flatten()
    
    mu_labels = ['mu_B', 'mu_C', 'mu_S', 'mu_Q']
    mu_names = [r'$\mu_B$', r'$\mu_C$', r'$\mu_S$', r'$\mu_Q$']
    
    for i, (mu_col, mu_name) in enumerate(zip(mu_labels, mu_names)):
        ax = axes[i]
        
        for j, T in enumerate(T_values):
            color = COLORS_SEQ[j]
            
            data_sfho = filter_data(df_sfho, T=T, Y_C=Y_C)
            data_sfhoy = filter_data(df_sfhoy, T=T, Y_C=Y_C)
            
            ax.plot(data_sfho['n_B_n0'], data_sfho[mu_col], 
                   color=color, ls='-', lw=STYLE['linewidth'],
                   label=f'T={T}' if i==0 else None)
            ax.plot(data_sfhoy['n_B_n0'], data_sfhoy[mu_col], 
                   color=color, ls='--', lw=STYLE['linewidth'])
        
        ax.set_xlabel(LABELS['nB_n0'], fontsize=FONTS['label'])
        ax.set_ylabel(f'{mu_name} [MeV]', fontsize=FONTS['label'])
        ax.set_title(mu_name, fontsize=FONTS['title'])
        apply_style(ax, legend=(i==0))
        ax.set_xlim(0, 10)
    
    add_panel_labels(axes)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, axes


def plot_meson_fields(df_sfho, df_sfhoy, Y_C=0.5,
                     T_values=[0.1, 10.0, 50.0, 100.0],
                     save_path=None):
    """
    Plot σ, ω, ρ, φ fields vs nB/n0 in 2x2 grid.
    """
    set_global_style()
    fig, axes = setup_scientific_figure(2, 2, gray_background=True)
    axes = axes.flatten()
    
    field_cols = ['sigma', 'omega', 'rho', 'phi']
    field_names = [r'$\sigma$', r'$\omega$', r'$\rho$', r'$\phi$']
    
    for i, (field, name) in enumerate(zip(field_cols, field_names)):
        ax = axes[i]
        
        for j, T in enumerate(T_values):
            color = COLORS_SEQ[j]
            
            data_sfho = filter_data(df_sfho, T=T, Y_C=Y_C)
            data_sfhoy = filter_data(df_sfhoy, T=T, Y_C=Y_C)
            
            ax.plot(data_sfho['n_B_n0'], data_sfho[field], 
                   color=color, ls='-', lw=STYLE['linewidth'],
                   label=f'T={T}' if i==0 else None)
            ax.plot(data_sfhoy['n_B_n0'], data_sfhoy[field], 
                   color=color, ls='--', lw=STYLE['linewidth'])
        
        ax.set_xlabel(LABELS['nB_n0'], fontsize=FONTS['label'])
        ax.set_ylabel(f'{name} [MeV]', fontsize=FONTS['label'])
        ax.set_title(name, fontsize=FONTS['title'])
        apply_style(ax, legend=(i==0))
        ax.set_xlim(0, 10)
    
    add_panel_labels(axes)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, axes


def plot_particle_fractions(df, T_values=[0.1, 10.0, 50.0, 100.0], Y_C=0.5,
                           include_hyperons=True, save_path=None):
    """
    Plot particle fractions Y_i vs nB/n0 in 4-panel figure.
    
    Each panel shows a different temperature, with all particle species.
    """
    set_global_style()
    fig, axes = setup_scientific_figure(2, 2, gray_background=True)
    axes = axes.flatten()
    
    # Compute fractions if not already done
    if 'Y_p' not in df.columns:
        df = compute_all_fractions_for_table(df, include_hyperons=include_hyperons)
    
    particle_names = ['p', 'n']
    if include_hyperons:
        particle_names.extend(['Lambda', 'Sigma+', 'Sigma0', 'Sigma-', 'Xi0', 'Xi-'])
    
    for i, T in enumerate(T_values):
        ax = axes[i]
        data = filter_data(df, T=T, Y_C=Y_C)
        
        if len(data) == 0:
            continue
        
        for name in particle_names:
            col = f'Y_{name}'
            if col in data.columns:
                y_vals = data[col].values
                # Only plot if there are nonzero values
                if np.any(y_vals > 1e-4):
                    color = PARTICLE_COLORS.get(name, 'black')
                    style = PARTICLE_STYLES.get(name, '-')
                    ax.semilogy(data['n_B_n0'], y_vals, 
                               color=color, ls=style, lw=STYLE['linewidth'],
                               label=name)
        
        ax.set_xlabel(LABELS['nB_n0'], fontsize=FONTS['label'])
        ax.set_ylabel(LABELS['Y_i'], fontsize=FONTS['label'])
        ax.set_title(f'$T = {T:.1f}$ MeV, $Y_C = {Y_C}$', fontsize=FONTS['title'])
        ax.set_ylim(1e-3, 2)
        ax.set_xlim(0, 10)
        apply_style(ax, legend=True)
    
    add_panel_labels(axes)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig, axes


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Generate all plots from EOS tables."""
    print("Loading EOS tables...")
    
    # Define paths
    table_dir = Path('sfho_tables_output')
    
    # Load tables
    sfho_file = table_dir / 'eos_sfho_nucleons_fixed_YC_neutral_no_muons.dat'
    sfhoy_file = table_dir / 'eos_sfhoy_nucleons_hyperons_fixed_YC_neutral_no_muons.dat'
    
    if not sfho_file.exists():
        print(f"Error: {sfho_file} not found")
        return
    if not sfhoy_file.exists():
        print(f"Error: {sfhoy_file} not found")
        return
    
    df_sfho = load_eos_table(sfho_file)
    df_sfhoy = load_eos_table(sfhoy_file)
    
    print(f"Loaded SFHo: {len(df_sfho)} points")
    print(f"Loaded SFHoY: {len(df_sfhoy)} points")
    
    # Create output directory
    output_dir = Path('eos_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating thermodynamic plots...")
    plot_thermodynamic_quantities(df_sfho, df_sfhoy, Y_C=0.5,
                                  save_path=str(output_dir / 'thermo_P_epsilon'))
    
    print("Generating chemical potential plots...")
    plot_chemical_potentials(df_sfho, df_sfhoy, Y_C=0.5,
                            save_path=str(output_dir / 'chemical_potentials'))
    
    print("Generating meson field plots...")
    plot_meson_fields(df_sfho, df_sfhoy, Y_C=0.5,
                     save_path=str(output_dir / 'meson_fields'))
    
    print("Computing particle fractions (this may take a moment)...")
    df_sfhoy = compute_all_fractions_for_table(df_sfhoy.copy(), include_hyperons=True)
    
    print("Generating particle fraction plots...")
    plot_particle_fractions(df_sfhoy, Y_C=0.5, include_hyperons=True,
                           save_path=str(output_dir / 'particle_fractions'))
    
    print(f"\nPlots saved to {output_dir}/")
    plt.show()


if __name__ == '__main__':
    main()
