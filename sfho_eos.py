"""
sfho_eos.py
===========
Single-point solvers for SFHo relativistic mean-field EOS.

This module provides solvers for different equilibrium conditions:
- Beta equilibrium (charge neutrality with leptons, strangeness β-eq: μ_S = 0)
- Fixed charge fraction Y_C (hadrons only, or with leptons for charge neutrality)
- Fixed charge and strangeness fractions Y_C, Y_S
- Trapped neutrinos (fixed Y_L = (n_e + n_ν)/n_B)

All thermodynamic functions are in sfho_thermodynamics_hadrons.py.

Usage:
    from sfho_eos import solve_sfho_beta_eq, solve_sfho_fixed_yc
    from sfho_parameters import get_sfho_2fam_phi
    from particles import Proton, Neutron, Lambda, ...
    
    params = get_sfho_2fam_phi()
    particles = [Proton, Neutron, Lambda, ...]
    
    result = solve_sfho_beta_eq(n_B=0.16, T=10.0, params=params, particles=particles)
    print(f"P = {result.P_total} MeV/fm³")

References:
- Fortin, Oertel, Providência, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17

see also:
- Guerrini, PhD Thesis (2026)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from scipy.optimize import root

from general_particles import (
    Particle, Proton, Neutron,
    Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM,
    DeltaPP, DeltaP, Delta0, DeltaM
)
from sfho_parameters import (
    SFHoParams, get_sfho_nucleonic, get_sfhoy_fortin,
    get_sfhoy_star_fortin, get_sfho_2fam_phi
)
from sfho_thermodynamics_hadrons import (
    compute_field_residuals,
    compute_meson_contribution,
    compute_sfho_thermo_from_mu_fields
)
from general_thermodynamics_leptons import electron_thermo, photon_thermo, neutrino_thermo, electron_thermo_from_density
from general_physics_constants import hc, hc3, PI2


# =============================================================================
# PARTICLE LISTS
# =============================================================================
NUCLEONS = [Proton, Neutron]
HYPERONS = [Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
DELTAS = [DeltaPP, DeltaP, Delta0, DeltaM]

BARYONS_N = NUCLEONS
BARYONS_NY = NUCLEONS + HYPERONS
BARYONS_NYD = NUCLEONS + HYPERONS + DELTAS


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class SFHoEOSResult:
    """Complete result from SFHo EOS calculation at one point."""
    # Convergence info
    converged: bool = False
    error: float = 0.0
    
    # Inputs
    n_B: float = 0.0       # Baryon density (fm⁻³)
    T: float = 0.0         # Temperature (MeV)
    Y_C: float = 0.0       # Charge fraction n_C/n_B
    Y_S: float = 0.0       # Strangeness fraction n_S/n_B
    Y_L: float = 0.0       # Lepton fraction (n_e + n_nu)/n_B
    
    # Meson fields (MeV)
    sigma: float = 0.0
    omega: float = 0.0
    rho: float = 0.0
    phi: float = 0.0
    
    # Chemical potentials (MeV)
    mu_B: float = 0.0
    mu_C: float = 0.0
    mu_S: float = 0.0
    mu_e: float = 0.0      # Electron chemical potential
    mu_L: float = 0.0      # Lepton chemical potential (for trapped neutrinos)
    mu_nu: float = 0.0     # Neutrino chemical potential (0 for non-trapped modes)
    
    # Densities (fm⁻³)
    n_C: float = 0.0       # Charge density
    n_S: float = 0.0   # Strangeness density
    n_e: float = 0.0       # Electron density
    n_nu: float = 0.0      # Neutrino density
    
    # Thermodynamics (MeV/fm³ for P, e; fm⁻³ for s)
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    f_total: float = 0.0   # Free energy density
    
    # Component contributions
    P_hadrons: float = 0.0
    P_leptons: float = 0.0
    P_photons: float = 0.0
    
    # Detailed baryon info
    baryon_densities: Dict[str, float] = field(default_factory=dict)
    m_eff: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# INITIAL GUESS GENERATION
# =============================================================================
def get_default_guess_beta_eq(
    n_B: float, T: float, params: SFHoParams
) -> np.ndarray:
    """
    Generate initial guess for beta equilibrium: [σ, ω, ρ, φ, μ_B, μ_C].
    
    Based on self-consistent solutions for symmetric nuclear matter:
    - At saturation (n_B = 0.16 fm⁻³): σ ~ 30 MeV, ω ~ 19 MeV
    - Fields scale roughly linearly with density
    - μ_C ~ -20 to -100 MeV (increasing magnitude with density)
    """
    n_sat = 0.158
    ratio = n_B / n_sat
    
    # Field estimates (scale linearly with density, capped)
    sigma = min(30.0 * ratio, 100.0)
    omega = min(19.0 * ratio, 80.0)
    
    # μ_B estimate (nearly constant, slight increase with density)
    if ratio < 0.5:
        mu_B = 940.0 + 5.0 * ratio
    elif ratio < 2.0:
        mu_B = 942.0 + 40.0 * (ratio - 0.5)
    else:
        mu_B = 1002.0 + 80.0 * (ratio - 2.0)
    
    # μ_C estimate (negative, magnitude increases with density)
    # At low n_B: μ_C ~ -20 MeV, at saturation: μ_C ~ -50 MeV
    # These values match known solutions from old tables
    if ratio < 0.1:
        mu_C = -20.0 - 100.0 * ratio  # -20 to -30 MeV
    elif ratio < 0.5:
        mu_C = -30.0 - 40.0 * ratio   # -30 to -50 MeV  
    elif ratio < 1.5:
        mu_C = -50.0 - 40.0 * (ratio - 0.5)  # -50 to -90 MeV
    else:
        mu_C = -90.0 - 30.0 * (ratio - 1.5)  # -90 and beyond
    
    return np.array([sigma, omega, 0.0, 0.0, mu_B, mu_C])


def get_default_guess_fixed_yc(
    n_B: float, Y_C: float, T: float, params: SFHoParams
) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C: [σ, ω, ρ, φ, μ_B, μ_C].
    """
    n_sat = 0.158
    ratio = n_B / n_sat
    
    # Field estimates (scale linearly with density, capped)
    sigma = min(30.0 * ratio, 100.0)
    omega = min(19.0 * ratio, 80.0)
    
    # ρ field scales with isospin asymmetry: negative for neutron-rich (Y_C < 0.5)
    # At saturation, |ρ| ~ 5 MeV for pure neutron matter
    asymmetry = 0.5 - Y_C  # Positive for neutron-rich
    rho = -5.0 * min(ratio, 2.0) * asymmetry / 0.5
    
    # φ = 0 for nucleons only
    phi = 0.0
    
    # μ_B estimate (nearly constant around nucleon mass, increases with density)
    if ratio < 0.5:
        mu_B = 940.0 + 5.0 * ratio
    elif ratio < 2.0:
        mu_B = 942.0 + 40.0 * (ratio - 0.5)
    else:
        mu_B = 1002.0 + 80.0 * (ratio - 2.0)
    
    # μ_C: more negative for neutron-rich matter
    # Y_C = 0.5 → symmetric, μ_C ~ 0
    # Y_C = 0 → pure neutron, μ_C ~ -100 to -150 MeV
    # Also scales with density
    base_mu_C = -150.0 * asymmetry  # -75 to 0 for Y_C=0 to 0.5
    density_factor = min(ratio, 2.0) / 1.0  # Increases with density
    mu_C = base_mu_C * (0.5 + 0.5 * density_factor)
    
    return np.array([sigma, omega, rho, phi, mu_B, mu_C])


def get_default_guess_fixed_yc_ys(
    n_B: float, Y_C: float, Y_S: float, T: float, params: SFHoParams
) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C and Y_S: [σ, ω, ρ, φ, μ_B, μ_C, μ_S].
    
    Based on analysis of converged solutions at n_B=0.01583:
    - Y_C=0.5, Y_S=0.5-0.9: mu_B~910-920, mu_C~15-35, mu_S~195-210 MeV
    - μ_S scales weakly with Y_S (~2 MeV per 0.1 Y_S)
    """
    guess_yc = get_default_guess_fixed_yc(n_B, Y_C, T, params)
    
    n_sat = 0.158
    ratio = n_B / n_sat
    
    # For symmetric matter with strangeness (Y_C ~0.5), use observed pattern
    if Y_C > 0.4 and Y_S > 0.3:
        phi = -0.5 * Y_S
        
        # At low density: mu_S ~ 195 + 15*Y_S (from converged solutions)
        # mu_C ~ 15 + 25*Y_S (increases with strangeness)
        if ratio < 0.5:
            mu_S = 195.0 + 20.0 * Y_S
            mu_C = 15.0 + 30.0 * Y_S
            mu_B = 920.0 - 10.0 * Y_S  # slight decrease with Y_S
        else:
            # Higher density: adjust
            mu_S = (195.0 + 20.0 * Y_S) * (1.0 - 0.3 * (ratio - 0.5))
            mu_C = (15.0 + 30.0 * Y_S) * (1.0 + 0.5 * ratio)
            mu_B = 920.0 + 50.0 * ratio
        
        return np.array([guess_yc[0], guess_yc[1], guess_yc[2], phi,
                         mu_B, mu_C, mu_S])
    
    # Default case for low Y_C or low Y_S
    phi = -0.5 * min(ratio, 5.0) * Y_S
    
    # Base mu_S depends mainly on density
    if ratio < 0.5:
        mu_S_base = 180.0
    elif ratio < 1.0:
        mu_S_base = 130.0 - 30.0 * (ratio - 0.5)
    elif ratio < 2.0:
        mu_S_base = 105.0 - 80.0 * (ratio - 1.0)
    elif ratio < 4.0:
        mu_S_base = 25.0 - 40.0 * (ratio - 2.0)
    else:
        mu_S_base = -55.0 - 20.0 * (ratio - 4.0)
    
    # Weak Y_S scaling
    if Y_S > 0:
        mu_S = mu_S_base * (1.0 + 0.1 * np.log10(Y_S / 0.1 + 1))
    else:
        mu_S = 0.0
    
    return np.array([guess_yc[0], guess_yc[1], guess_yc[2], phi,
                     guess_yc[4], guess_yc[5], mu_S])


def get_default_guess_trapped(
    n_B: float, Y_L: float, T: float, params: SFHoParams
) -> np.ndarray:
    """
    Generate initial guess for trapped neutrinos: [σ, ω, ρ, φ, μ_B, μ_C, μ_L].
    
    In trapped neutrino regime, Y_L = (n_e + n_ν)/n_B is fixed,
    and μ_L is the lepton chemical potential (μ_e = μ_L - μ_C).
    """
    guess_beta = get_default_guess_beta_eq(n_B, T, params)
    
    # Estimate μ_L from Y_L (higher Y_L needs higher μ_L)
    mu_L_est = 10.0 + 50.0 * Y_L
    
    return np.array([guess_beta[0], guess_beta[1], guess_beta[2], guess_beta[3],
                     guess_beta[4], guess_beta[5], mu_L_est])


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_sfho_beta_eq(
    n_B: float, T: float,
    params: SFHoParams,
    particles: List[Particle],
    include_photons: bool = True,
    include_muons: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> SFHoEOSResult:
    """
    Solve SFHo EOS in beta equilibrium.
    
    Solves 6 equations for 6 unknowns: [σ, ω, ρ, φ, μ_B, μ_C]
    (μ_S = 0 for strangeness β-equilibrium)
    
    Equations:
        1-4. Field equations (σ, ω, ρ, φ self-consistency)
        5.   n_B = n_B_target (baryon number conservation)
        6.   n_C_hadrons + n_C_mesons + n_Q_leptons = 0 (charge neutrality)
    
    Args:
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        params: SFHo parameters
        particles: List of baryon species
        include_photons: Include photon contributions
        include_muons: Include muons (not implemented yet)
        include_pseudoscalar_mesons: Include π, K, η meson contributions
        initial_guess: Initial guess [σ, ω, ρ, φ, μ_B, μ_C]
        
    Returns:
        SFHoEOSResult with all thermodynamic quantities
    """
    result = SFHoEOSResult(n_B=n_B, T=T)
    
    if initial_guess is None:
        x0 = get_default_guess_beta_eq(n_B, T, params)
    else:
        x0 = initial_guess
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C = x
        mu_S = 0.0  # Strangeness β-equilibrium
        
        # Compute hadron thermodynamics (includes source terms and optional pseudoscalar mesons)
        hadron = compute_sfho_thermo_from_mu_fields(
            mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Field equation residuals using source terms from hadron
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        # Electron thermodynamics (μ_e = -μ_C in beta equilibrium)
        mu_e = -mu_C
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        
        # Normalize field residuals by m² × σ₀ (MeV³) to make them dimensionless
        sigma_0 = 30.0
        eq1 = res_sigma / (params.m_sigma**2 * sigma_0)
        eq2 = res_omega / (params.m_omega**2 * sigma_0)
        eq3 = res_rho / (params.m_rho**2 * sigma_0)
        eq4 = res_phi / (params.m_phi**2 * sigma_0)
        
        # Baryon number constraint
        eq5 = (hadron.n_B - n_B)
        
        # Charge neutrality: n_C_hadrons (includes meson charge) = n_e
        eq6 = (hadron.n_C - e_thermo.n)
        
        return [eq1, eq2, eq3, eq4, eq5, eq6]
    
    # Solve with hybr, fallback to lm
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = (error < 1e-6) or (sol.success and error < 1e-4)
    result.error = error
    
    # Store fields and chemical potentials
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    result.mu_e = -mu_C
    
    # Compute final thermodynamics using unified function
    thermo = compute_sfho_thermo_from_mu_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    # Electron thermodynamics
    e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
    
    # Store densities (thermo.n_C already includes meson charge if enabled)
    result.n_C = thermo.n_C
    result.n_S_val = thermo.n_S
    result.n_e = e_thermo.n
    result.Y_C = thermo.Y_C
    result.Y_S = thermo.Y_S
    
    # Thermodynamics
    result.P_hadrons = thermo.P
    result.P_leptons = e_thermo.P
    
    result.P_total = thermo.P + e_thermo.P
    result.e_total = thermo.e + e_thermo.e
    result.s_total = thermo.s + e_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    
    # Baryon details (from thermo.states)
    result.baryon_densities = {name: state.n for name, state in thermo.states.items()}
    result.m_eff = {name: state.m_eff for name, state in thermo.states.items()}
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C
# =============================================================================
def solve_sfho_fixed_yc(
    n_B: float, Y_C: float, T: float,
    params: SFHoParams,
    particles: List[Particle],
    include_electrons: bool = False,
    include_photons: bool = False,
    include_thermal_neutrinos: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> SFHoEOSResult:
    """
    Solve SFHo EOS with fixed charge fraction Y_C.
    
    Solves 6 equations for 6 unknowns: [σ, ω, ρ, φ, μ_B, μ_C]
    (μ_S = 0 for strangeness β-equilibrium)
    
    Equations:
        1-4. Field equations (σ, ω, ρ, φ self-consistency)
        5.   n_B = n_B_target
        6.   n_C / n_B = Y_C_target (includes meson charge if enabled)
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction n_C/n_B
        T: Temperature (MeV)
        params: SFHo parameters
        particles: List of baryon species
        include_electrons: If True, add electrons with n_e = n_C for neutrality
        include_photons: Include photon contributions
        include_pseudoscalar_mesons: Include π, K, η meson contributions
        initial_guess: Initial guess [σ, ω, ρ, φ, μ_B, μ_C]
        
    Returns:
        SFHoEOSResult with all thermodynamic quantities
    """
    result = SFHoEOSResult(n_B=n_B, T=T, Y_C=Y_C)
    
    if initial_guess is None:
        x0 = get_default_guess_fixed_yc(n_B, Y_C, T, params)
    else:
        x0 = initial_guess
    
    n_C_target = Y_C * n_B
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C = x
        mu_S = 0.0
        
        # Compute hadron thermodynamics (includes source terms and optional pseudoscalar mesons)
        hadron = compute_sfho_thermo_from_mu_fields(
            mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Field equation residuals using source terms from hadron
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        # Normalize field residuals
        sigma_0 = 30.0
        eq1 = res_sigma / (params.m_sigma**2 * sigma_0)
        eq2 = res_omega / (params.m_omega**2 * sigma_0)
        eq3 = res_rho / (params.m_rho**2 * sigma_0)
        eq4 = res_phi / (params.m_phi**2 * sigma_0)
        eq5 = (hadron.n_B - n_B)
        
        # Fixed Y_C: n_C (includes meson charge) = Y_C * n_B
        eq6 = (hadron.n_C - n_C_target)
        
        return [eq1, eq2, eq3, eq4, eq5, eq6]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = (error < 1e-6) or (sol.success and error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    
    # Compute final thermodynamics using unified function
    thermo = compute_sfho_thermo_from_mu_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    result.n_C = thermo.n_C
    result.n_S_val = thermo.n_S
    result.Y_C = thermo.Y_C
    result.Y_S = thermo.Y_S
    
    result.P_hadrons = thermo.P
    result.P_total = thermo.P
    result.e_total = thermo.e
    result.s_total = thermo.s
    
    if include_electrons:
        # Get mu_e_guess from initial_guess if provided
        if initial_guess is not None and len(initial_guess) > 6:
            mu_e_guess = initial_guess[6]
        else:
            mu_e_guess = None
        # Use electron_thermo_from_density to find μ_e that gives n_e = n_C
        e_result = electron_thermo_from_density(thermo.n_C, T, mu_e_guess=mu_e_guess)
        result.mu_e = e_result.mu
        result.n_e = e_result.n
        result.P_leptons = e_result.P
        result.P_total += e_result.P
        result.e_total += e_result.e
        result.s_total += e_result.s
    
    if include_thermal_neutrinos and T > 0:
        # Thermal neutrinos with μ_ν = 0 (3 flavors)
        nu_thermo = neutrino_thermo(0.0, T, include_antiparticles=True)
        result.P_total += 3.0 * nu_thermo.P
        result.e_total += 3.0 * nu_thermo.e
        result.s_total += 3.0 * nu_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    
    # Baryon details
    result.baryon_densities = {name: state.n for name, state in thermo.states.items()}
    result.m_eff = {name: state.m_eff for name, state in thermo.states.items()}
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C AND Y_S
# =============================================================================
def solve_sfho_fixed_yc_ys(
    n_B: float, Y_C: float, Y_S: float, T: float,
    params: SFHoParams,
    particles: List[Particle],
    include_electrons: bool = False,
    include_photons: bool = False,
    include_thermal_neutrinos: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> SFHoEOSResult:
    """
    Solve SFHo EOS with fixed charge AND strangeness fractions.
    
    Solves 7 equations for 7 unknowns: [σ, ω, ρ, φ, μ_B, μ_C, μ_S]
    
    Equations:
        1-4. Field equations
        5.   n_B = n_B_target
        6.   n_C / n_B = Y_C_target (includes meson charge if enabled)
        7.   n_S / n_B = Y_S_target (includes meson strangeness if enabled)
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction n_C/n_B
        Y_S: Strangeness fraction n_S/n_B
        T: Temperature (MeV)
        params: SFHo parameters
        particles: List of baryon species (should include hyperons)
        include_electrons: If True, add electrons with n_e = n_C
        include_photons: Include photon contributions
        include_pseudoscalar_mesons: Include π, K, η meson contributions
        initial_guess: Initial guess [σ, ω, ρ, φ, μ_B, μ_C, μ_S]
        
    Returns:
        SFHoEOSResult with all thermodynamic quantities
    """
    result = SFHoEOSResult(n_B=n_B, T=T, Y_C=Y_C, Y_S=Y_S)
    
    if initial_guess is None:
        x0 = get_default_guess_fixed_yc_ys(n_B, Y_C, Y_S, T, params)
    else:
        x0 = initial_guess
    
    n_C_target = Y_C * n_B
    n_S_target = Y_S * n_B
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C, mu_S = x
        
        # Compute hadron thermodynamics (includes source terms and optional pseudoscalar mesons)
        hadron = compute_sfho_thermo_from_mu_fields(
            mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Field equation residuals using source terms from hadron
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        # Normalize field residuals
        sigma_0 = 30.0
        eq1 = res_sigma / (params.m_sigma**2 * sigma_0)
        eq2 = res_omega / (params.m_omega**2 * sigma_0)
        eq3 = res_rho / (params.m_rho**2 * sigma_0)
        eq4 = res_phi / (params.m_phi**2 * sigma_0)
        eq5 = (hadron.n_B - n_B)
        
        # Fixed Y_C and Y_S: n_C and n_S (includes meson contributions) = targets
        eq6 = (hadron.n_C - n_C_target)
        eq7 = (hadron.n_S - n_S_target)
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C, mu_S = sol.x
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = (error < 1e-6) or (sol.success and error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    
    # Compute final thermodynamics using unified function
    thermo = compute_sfho_thermo_from_mu_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    result.n_C = thermo.n_C
    result.n_S_val = thermo.n_S
    result.Y_C = thermo.Y_C
    result.Y_S = thermo.Y_S
    
    result.P_hadrons = thermo.P
    result.P_total = thermo.P
    result.e_total = thermo.e
    result.s_total = thermo.s
    
    if include_electrons:
        # Get mu_e_guess from initial_guess if provided
        if initial_guess is not None and len(initial_guess) > 7:
            mu_e_guess = initial_guess[7]
        else:
            mu_e_guess = None
        # Use electron_thermo_from_density to find μ_e that gives n_e = n_C
        e_result = electron_thermo_from_density(thermo.n_C, T, mu_e_guess=mu_e_guess)
        result.mu_e = e_result.mu
        result.n_e = e_result.n
        result.P_leptons = e_result.P
        result.P_total += e_result.P
        result.e_total += e_result.e
        result.s_total += e_result.s
    
    if include_thermal_neutrinos and T > 0:
        # Thermal neutrinos with μ_ν = 0 (3 flavors)
        nu_thermo = neutrino_thermo(0.0, T, include_antiparticles=True)
        result.P_total += 3.0 * nu_thermo.P
        result.e_total += 3.0 * nu_thermo.e
        result.s_total += 3.0 * nu_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    
    # Baryon details
    result.baryon_densities = {name: state.n for name, state in thermo.states.items()}
    result.m_eff = {name: state.m_eff for name, state in thermo.states.items()}
    
    return result


# =============================================================================
# SOLVER: TRAPPED NEUTRINOS (FIXED Y_L)
# =============================================================================
def solve_sfho_trapped_neutrinos(
    n_B: float, Y_L: float, T: float,
    params: SFHoParams,
    particles: List[Particle],
    include_photons: bool = True,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> SFHoEOSResult:
    """
    Solve SFHo EOS with trapped neutrinos (fixed lepton fraction Y_L).
    
    Solves 7 equations for 7 unknowns: [σ, ω, ρ, φ, μ_B, μ_C, μ_L]
    (μ_S = 0 for strangeness β-equilibrium)
    
    In this regime, neutrinos are trapped (before neutrino sphere),
    and μ_e = μ_L - μ_C (beta equilibrium with trapped neutrinos).
    
    Equations:
        1-4. Field equations (σ, ω, ρ, φ self-consistency)
        5.   n_B = n_B_target
        6.   n_C - n_e = 0 (charge neutrality, includes meson charge if enabled)
        7.   (n_e + n_ν)/n_B = Y_L (fixed lepton fraction)
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_L: Lepton fraction (n_e + n_ν)/n_B
        T: Temperature (MeV)
        params: SFHo parameters
        particles: List of baryon species
        include_photons: Include photon contributions
        include_pseudoscalar_mesons: Include π, K, η meson contributions
        initial_guess: Initial guess [σ, ω, ρ, φ, μ_B, μ_C, μ_L]
        
    Returns:
        SFHoEOSResult with all thermodynamic quantities
    """
    result = SFHoEOSResult(n_B=n_B, T=T, Y_L=Y_L)
    
    if initial_guess is None:
        x0 = get_default_guess_trapped(n_B, Y_L, T, params)
    else:
        x0 = initial_guess
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C, mu_L = x
        mu_S = 0.0
        
        # Compute hadron thermodynamics (includes source terms and optional pseudoscalar mesons)
        hadron = compute_sfho_thermo_from_mu_fields(
            mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Field equation residuals using source terms from hadron
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        # Electron and neutrino: μ_e = μ_L - μ_C (trapped beta eq)
        mu_e = mu_L - mu_C
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_thermo = neutrino_thermo(mu_L, T, include_antiparticles=True)
        
        # Normalize field residuals
        sigma_0 = 30.0
        eq1 = res_sigma / (params.m_sigma**2 * sigma_0)
        eq2 = res_omega / (params.m_omega**2 * sigma_0)
        eq3 = res_rho / (params.m_rho**2 * sigma_0)
        eq4 = res_phi / (params.m_phi**2 * sigma_0)
        eq5 = (hadron.n_B - n_B)
        eq6 = (hadron.n_C - e_thermo.n)  # charge neutrality (includes meson charge)
        eq7 = ((e_thermo.n + nu_thermo.n) / n_B - Y_L)  # lepton fraction
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C, mu_L = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = (error < 1e-6) or (sol.success and error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    result.mu_L = mu_L
    result.mu_e = mu_L - mu_C
    
    # Compute final thermodynamics using unified function
    thermo = compute_sfho_thermo_from_mu_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    # Lepton thermodynamics
    e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(mu_L, T, include_antiparticles=True)
    
    result.n_C = thermo.n_C
    result.n_S_val = thermo.n_S
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_C = thermo.Y_C
    result.Y_S = thermo.Y_S
    
    result.P_hadrons = thermo.P
    result.P_leptons = e_thermo.P + nu_thermo.P
    
    result.P_total = thermo.P + e_thermo.P + nu_thermo.P
    result.e_total = thermo.e + e_thermo.e + nu_thermo.e
    result.s_total = thermo.s + e_thermo.s + nu_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    
    # Baryon details
    result.baryon_densities = {name: state.n for name, state in thermo.states.items()}
    result.m_eff = {name: state.m_eff for name, state in thermo.states.items()}
    
    return result


# =============================================================================
# RESULT TO GUESS CONVERSION
# =============================================================================
def result_to_guess(
    result: SFHoEOSResult, eq_type: str = 'fixed_yc'
) -> np.ndarray:
    """
    Convert result to initial guess array for next point.
    
    Args:
        result: Previous result
        eq_type: 'beta_eq', 'fixed_yc', 'fixed_yc_ys', 'trapped', 
                 'isentropic_beta_eq', 'isentropic_trapped'
    """
    if eq_type in ['beta_eq', 'fixed_yc']:
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C
        ])
    elif eq_type == 'fixed_yc_ys':
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C, result.mu_S
        ])
    elif eq_type in ['trapped', 'trapped_neutrinos']:
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C, result.mu_L
        ])
    elif eq_type == 'isentropic_beta_eq':
        # For isentropic: include T in the guess
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C, result.T
        ])
    elif eq_type == 'isentropic_trapped':
        # For isentropic trapped: include mu_L and T
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C, result.mu_L, result.T
        ])
    else:
        # Default fallback
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C
        ])


# =============================================================================
# SOLVER: ISENTROPIC BETA EQUILIBRIUM
# =============================================================================
def solve_sfho_isentropic_beta_eq(
    n_B: float, S_target: float,
    params: SFHoParams,
    particles: List[Particle],
    include_photons: bool = True,
    include_muons: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> SFHoEOSResult:
    """
    Solve SFHo EOS in beta equilibrium with fixed entropy per baryon S = s/n_B.
    
    Solves 7 equations for 7 unknowns: [σ, ω, ρ, φ, μ_B, μ_C, T]
    
    Args:
        n_B: Baryon density (fm⁻³)
        S_target: Entropy per baryon (dimensionless)
        params: SFHo parameters
        particles: List of baryon species
        include_photons: Include photon contributions
        initial_guess: Initial guess [σ, ω, ρ, φ, μ_B, μ_C, T]
        
    Returns:
        SFHoEOSResult with T as output (stored in result.T)
    """
    if initial_guess is None:
        guess_beta = get_default_guess_beta_eq(n_B, 10.0, params)
        x0 = np.append(guess_beta, 10.0)  # Add T guess
    else:
        x0 = initial_guess
    
    result = SFHoEOSResult(n_B=n_B)
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C, T = x
        mu_S = 0.0
        
        # Compute hadron thermodynamics (includes source terms and optional pseudoscalar mesons)
        hadron = compute_sfho_thermo_from_mu_fields(
            mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Field equation residuals using source terms from hadron
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        mu_e = -mu_C
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        
        # Total entropy (hadron.s now includes mesons if enabled)
        s_total = hadron.s + e_thermo.s
        if include_photons:
            gamma = photon_thermo(T)
            s_total += gamma.s
        
        sigma_0 = 30.0
        eq1 = res_sigma / (params.m_sigma**2 * sigma_0)
        eq2 = res_omega / (params.m_omega**2 * sigma_0)
        eq3 = res_rho / (params.m_rho**2 * sigma_0)
        eq4 = res_phi / (params.m_phi**2 * sigma_0)
        eq5 = (hadron.n_B - n_B)
        eq6 = (hadron.n_C - e_thermo.n)  # charge neutrality (includes meson charge)
        eq7 = (s_total / n_B - S_target)
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]

    sol = root(equations, x0, method='hybr', options={'maxfev': 3000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 3000})
        
    sigma, omega, rho, phi, mu_B, mu_C, T = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    result.error = max(abs(r) for r in residuals)
    result.converged = (result.error < 1e-4) or (sol.success and result.error < 1e-3)
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    result.mu_e = -mu_C
    result.T = T
    
    thermo = compute_sfho_thermo_from_mu_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params
    )
    e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
    
    result.n_C = thermo.n_C
    result.n_e = e_thermo.n
    result.Y_C = thermo.Y_C
    result.Y_S = thermo.Y_S
    
    result.P_hadrons = thermo.P
    result.P_leptons = e_thermo.P
    result.P_total = thermo.P + e_thermo.P
    result.e_total = thermo.e + e_thermo.e
    result.s_total = thermo.s + e_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
        
    result.f_total = result.e_total - T * result.s_total
    result.baryon_densities = {name: state.n for name, state in thermo.states.items()}
    result.m_eff = {name: state.m_eff for name, state in thermo.states.items()}
    
    return result


# =============================================================================
# SOLVER: ISENTROPIC TRAPPED NEUTRINOS
# =============================================================================
def solve_sfho_isentropic_trapped(
    n_B: float, S_target: float, Y_L_target: float,
    params: SFHoParams,
    particles: List[Particle],
    include_photons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> SFHoEOSResult:
    """
    Solve SFHo EOS with trapped neutrinos and fixed entropy per baryon.
    
    Solves 8 equations for 8 unknowns: [σ, ω, ρ, φ, μ_B, μ_C, μ_L, T]
    
    Args:
        n_B: Baryon density (fm⁻³)
        S_target: Entropy per baryon
        Y_L_target: Lepton fraction (n_e + n_nu)/n_B
        params: SFHo parameters
        particles: List of baryon species
        include_photons: Include photon contributions
        initial_guess: Initial guess [σ, ω, ρ, φ, μ_B, μ_C, μ_L, T]
        
    Returns:
        SFHoEOSResult with T as output
    """
    if initial_guess is None:
        guess_trapped = get_default_guess_trapped(n_B, Y_L_target, 10.0, params)
        x0 = np.append(guess_trapped, 10.0)
    else:
        x0 = initial_guess
        
    result = SFHoEOSResult(n_B=n_B, Y_L=Y_L_target)
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C, mu_L, T = x
        mu_S = 0.0
        
        if T < 0.1: T = 0.1
        
        mu_e = mu_L - mu_C
        mu_nu = mu_L
        
        # Compute hadron thermodynamics (includes source terms)
        hadron = compute_sfho_thermo_from_mu_fields(
            mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params
        )
        
        # Field equation residuals using source terms from hadron
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_thermo = neutrino_thermo(mu_nu, T, include_antiparticles=True)
        
        # Total entropy (hadron.s is from unified function)
        s_total = hadron.s + e_thermo.s + nu_thermo.s
        if include_photons:
            gamma = photon_thermo(T)
            s_total += gamma.s
        
        sigma_0 = 30.0
        eq1 = res_sigma / (params.m_sigma**2 * sigma_0)
        eq2 = res_omega / (params.m_omega**2 * sigma_0)
        eq3 = res_rho / (params.m_rho**2 * sigma_0)
        eq4 = res_phi / (params.m_phi**2 * sigma_0)
        eq5 = (hadron.n_B - n_B)
        eq6 = (hadron.n_C - e_thermo.n)  # Charge neutrality
        eq7 = (e_thermo.n + nu_thermo.n)/n_B - Y_L_target  # Lepton fraction
        eq8 = (s_total / n_B - S_target)  # Entropy constraint
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]

    sol = root(equations, x0, method='hybr', options={'maxfev': 3000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 3000})

    sigma, omega, rho, phi, mu_B, mu_C, mu_L, T = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    result.error = max(abs(r) for r in residuals)
    result.converged = (result.error < 1e-4) or (sol.success and result.error < 1e-3)
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    result.mu_e = mu_L - mu_C
    result.mu_nu = mu_L
    result.mu_L = mu_L
    result.T = T
    
    thermo = compute_sfho_thermo_from_mu_fields(
        mu_B, mu_C, mu_S, sigma, omega, rho, phi, T, particles, params
    )
    
    e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(result.mu_nu, T, include_antiparticles=True)
    
    result.n_C = thermo.n_C
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_C = thermo.Y_C
    result.Y_S = thermo.Y_S
    
    result.P_hadrons = thermo.P
    result.P_leptons = e_thermo.P + nu_thermo.P
    result.P_total = thermo.P + result.P_leptons
    result.e_total = thermo.e + e_thermo.e + nu_thermo.e
    result.s_total = thermo.s + e_thermo.s + nu_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
        
    result.f_total = result.e_total - T * result.s_total
    
    result.baryon_densities = {name: state.n for name, state in thermo.states.items()}
    result.m_eff = {name: state.m_eff for name, state in thermo.states.items()}
    
    return result


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("SFHo EOS Solvers Test")
    print("=" * 60)
    
    params = get_sfho_2fam_phi()
    n_B = 0.16
    T = 10.0
    
    print(f"\nTest at n_B = {n_B} fm⁻³, T = {T} MeV")
    
    # Test 1: Beta equilibrium with nucleons only
    print("\n" + "-" * 50)
    print("TEST 1: Beta equilibrium (nucleons only)")
    r = solve_sfho_beta_eq(n_B, T, params, BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  σ = {r.sigma:.2f} MeV, ω = {r.omega:.2f} MeV")
    print(f"  Y_C = {r.Y_C:.4f}, P = {r.P_total:.2f} MeV/fm³")
    print(f"  n_p = {r.baryon_densities.get('p', 0):.4e} fm⁻³")
    print(f"  n_n = {r.baryon_densities.get('n', 0):.4e} fm⁻³")
    
    # Test 2: Beta equilibrium with hyperons
    print("\n" + "-" * 50)
    print("TEST 2: Beta equilibrium (nucleons + hyperons)")
    r = solve_sfho_beta_eq(0.32, T, params, BARYONS_NY)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, Y_S = {r.Y_S:.4f}")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    
    # Test 3: Fixed Y_C (hadrons only)
    print("\n" + "-" * 50)
    print("TEST 3: Fixed Y_C = 0.3 (hadrons only)")
    r = solve_sfho_fixed_yc(n_B, Y_C=0.3, T=T, params=params, particles=BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, P = {r.P_total:.2f} MeV/fm³")
    
    # Test 4: Fixed Y_C = 0.5 (symmetric)
    print("\n" + "-" * 50)
    print("TEST 4: Fixed Y_C = 0.5 (symmetric matter)")
    r = solve_sfho_fixed_yc(n_B, Y_C=0.5, T=T, params=params, particles=BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    print(f"  n_p = {r.baryon_densities.get('p', 0):.4e} fm⁻³")
    print(f"  n_n = {r.baryon_densities.get('n', 0):.4e} fm⁻³")
    
    # Test 5: Fixed Y_C and Y_S
    print("\n" + "-" * 50)
    print("TEST 5: Fixed Y_C = 0.4, Y_S = 0.1 (with hyperons)")
    r = solve_sfho_fixed_yc_ys(0.32, Y_C=0.4, Y_S=0.1, T=T,
                               params=params, particles=BARYONS_NY)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, Y_S = {r.Y_S:.4f}")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    print(f"  μ_S = {r.mu_S:.2f} MeV")
    
    # Test 6: Trapped neutrinos
    print("\n" + "-" * 50)
    print("TEST 6: Trapped neutrinos Y_L = 0.4")
    r = solve_sfho_trapped_neutrinos(n_B, Y_L=0.4, T=50.0, params=params, 
                                      particles=BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, Y_L = {r.Y_L:.4f}")
    print(f"  n_e = {r.n_e:.4e}, n_ν = {r.n_nu:.4e}")
    print(f"  μ_L = {r.mu_L:.2f} MeV")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    
    print("\n" + "=" * 60)
    print("All tests completed!")