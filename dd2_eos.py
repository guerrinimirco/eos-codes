"""
dd2_eos.py
===========
Single-point solvers for DD2 relativistic mean-field EOS with density-dependent couplings.

This module provides solvers for different equilibrium conditions:
- Beta equilibrium (charge neutrality with leptons, strangeness β-eq: μ_S = 0)
- Fixed charge fraction Y_C (hadrons only, or with leptons for charge neutrality)
- Fixed charge and strangeness fractions Y_C, Y_S
- Trapped neutrinos (fixed Y_L = (n_e + n_ν)/n_B)
- Isentropic conditions

Key difference from SFHo:
- Density-dependent couplings: g_M(n_B) = g_M(n_0) × h_M(x)
- Rearrangement term R^0 required for thermodynamic consistency
- No non-linear self-interactions (linear field equations)

References:
- Typel, Röpke, Klähn, Blaschke, Wolter, Phys. Rev. C 81 (2010) 015803
- Fortin, Oertel, Providência, PASA 35 (2018) e044
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from scipy.optimize import root

from general_particles import (
    Particle, Proton, Neutron,
    Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM,
    DeltaPP, DeltaP, Delta0, DeltaM
)
from dd2_parameters import (
    DD2Params, get_dd2_nucleonic, get_dd2y_fortin, get_dd2y_with_deltas
)
from dd2_thermodynamics_hadrons import (
    compute_hadron_thermo, compute_field_residuals,
    compute_meson_contribution, compute_rearrangement_contribution,
    compute_pseudoscalar_meson_thermo, compute_total_pressure
)
from general_thermodynamics_leptons import (
    electron_thermo, photon_thermo, neutrino_thermo, electron_thermo_from_density
)
from general_physics_constants import hc3


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
class DD2EOSResult:
    """Complete result from DD2 EOS calculation at one point."""
    converged: bool = False
    error: float = 0.0
    
    # Inputs
    n_B: float = 0.0
    T: float = 0.0
    Y_C: float = 0.0
    Y_S: float = 0.0
    Y_L: float = 0.0
    
    # Meson fields (MeV)
    sigma: float = 0.0
    omega: float = 0.0
    rho: float = 0.0
    phi: float = 0.0
    R0: float = 0.0  # Rearrangement term (unique to DD2)
    
    # Chemical potentials (MeV)
    mu_B: float = 0.0
    mu_C: float = 0.0
    mu_S: float = 0.0
    mu_e: float = 0.0
    mu_L: float = 0.0
    mu_nu: float = 0.0
    
    # Densities (fm⁻³)
    n_C: float = 0.0
    n_S_val: float = 0.0
    n_e: float = 0.0
    n_nu: float = 0.0
    
    # Thermodynamics (MeV/fm³ for P, e; fm⁻³ for s)
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    f_total: float = 0.0
    
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
def get_default_guess_beta_eq(n_B: float, T: float, params: DD2Params) -> np.ndarray:
    """Generate initial guess for beta equilibrium: [σ, ω, ρ, φ, μ_B, μ_C]."""
    x = n_B / params.n_sat
    
    # DD2 fields scale as sqrt(x) at saturation: σ ~ 58 MeV, ω ~ 74 MeV
    sigma = 58.0 * x**0.5 if x > 0 else 30.0
    omega = 74.0 * x**0.5 if x > 0 else 35.0
    rho = 3.0  # Small for beta equilibrium (near symmetric)
    
    mu_B = 923.0 if x < 0.5 else 923.0 + 100.0 * (x - 0.5)
    mu_C = -20.0  # Initial guess for charge chemical potential
    
    return np.array([sigma, omega, rho, 0.0, mu_B, mu_C])


def get_default_guess_fixed_yc(n_B: float, Y_C: float, T: float, params: DD2Params) -> np.ndarray:
    """Generate initial guess for fixed Y_C: [σ, ω, ρ, φ, μ_B, μ_C]."""
    x = n_B / params.n_sat
    
    sigma = 58.0 * x**0.5 if x > 0 else 30.0
    omega = 74.0 * x**0.5 if x > 0 else 35.0
    
    # rho field scales with asymmetry (positive for neutron-rich)
    asymmetry = 0.5 - Y_C
    rho = 5.0 * asymmetry / 0.5
    
    mu_B = 923.0 if x < 0.5 else 923.0 + 100.0 * (x - 0.5)
    mu_C = -50.0 * asymmetry  # Scale with neutron excess
    
    return np.array([sigma, omega, rho, 0.0, mu_B, mu_C])


def get_default_guess_fixed_yc_ys(n_B: float, Y_C: float, Y_S: float, T: float, params: DD2Params) -> np.ndarray:
    """Generate initial guess for fixed Y_C and Y_S: [σ, ω, ρ, φ, μ_B, μ_C, μ_S]."""
    guess_yc = get_default_guess_fixed_yc(n_B, Y_C, T, params)
    ratio = n_B / params.n_sat
    
    phi = -0.5 * min(ratio, 5.0) * Y_S
    
    if ratio < 0.5:
        mu_S = 180.0
    elif ratio < 1.0:
        mu_S = 130.0 - 30.0 * (ratio - 0.5)
    else:
        mu_S = 100.0 - 50.0 * (ratio - 1.0)
    
    return np.array([guess_yc[0], guess_yc[1], guess_yc[2], phi, guess_yc[4], guess_yc[5], mu_S])


def get_default_guess_trapped(n_B: float, Y_L: float, T: float, params: DD2Params) -> np.ndarray:
    """Generate initial guess for trapped neutrinos: [σ, ω, ρ, φ, μ_B, μ_C, μ_L]."""
    guess_beta = get_default_guess_beta_eq(n_B, T, params)
    mu_L_est = 10.0 + 50.0 * Y_L
    return np.array([guess_beta[0], guess_beta[1], guess_beta[2], guess_beta[3],
                     guess_beta[4], guess_beta[5], mu_L_est])


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_dd2_beta_eq(
    n_B: float, T: float,
    params: DD2Params,
    particles: List[Particle],
    include_photons: bool = True,
    include_muons: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> DD2EOSResult:
    """
    Solve DD2 EOS in beta equilibrium.
    
    Solves 6 equations for 6 unknowns: [σ, ω, ρ, φ, μ_B, μ_C]
    (μ_S = 0 for strangeness β-equilibrium)
    """
    result = DD2EOSResult(n_B=n_B, T=T)
    
    if initial_guess is None:
        x0 = get_default_guess_beta_eq(n_B, T, params)
    else:
        x0 = initial_guess
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C = x
        mu_S = 0.0
        
        # Ensure positive fields
        sigma = abs(sigma)
        omega = abs(omega)
        
        hadron = compute_hadron_thermo(
            T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params
        )
        
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        n_C_mesons = 0.0
        if include_pseudoscalar_mesons:
            meson_result = compute_pseudoscalar_meson_thermo(T, mu_C, mu_S, params)
            n_C_mesons = meson_result.n_Q_mesons
        
        mu_e = -mu_C
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        
        # Normalize field residuals by 1e6 (consistent with dd2_compute_tables.py)
        eq1 = res_sigma / 1e6
        eq2 = res_omega / 1e6
        eq3 = res_rho / 1e6
        eq4 = res_phi / 1e6
        eq5 = (hadron.n_B - n_B) * 1e3
        eq6 = (hadron.n_Q + n_C_mesons - e_thermo.n) * 1e3
        
        return [eq1, eq2, eq3, eq4, eq5, eq6]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    # Match dd2_compute_tables.py: sol.success OR reasonable error
    result.converged = sol.success or (error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    result.mu_e = -mu_C
    
    # Final thermodynamics
    P_tot, e_tot, s_tot, hadron_res, _ = compute_total_pressure(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
    
    result.R0 = hadron_res.R0
    result.n_C = hadron_res.n_Q
    result.n_S_val = hadron_res.n_S
    result.n_e = e_thermo.n
    result.Y_C = hadron_res.n_Q / n_B if n_B > 0 else 0
    result.Y_S = hadron_res.n_S / n_B if n_B > 0 else 0
    
    result.P_hadrons = P_tot
    result.P_leptons = e_thermo.P
    result.P_total = P_tot + e_thermo.P
    result.e_total = e_tot + e_thermo.e
    result.s_total = s_tot + e_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    result.baryon_densities = {name: state.n for name, state in hadron_res.states.items()}
    result.m_eff = {name: state.m_eff for name, state in hadron_res.states.items()}
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C
# =============================================================================
def solve_dd2_fixed_yc(
    n_B: float, Y_C: float, T: float,
    params: DD2Params,
    particles: List[Particle],
    include_electrons: bool = False,
    include_photons: bool = False,
    include_thermal_neutrinos: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> DD2EOSResult:
    """Solve DD2 EOS with fixed charge fraction Y_C."""
    result = DD2EOSResult(n_B=n_B, T=T, Y_C=Y_C)
    
    if initial_guess is None:
        x0 = get_default_guess_fixed_yc(n_B, Y_C, T, params)
    else:
        x0 = initial_guess
    
    n_C_target = Y_C * n_B
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C = x
        mu_S = 0.0
        
        sigma = abs(sigma)
        omega = abs(omega)
        
        hadron = compute_hadron_thermo(
            T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params
        )
        
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        n_C_mesons = 0.0
        if include_pseudoscalar_mesons:
            meson_result = compute_pseudoscalar_meson_thermo(T, mu_C, mu_S, params)
            n_C_mesons = meson_result.n_Q_mesons
        
        n_C_total = hadron.n_Q + n_C_mesons
        
        # Normalize field residuals by 1e6
        eq1 = res_sigma / 1e6
        eq2 = res_omega / 1e6
        eq3 = res_rho / 1e6
        eq4 = res_phi / 1e6
        eq5 = (hadron.n_B - n_B) * 1e3
        eq6 = (n_C_total - n_C_target) * 1e3
        
        return [eq1, eq2, eq3, eq4, eq5, eq6]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = sol.success or (error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    
    P_tot, e_tot, s_tot, hadron_res, _ = compute_total_pressure(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    result.R0 = hadron_res.R0
    result.n_C = hadron_res.n_Q
    result.n_S_val = hadron_res.n_S
    result.Y_C = hadron_res.n_Q / n_B if n_B > 0 else 0
    result.Y_S = hadron_res.n_S / n_B if n_B > 0 else 0
    
    result.P_hadrons = P_tot
    result.P_total = P_tot
    result.e_total = e_tot
    result.s_total = s_tot
    
    if include_electrons:
        e_result = electron_thermo_from_density(hadron_res.n_Q, T)
        result.mu_e = e_result.mu
        result.n_e = e_result.n
        result.P_leptons = e_result.P
        result.P_total += e_result.P
        result.e_total += e_result.e
        result.s_total += e_result.s
    
    if include_thermal_neutrinos and T > 0:
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
    result.baryon_densities = {name: state.n for name, state in hadron_res.states.items()}
    result.m_eff = {name: state.m_eff for name, state in hadron_res.states.items()}
    
    return result


# =============================================================================
# SOLVER: FIXED Y_C AND Y_S
# =============================================================================
def solve_dd2_fixed_yc_ys(
    n_B: float, Y_C: float, Y_S: float, T: float,
    params: DD2Params,
    particles: List[Particle],
    include_electrons: bool = False,
    include_photons: bool = False,
    include_thermal_neutrinos: bool = False,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> DD2EOSResult:
    """Solve DD2 EOS with fixed charge AND strangeness fractions."""
    result = DD2EOSResult(n_B=n_B, T=T, Y_C=Y_C, Y_S=Y_S)
    
    if initial_guess is None:
        x0 = get_default_guess_fixed_yc_ys(n_B, Y_C, Y_S, T, params)
    else:
        x0 = initial_guess
    
    n_C_target = Y_C * n_B
    n_S_target = Y_S * n_B
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C, mu_S = x
        
        sigma = abs(sigma)
        omega = abs(omega)
        
        hadron = compute_hadron_thermo(
            T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params
        )
        
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        n_C_mesons, n_S_mesons = 0.0, 0.0
        if include_pseudoscalar_mesons:
            meson_result = compute_pseudoscalar_meson_thermo(T, mu_C, mu_S, params)
            n_C_mesons = meson_result.n_Q_mesons
            n_S_mesons = meson_result.n_S_mesons
        
        n_C_total = hadron.n_Q + n_C_mesons
        n_S_total = hadron.n_S + n_S_mesons
        
        # Normalize field residuals by 1e6
        eq1 = res_sigma / 1e6
        eq2 = res_omega / 1e6
        eq3 = res_rho / 1e6
        eq4 = res_phi / 1e6
        eq5 = (hadron.n_B - n_B) * 1e3
        eq6 = (n_C_total - n_C_target) * 1e3
        eq7 = (n_S_total - n_S_target) * 1e3
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C, mu_S = sol.x
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = sol.success or (error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    
    P_tot, e_tot, s_tot, hadron_res, _ = compute_total_pressure(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    result.R0 = hadron_res.R0
    result.n_C = hadron_res.n_Q
    result.n_S_val = hadron_res.n_S
    result.Y_C = hadron_res.n_Q / n_B if n_B > 0 else 0
    result.Y_S = hadron_res.n_S / n_B if n_B > 0 else 0
    
    result.P_hadrons = P_tot
    result.P_total = P_tot
    result.e_total = e_tot
    result.s_total = s_tot
    
    if include_electrons:
        e_result = electron_thermo_from_density(hadron_res.n_Q, T)
        result.mu_e = e_result.mu
        result.n_e = e_result.n
        result.P_leptons = e_result.P
        result.P_total += e_result.P
        result.e_total += e_result.e
        result.s_total += e_result.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    result.baryon_densities = {name: state.n for name, state in hadron_res.states.items()}
    result.m_eff = {name: state.m_eff for name, state in hadron_res.states.items()}
    
    return result


# =============================================================================
# SOLVER: TRAPPED NEUTRINOS (FIXED Y_L)
# =============================================================================
def solve_dd2_trapped_neutrinos(
    n_B: float, Y_L: float, T: float,
    params: DD2Params,
    particles: List[Particle],
    include_photons: bool = True,
    include_pseudoscalar_mesons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> DD2EOSResult:
    """Solve DD2 EOS with trapped neutrinos (fixed lepton fraction Y_L)."""
    result = DD2EOSResult(n_B=n_B, T=T, Y_L=Y_L)
    
    if initial_guess is None:
        x0 = get_default_guess_trapped(n_B, Y_L, T, params)
    else:
        x0 = initial_guess
    
    def equations(x):
        sigma, omega, rho, phi, mu_B, mu_C, mu_L = x
        mu_S = 0.0
        
        sigma = abs(sigma)
        omega = abs(omega)
        
        hadron = compute_hadron_thermo(
            T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params
        )
        
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            hadron.src_sigma, hadron.src_omega, hadron.src_rho, hadron.src_phi,
            params
        )
        
        n_C_mesons = 0.0
        if include_pseudoscalar_mesons:
            meson_result = compute_pseudoscalar_meson_thermo(T, mu_C, mu_S, params)
            n_C_mesons = meson_result.n_Q_mesons
        
        n_C_total = hadron.n_Q + n_C_mesons
        
        mu_e = mu_L - mu_C
        e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_thermo = neutrino_thermo(mu_L, T, include_antiparticles=True)
        
        # Normalize field residuals by 1e6
        eq1 = res_sigma / 1e6
        eq2 = res_omega / 1e6
        eq3 = res_rho / 1e6
        eq4 = res_phi / 1e6
        eq5 = (hadron.n_B - n_B) * 1e3
        eq6 = (n_C_total - e_thermo.n) * 1e3  # charge neutrality
        eq7 = ((e_thermo.n + nu_thermo.n) / n_B - Y_L) * 10  # lepton fraction
        
        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7]
    
    sol = root(equations, x0, method='hybr', options={'maxfev': 2000})
    if not sol.success:
        sol = root(equations, x0, method='lm', options={'maxiter': 2000})
    
    sigma, omega, rho, phi, mu_B, mu_C, mu_L = sol.x
    mu_S = 0.0
    
    residuals = equations(sol.x)
    error = max(abs(r) for r in residuals)
    result.converged = sol.success or (error < 1e-4)
    result.error = error
    
    result.sigma, result.omega, result.rho, result.phi = sigma, omega, rho, phi
    result.mu_B, result.mu_C, result.mu_S = mu_B, mu_C, mu_S
    result.mu_L = mu_L
    result.mu_e = mu_L - mu_C
    
    P_tot, e_tot, s_tot, hadron_res, _ = compute_total_pressure(
        T, mu_B, mu_C, mu_S, sigma, omega, rho, phi, n_B, particles, params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    
    e_thermo = electron_thermo(result.mu_e, T, include_antiparticles=True)
    nu_thermo = neutrino_thermo(mu_L, T, include_antiparticles=True)
    
    result.R0 = hadron_res.R0
    result.n_C = hadron_res.n_Q
    result.n_S_val = hadron_res.n_S
    result.n_e = e_thermo.n
    result.n_nu = nu_thermo.n
    result.Y_C = hadron_res.n_Q / n_B if n_B > 0 else 0
    result.Y_S = hadron_res.n_S / n_B if n_B > 0 else 0
    
    result.P_hadrons = P_tot
    result.P_leptons = e_thermo.P + nu_thermo.P
    result.P_total = P_tot + e_thermo.P + nu_thermo.P
    result.e_total = e_tot + e_thermo.e + nu_thermo.e
    result.s_total = s_tot + e_thermo.s + nu_thermo.s
    
    if include_photons:
        gamma = photon_thermo(T)
        result.P_photons = gamma.P
        result.P_total += gamma.P
        result.e_total += gamma.e
        result.s_total += gamma.s
    
    result.f_total = result.e_total - T * result.s_total
    result.baryon_densities = {name: state.n for name, state in hadron_res.states.items()}
    result.m_eff = {name: state.m_eff for name, state in hadron_res.states.items()}
    
    return result


# =============================================================================
# RESULT TO GUESS CONVERSION
# =============================================================================
def result_to_guess(result: DD2EOSResult, eq_type: str = 'fixed_yc') -> np.ndarray:
    """Convert result to initial guess array for next point."""
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
    else:
        return np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_C
        ])


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("DD2 EOS Solvers Test")
    print("=" * 60)
    
    params = get_dd2y_fortin()
    n_B = 0.16
    T = 10.0
    
    print(f"\nTest at n_B = {n_B} fm⁻³, T = {T} MeV")
    
    # Test 1: Beta equilibrium with nucleons only
    print("\n" + "-" * 50)
    print("TEST 1: Beta equilibrium (nucleons only)")
    r = solve_dd2_beta_eq(n_B, T, params, BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  σ = {r.sigma:.2f} MeV, ω = {r.omega:.2f} MeV")
    print(f"  R⁰ = {r.R0:.4f} MeV (rearrangement term)")
    print(f"  Y_C = {r.Y_C:.4f}, P = {r.P_total:.2f} MeV/fm³")
    
    # Test 2: Beta equilibrium with hyperons
    print("\n" + "-" * 50)
    print("TEST 2: Beta equilibrium (nucleons + hyperons)")
    r = solve_dd2_beta_eq(0.32, T, params, BARYONS_NY)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, Y_S = {r.Y_S:.4f}")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    
    # Test 3: Fixed Y_C
    print("\n" + "-" * 50)
    print("TEST 3: Fixed Y_C = 0.3")
    r = solve_dd2_fixed_yc(n_B, Y_C=0.3, T=T, params=params, particles=BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, P = {r.P_total:.2f} MeV/fm³")
    
    # Test 4: Fixed Y_C = 0.5
    print("\n" + "-" * 50)
    print("TEST 4: Fixed Y_C = 0.5 (symmetric matter)")
    r = solve_dd2_fixed_yc(n_B, Y_C=0.5, T=T, params=params, particles=BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    
    # Test 5: Trapped neutrinos
    print("\n" + "-" * 50)
    print("TEST 5: Trapped neutrinos Y_L = 0.4")
    r = solve_dd2_trapped_neutrinos(n_B, Y_L=0.4, T=50.0, params=params, particles=BARYONS_N)
    print(f"  converged = {r.converged}, error = {r.error:.2e}")
    print(f"  Y_C = {r.Y_C:.4f}, Y_L = {r.Y_L:.4f}")
    print(f"  n_e = {r.n_e:.4e}, n_ν = {r.n_nu:.4e}")
    print(f"  μ_L = {r.mu_L:.2f} MeV")
    print(f"  P = {r.P_total:.2f} MeV/fm³")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
