"""
sfhoalphabag_mixed_phase_eos.py
===============================
SFHo (hadronic) + AlphaBag (quark) mixed phase EOS solver.

This module provides:
    - Mixed phase solvers for all η values (Gibbs, Maxwell, intermediate)
    - Phase boundary finding (onset at χ=0, offset at χ=1)
    - Unified EOS table generation

Equilibrium modes:
    - BETA: Beta equilibrium with electrons (charge neutrality)
    - FIXED_YC: Fixed charge fraction Y_C (with electrons for neutrality)

Solver unknowns:
    - η=0 (global neutrality): 11 [σ, ω, ρ, φ, μu, μd, μs, μeG, χ, μB_H, μC_H]
    - η=1 (local neutrality): 12 [σ, ω, ρ, φ, μu, μd, μs, μeL_H, μeL_Q, χ, μB_H, μC_H]
    - 0<η<1: 13 [σ, ω, ρ, φ, μu, μd, μs, μeL_H, μeL_Q, μeG, χ, μB_H, μC_H]
"""

import numpy as np
from scipy.optimize import root
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# SFHo (hadronic) modules
from sfho_parameters import SFHoParams, get_sfho_nucleonic
from sfho_thermodynamics_hadrons import (
    compute_sfho_thermo_from_mu_fields,
    compute_field_residuals,
    compute_meson_contribution
)
from sfho_eos import (
    solve_sfho_beta_eq as solve_pure_H_beta,
    solve_sfho_fixed_yc as solve_pure_H_fixed_yc,
    SFHoEOSResult
)

# AlphaBag (quark) modules
from alphabag_parameters import AlphaBagParams, get_alphabag_default
from alphabag_thermodynamics_quarks import (
    compute_alphabag_thermo_from_mu,
    AlphaBagThermo,
    gluon_thermo
)
from alphabag_eos import (
    solve_alphabag_beta_eq as solve_pure_Q_beta,
    solve_alphabag_fixed_yc as solve_pure_Q_fixed_yc,
    AlphaBagEOSResult
)

# General physics modules
from general_thermodynamics_leptons import electron_thermo, photon_thermo, neutrino_thermo
from general_physics_constants import hc, hc3
from general_particles import Proton, Neutron


# =============================================================================
# EQUILIBRIUM MODE ENUM
# =============================================================================
class EquilibriumMode(Enum):
    """Equilibrium condition for mixed phase calculation."""
    BETA = "beta"
    FIXED_YC = "fixed_yc"


# =============================================================================
# DEFAULT PARTICLES FOR SFHO
# =============================================================================
NUCLEONS = [Proton, Neutron]


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class MixedPhaseResult:
    """Result from SFHo+AlphaBag mixed phase calculation."""
    converged: bool
    error: float
    
    # Inputs
    n_B: float
    T: float
    eta: float
    
    # Solution
    chi: float  # Quark volume fraction
    
    # Hadronic phase - meson fields
    sigma_H: float = 0.0
    omega_H: float = 0.0
    rho_H: float = 0.0
    phi_H: float = 0.0
    
    # Hadronic phase - conserved charge chemical potentials
    mu_B_H: float = 0.0
    mu_C_H: float = 0.0
    mu_S_H: float = 0.0
    
    # Hadronic phase - densities and thermodynamics
    n_B_H: float = 0.0
    n_C_H: float = 0.0
    n_S_H: float = 0.0
    P_H: float = 0.0
    e_H: float = 0.0
    s_H: float = 0.0
    f_H: float = 0.0
    
    # Quark phase - particle chemical potentials
    mu_u_Q: float = 0.0
    mu_d_Q: float = 0.0
    mu_s_Q: float = 0.0
    
    # Quark phase - densities
    n_u_Q: float = 0.0
    n_d_Q: float = 0.0
    n_s_Q: float = 0.0
    
    # Quark phase - conserved charge quantities
    mu_B_Q: float = 0.0
    mu_C_Q: float = 0.0
    mu_S_Q: float = 0.0
    n_B_Q: float = 0.0
    n_C_Q: float = 0.0
    n_S_Q: float = 0.0
    P_Q: float = 0.0
    e_Q: float = 0.0
    s_Q: float = 0.0
    f_Q: float = 0.0
    
    # Electrons
    mu_eL_H: float = 0.0
    mu_eL_Q: float = 0.0
    mu_eG: float = 0.0
    n_eL_H: float = 0.0
    n_eL_Q: float = 0.0
    n_eG: float = 0.0
    P_eL_H: float = 0.0
    P_eL_Q: float = 0.0
    P_eG: float = 0.0
    e_eL_H: float = 0.0
    e_eL_Q: float = 0.0
    e_eG: float = 0.0
    s_eL_H: float = 0.0
    s_eL_Q: float = 0.0
    s_eG: float = 0.0
    
    # Photons
    P_gamma: float = 0.0
    e_gamma: float = 0.0
    s_gamma: float = 0.0
    
    # Gluons (only in Q phase)
    P_gluon: float = 0.0
    e_gluon: float = 0.0
    s_gluon: float = 0.0
    
    # Thermal neutrinos (mu=0)
    P_nu_th: float = 0.0
    e_nu_th: float = 0.0
    s_nu_th: float = 0.0
    
    # Equilibrium mode
    eq_mode: str = "beta"
    Y_C_input: float = 0.0
    
    # ==========================================================================
    # Total quantities (DERIVED via @property)
    # ==========================================================================
    
    @property
    def P_e_tot(self) -> float:
        """Total electron pressure (volume-weighted)."""
        return (self.eta * (1 - self.chi) * self.P_eL_H + 
                self.eta * self.chi * self.P_eL_Q + 
                (1 - self.eta) * self.P_eG)
    
    @property
    def e_e_tot(self) -> float:
        """Total electron energy density (volume-weighted)."""
        return (self.eta * (1 - self.chi) * self.e_eL_H + 
                self.eta * self.chi * self.e_eL_Q + 
                (1 - self.eta) * self.e_eG)
    
    @property
    def s_e_tot(self) -> float:
        """Total electron entropy density (volume-weighted)."""
        return (self.eta * (1 - self.chi) * self.s_eL_H + 
                self.eta * self.chi * self.s_eL_Q + 
                (1 - self.eta) * self.s_eG)
    
    @property
    def f_e_tot(self) -> float:
        """Total electron free energy density."""
        return self.e_e_tot - self.T * self.s_e_tot
    
    @property
    def n_e_tot(self) -> float:
        """Total electron density (volume-weighted)."""
        return (self.eta * (1 - self.chi) * self.n_eL_H + 
                self.eta * self.chi * self.n_eL_Q + 
                (1 - self.eta) * self.n_eG)
    
    # ==========================================================================
    # Total thermodynamic quantities (volume-weighted + all components)
    # ==========================================================================
    
    @property
    def P_total(self) -> float:
        """Total pressure: (1-χ)*P_H + χ*(P_Q + P_gluon) + P_e_tot + P_γ + P_ν_th"""
        return ((1 - self.chi) * self.P_H + self.chi * (self.P_Q + self.P_gluon) + 
                self.P_e_tot + self.P_gamma + self.P_nu_th)
    
    @property
    def e_total(self) -> float:
        """Total energy density (volume-weighted + leptons + photons + gluons + neutrinos)."""
        return ((1 - self.chi) * self.e_H + self.chi * (self.e_Q + self.e_gluon) + 
                self.e_e_tot + self.e_gamma + self.e_nu_th)
    
    @property
    def s_total(self) -> float:
        """Total entropy density (volume-weighted + leptons + photons + gluons + neutrinos)."""
        return ((1 - self.chi) * self.s_H + self.chi * (self.s_Q + self.s_gluon) + 
                self.s_e_tot + self.s_gamma + self.s_nu_th)
    
    @property
    def f_total(self) -> float:
        """Total free energy density: f = e - Ts"""
        return self.e_total - self.T * self.s_total
    
    @property
    def n_B_tot(self) -> float:
        """Total baryon density."""
        return (1 - self.chi) * self.n_B_H + self.chi * self.n_B_Q
    
    @property
    def n_C_tot(self) -> float:
        """Total charge density."""
        return (1 - self.chi) * self.n_C_H + self.chi * self.n_C_Q
    
    @property
    def Y_C_tot(self) -> float:
        """Total charge fraction."""
        return self.n_C_tot / self.n_B if self.n_B > 0 else 0.0
    
    @property
    def Y_e_tot(self) -> float:
        """Total electron fraction."""
        return self.n_e_tot / self.n_B if self.n_B > 0 else 0.0


# =============================================================================
# PHASE BOUNDARY RESULT
# =============================================================================
@dataclass
class PhaseBoundaryResult:
    """Result from phase boundary finding."""
    T: float
    eta: float
    converged: bool = False
    converged_onset: bool = False
    converged_offset: bool = False
    
    # Onset (χ=0)
    n_B_onset: float = 0.0
    sigma_H_onset: float = 0.0
    omega_H_onset: float = 0.0
    rho_H_onset: float = 0.0
    phi_H_onset: float = 0.0
    mu_B_H_onset: float = 0.0
    mu_C_H_onset: float = 0.0
    mu_u_Q_onset: float = 0.0
    mu_d_Q_onset: float = 0.0
    mu_s_Q_onset: float = 0.0
    mu_eG_onset: float = 0.0
    mu_eL_H_onset: float = 0.0
    mu_eL_Q_onset: float = 0.0
    
    # Offset (χ=1)
    n_B_offset: float = 0.0
    sigma_H_offset: float = 0.0
    omega_H_offset: float = 0.0
    rho_H_offset: float = 0.0
    phi_H_offset: float = 0.0
    mu_B_H_offset: float = 0.0
    mu_C_H_offset: float = 0.0
    mu_u_Q_offset: float = 0.0
    mu_d_Q_offset: float = 0.0
    mu_s_Q_offset: float = 0.0
    mu_eG_offset: float = 0.0
    mu_eL_H_offset: float = 0.0
    mu_eL_Q_offset: float = 0.0


# =============================================================================
# η=0 GCN beta equilibrium [Gibbs construction]
# =============================================================================
def solve_eta0_beta(n_B: float, T: float,
                    sfho_params: SFHoParams = None,
                    alphabag_params: AlphaBagParams = None,
                    particles: List = None,
                    initial_guess: np.ndarray = None,
                    include_pseudoscalar_mesons: bool = True,
                    include_gluons: bool = True,
                    include_photons: bool = True,
                    include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 (Gibbs construction).
    
    11 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, χ]
    
    Equations:
        1-4. Field equations for σ, ω, ρ, φ
        5. μ_C_H + μ_eG = 0                            (β-eq in H sector)
        6. μ_C_Q + μ_eG = 0                            (β-eq in Q sector)
        7. μ_S_Q = 0                                   (strangeness eq in Q sector)
        8. (1-χ)*n_B_H + χ*n_B_Q = n_B                 (baryon conservation)
        9. (1-χ)*n_C_H + χ*n_C_Q - n_e = 0             (global charge neutrality)
        10. μ_B_H = μ_B_Q                              (baryon chemical equilibrium)
        11. P_H = P_Q                                  (mechanical equilibrium)
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,  # σ, ω, ρ, φ
            950.0, -50.0,           # μ_B_H, μ_C_H
            300.0, 350.0, 350.0,    # μu, μd, μs
            50.0,                   # μeG
            0.5                     # χ
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, chi = x
        mu_S_H = 0.0  # Strangeness β-eq in hadronic phase
        
        # Hadronic thermodynamics (now includes source terms)
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params, include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Get field residuals using source terms from had
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        # Quark thermodynamics
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        
        # Gluon thermodynamics (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        # Electron thermodynamics
        ele = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        res = np.zeros(11)
        # Field equations (scaled)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        # β-equilibrium
        res[4] = mu_C_H + mu_eG
        res[5] = qua.mu_C + mu_eG
        # Strangeness equilibrium
        res[6] = qua.mu_S
        # Baryon conservation
        res[7] = (1 - chi) * had.n_B + chi * qua.n_B - n_B
        # Global charge neutrality
        res[8] = (1 - chi) * had.n_C + chi * qua.n_C - ele.n
        # Chemical equilibrium
        res[9] = (mu_B_H - qua.mu_B) / 1000.0
        # Mechanical equilibrium (gluons in Q phase if included)
        res[10] = (had.P - qua.P - glu_P) / 100.0
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-8)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, chi = sol.x
    mu_S_H = 0.0
    
    # Final thermodynamics
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params, include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    ele = electron_thermo(mu_eG, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eG=mu_eG, n_eG=ele.n,
        P_eG=ele.P, e_eG=ele.e, s_eG=ele.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


# =============================================================================
# η=1 BETA EQUILIBRIUM (Maxwell construction)
# =============================================================================
def solve_eta1_beta(n_B: float, T: float,
                    sfho_params: SFHoParams = None,
                    alphabag_params: AlphaBagParams = None,
                    particles: List = None,
                    initial_guess: np.ndarray = None,
                    include_pseudoscalar_mesons: bool = True,
                    include_gluons: bool = True,
                    include_photons: bool = True,
                    include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 (Maxwell construction).
    
    12 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μ_eL_H, μ_eL_Q, χ]
    
    Equations:
        1-4. Field equations for σ, ω, ρ, φ
        5.   μ_C_H + μ_eL_H = 0                        (β-eq in H sector with local e)
        6.   μ_C_Q + μ_eL_Q = 0                        (β-eq in Q sector with local e)
        7.   μ_S_Q = 0                                 (strangeness eq in Q sector)
        8.   (1-χ)*n_B_H + χ*n_B_Q = n_B               (baryon conservation)
        9.   n_C_H = n_eL_H                            (local neutrality in H)
        10.  n_C_Q = n_eL_Q                            (local neutrality in Q)
        11.  μ_B_H = μ_B_Q                             (baryon chemical equilibrium)
        12.  P_H + P_eL_H = P_Q + P_eL_Q               (mechanical equilibrium with electrons)
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    if initial_guess is None:
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,   # σ, ω, ρ, φ
            950.0, -50.0,            # μ_B_H, μ_C_H
            300.0, 350.0, 350.0,     # μu, μd, μs
            50.0, 50.0,              # μ_eL_H, μ_eL_Q
            0.5                      # χ
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, chi = x
        mu_S_H = 0.0
        
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params, include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        # Gluon pressure (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        res = np.zeros(12)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        res[4] = mu_C_H + mu_eL_H               # β-eq in H
        res[5] = qua.mu_C + mu_eL_Q             # β-eq in Q
        res[6] = qua.mu_S                       # strangeness eq
        res[7] = (1 - chi) * had.n_B + chi * qua.n_B - n_B  # baryon conservation
        res[8] = had.n_C - eleH.n               # local neutrality H
        res[9] = qua.n_C - eleQ.n               # local neutrality Q
        res[10] = (mu_B_H - qua.mu_B) / 1000.0  # chemical equilibrium
        res[11] = (had.P + eleH.P - qua.P - glu_P - eleQ.P) / 100.0  # mechanical equilibrium (gluons in Q)
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 2000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, chi = sol.x
    mu_S_H = 0.0
    
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    max_err = np.max(np.abs(sol.fun))
    is_converged = sol.success or max_err < 1e-6
    
    return MixedPhaseResult(
        converged=is_converged,
        error=max_err,
        n_B=n_B, T=T, eta=1.0, chi=chi,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q,
        P_eL_H=eleH.P, e_eL_H=eleH.e, s_eL_H=eleH.s,
        P_eL_Q=eleQ.P, e_eL_Q=eleQ.e, s_eL_Q=eleQ.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


# =============================================================================
# η=0 FIXED Y_C
# =============================================================================
def solve_eta0_fixed_yc(n_B: float, Y_C: float, T: float,
                        sfho_params: SFHoParams = None,
                        alphabag_params: AlphaBagParams = None,
                        particles: List = None,
                        initial_guess: np.ndarray = None,
                        include_pseudoscalar_mesons: bool = True,
                        include_gluons: bool = True,
                        include_photons: bool = True,
                        include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 with fixed charge fraction Y_C.
    
    11 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, χ]
    
    Replaces β-eq equations with:
        5. (1-χ)*n_C_H + χ*n_C_Q = n_B * Y_C    (charge global conservation)
        6. (1-χ)*n_C_H + χ*n_C_Q - n_e = 0      (global charge neutrality with electrons)
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    if initial_guess is None:
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,
            950.0, -50.0,
            300.0, 350.0, 350.0,
            50.0,
            0.5
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, chi = x
        mu_S_H = 0.0
        
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params, include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        ele = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        res = np.zeros(11)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        # Charge conservation
        res[4] = (1 - chi) * had.n_C + chi * qua.n_C - n_B * Y_C
        # Charge neutrality with electrons
        res[5] = (1 - chi) * had.n_C + chi * qua.n_C - ele.n
        # Strangeness equilibrium
        res[6] = qua.mu_S
        # Baryon conservation
        res[7] = (1 - chi) * had.n_B + chi * qua.n_B - n_B
        # Chemical equilibrium
        res[8] = (mu_B_H - qua.mu_B) / 1000.0
        # Mechanical equilibrium
        res[9] = (had.P - qua.P) / 100.0
        # Charge chemical potential equilibrium
        res[10] = (mu_C_H - qua.mu_C) / 100.0
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-8)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, chi = sol.x
    mu_S_H = 0.0
    
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    ele = electron_thermo(mu_eG, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eG=mu_eG, n_eG=ele.n,
        P_eG=ele.P, e_eG=ele.e, s_eG=ele.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


# =============================================================================
# η=1 FIXED Y_C (Maxwell construction)
# =============================================================================
def solve_eta1_fixed_yc(n_B: float, Y_C: float, T: float,
                        sfho_params: SFHoParams = None,
                        alphabag_params: AlphaBagParams = None,
                        particles: List = None,
                        initial_guess: np.ndarray = None,
                        include_pseudoscalar_mesons: bool = True,
                        include_gluons: bool = True,
                        include_photons: bool = True,
                        include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 (Maxwell) with fixed charge fraction Y_C.
    
    12 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μ_eL_H, μ_eL_Q, χ]
    
    Equations:
        1-4.  Field equations for σ, ω, ρ, φ
        5.    (1-χ)*n_C_H + χ*n_C_Q = n_B * Y_C     (charge global conservation)
        6.    n_C_H = n_eL_H                        (local neutrality H)
        7.    n_C_Q = n_eL_Q                        (local neutrality Q)
        8.    μ_S_Q = 0                             (strangeness eq in Q)
        9.    (1-χ)*n_B_H + χ*n_B_Q = n_B           (baryon conservation)
        10.   μ_B_H = μ_B_Q                         (chemical equilibrium)
        11.   P_H + P_eL_H = P_Q + P_eL_Q           (mechanical equilibrium)
        12.   μ_C_H = μ_C_Q                         (charge chemical equilibrium)
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    if initial_guess is None:
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,   # σ, ω, ρ, φ
            950.0, -50.0,            # μ_B_H, μ_C_H
            300.0, 350.0, 350.0,     # μu, μd, μs
            50.0, 50.0,              # μ_eL_H, μ_eL_Q
            0.5                      # χ
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, chi = x
        mu_S_H = 0.0
        
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        # Gluon pressure (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        res = np.zeros(12)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        res[4] = (1 - chi) * had.n_C + chi * qua.n_C - n_B * Y_C  # charge conservation
        res[5] = had.n_C - eleH.n               # local neutrality H
        res[6] = qua.n_C - eleQ.n               # local neutrality Q
        res[7] = qua.mu_S                       # strangeness eq
        res[8] = (1 - chi) * had.n_B + chi * qua.n_B - n_B  # baryon conservation
        res[9] = (mu_B_H - qua.mu_B) / 1000.0   # chemical equilibrium
        res[10] = (had.P + eleH.P - qua.P - glu_P - eleQ.P) / 100.0  # mechanical equilibrium (gluons only in Q)
        res[11] = (mu_C_H - qua.mu_C) / 100.0   # charge chemical equilibrium
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 2000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, chi = sol.x
    mu_S_H = 0.0
    
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    max_err = np.max(np.abs(sol.fun))
    is_converged = sol.success or max_err < 1e-6
    
    return MixedPhaseResult(
        converged=is_converged,
        error=max_err,
        n_B=n_B, T=T, eta=1.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q,
        P_eL_H=eleH.P, e_eL_H=eleH.e, s_eL_H=eleH.s,
        P_eL_Q=eleQ.P, e_eL_Q=eleQ.e, s_eL_Q=eleQ.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


# =============================================================================
# DISPATCHER
# =============================================================================
def solve_mixed_phase(n_B: float, T: float, eta: float,
                      sfho_params: SFHoParams = None,
                      alphabag_params: AlphaBagParams = None,
                      particles: List = None,
                      eq_mode: str = "beta",
                      Y_C: float = None,
                      initial_guess: np.ndarray = None,
                      include_pseudoscalar_mesons: bool = True,
                      include_gluons: bool = True,
                      include_photons: bool = True,
                      include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """Dispatcher to select appropriate solver based on η and equilibrium mode."""
    
    physics_opts = dict(
        include_pseudoscalar_mesons=include_pseudoscalar_mesons,
        include_gluons=include_gluons,
        include_photons=include_photons,
        include_thermal_neutrinos=include_thermal_neutrinos
    )
    
    if eq_mode == "beta":
        if abs(eta) < 1e-10:
            return solve_eta0_beta(n_B, T, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        elif abs(eta - 1.0) < 1e-10:
            return solve_eta1_beta(n_B, T, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        else:
            raise NotImplementedError(f"Mixed phase solver for eta={eta} not yet implemented")
    elif eq_mode == "fixed_yc":
        if Y_C is None:
            raise ValueError("Y_C must be provided for fixed_yc mode")
        if abs(eta) < 1e-10:
            return solve_eta0_fixed_yc(n_B, Y_C, T, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        elif abs(eta - 1.0) < 1e-10:
            return solve_eta1_fixed_yc(n_B, Y_C, T, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        else:
            raise NotImplementedError(f"Mixed phase solver for eta={eta} not yet implemented")
    else:
        raise ValueError(f"Unknown equilibrium mode: {eq_mode}")


# =============================================================================
# FIXED-CHI SOLVERS (for finding phase boundaries)
# =============================================================================

def solve_eta0_fixed_chi_beta(T: float, chi: float,
                               sfho_params: SFHoParams = None,
                               alphabag_params: AlphaBagParams = None,
                               particles: List = None,
                               initial_guess: np.ndarray = None,
                               include_pseudoscalar_mesons: bool = True,
                               include_gluons: bool = True,
                               include_photons: bool = True,
                               include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 with χ FIXED.
    
    Used to find phase boundaries:
    - χ=0: onset of mixed phase (n_B_onset)
    - χ=1: end of mixed phase (n_B_offset)
    
    12 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, n_B]
    Note: n_B is now the unknown, χ is the input!
    
    Equations:
        1-4. Field equations for σ, ω, ρ, φ (meson self-consistency)
        5.   μ_C_H + μ_eG = 0                            (β-eq in H sector)
        6.   μ_C_Q + μ_eG = 0                            (β-eq in Q sector)
        7.   μ_S_Q = 0                                   (strangeness eq in Q sector)
        8.   (1-χ)*n_B_H + χ*n_B_Q = n_B                 (baryon conservation)
        9.   (1-χ)*n_C_H + χ*n_C_Q - n_e = 0             (global charge neutrality)
        10.  μ_B_H = μ_B_Q                               (baryon chemical equilibrium)
        11.  P_H = P_Q                                   (mechanical equilibrium)
        12.  ... (automatic, system determines n_B)
    
    Since we have 12 unknowns but listed 11 physics equations, the 12th
    constraint comes from the consistency of the problem - we're looking
    for which n_B satisfies all constraints simultaneously.
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    # Default initial guess for n_B depends on chi
    if initial_guess is None:
        n_B_est = 0.4 if chi < 0.5 else 1.0
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,  # σ, ω, ρ, φ
            950.0, -50.0,           # μ_B_H, μ_C_H
            300.0, 350.0, 350.0,    # μu, μd, μs
            50.0,                   # μeG
            n_B_est                 # n_B (the unknown)
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, n_B = x
        mu_S_H = 0.0  # Strangeness β-eq in hadronic phase
        
        # Hadronic thermodynamics (includes source terms)
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Get field residuals using source terms from had
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        # Quark thermodynamics
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        
        # Gluon thermodynamics (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        # Electron thermodynamics
        ele = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        res = np.zeros(11)
        # Field equations (scaled)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        # β-equilibrium
        res[4] = mu_C_H + mu_eG
        res[5] = qua.mu_C + mu_eG
        # Strangeness equilibrium
        res[6] = qua.mu_S
        # Baryon conservation (with χ fixed!)
        res[7] = (1 - chi) * had.n_B + chi * qua.n_B - n_B
        # Global charge neutrality
        res[8] = (1 - chi) * had.n_C + chi * qua.n_C - ele.n
        # Chemical equilibrium
        res[9] = (mu_B_H - qua.mu_B) / 1000.0
        # Mechanical equilibrium (gluons in Q phase if included)
        res[10] = (had.P - qua.P - glu_P) / 100.0
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 5000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, n_B = sol.x
    mu_S_H = 0.0
    
    # Final thermodynamics
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    ele = electron_thermo(mu_eG, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eG=mu_eG, n_eG=ele.n,
        P_eG=ele.P, e_eG=ele.e, s_eG=ele.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


def solve_eta0_fixed_chi_yc(T: float, chi: float, Y_C: float,
                             sfho_params: SFHoParams = None,
                             alphabag_params: AlphaBagParams = None,
                             particles: List = None,
                             initial_guess: np.ndarray = None,
                             include_pseudoscalar_mesons: bool = True,
                             include_gluons: bool = True,
                             include_photons: bool = True,
                             include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 with χ FIXED and fixed charge fraction Y_C.
    
    11 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, n_B]
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    if initial_guess is None:
        n_B_est = 0.4 if chi < 0.5 else 1.0
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,
            950.0, -50.0,
            300.0, 350.0, 350.0,
            50.0,
            n_B_est
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, n_B = x
        mu_S_H = 0.0
        
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        
        # Gluon thermodynamics (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        ele = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        res = np.zeros(11)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        # Charge conservation (fixed Y_C)
        res[4] = (1 - chi) * had.n_C + chi * qua.n_C - n_B * Y_C
        # Charge neutrality with electrons
        res[5] = (1 - chi) * had.n_C + chi * qua.n_C - ele.n
        # Strangeness equilibrium
        res[6] = qua.mu_S
        # Baryon conservation
        res[7] = (1 - chi) * had.n_B + chi * qua.n_B - n_B
        # Chemical equilibrium
        res[8] = (mu_B_H - qua.mu_B) / 1000.0
        # Mechanical equilibrium (gluons in Q phase if included)
        res[9] = (had.P - qua.P - glu_P) / 100.0
        # Charge chemical potential equilibrium
        res[10] = (mu_C_H - qua.mu_C) / 100.0
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 5000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eG, n_B = sol.x
    mu_S_H = 0.0
    
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    ele = electron_thermo(mu_eG, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eG=mu_eG, n_eG=ele.n,
        P_eG=ele.P, e_eG=ele.e, s_eG=ele.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


def solve_eta1_fixed_chi_beta(T: float, chi: float,
                               sfho_params: SFHoParams = None,
                               alphabag_params: AlphaBagParams = None,
                               particles: List = None,
                               initial_guess: np.ndarray = None,
                               include_pseudoscalar_mesons: bool = True,
                               include_gluons: bool = True,
                               include_photons: bool = True,
                               include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 (Maxwell) with χ FIXED.
    
    For η=1, each phase has LOCAL charge neutrality with separate electrons.
    
    14 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeL_H, μeL_Q, n_B]
    
    Equations:
        1-4.  Field equations for σ, ω, ρ, φ
        5.    n_C_H = n_eL_H                               (local neutrality H)
        6.    n_C_Q = n_eL_Q                               (local neutrality Q)
        7.    μ_C_H + μ_eL_H = 0                           (beta-eq in H)
        8.    μ_C_Q + μ_eL_Q = 0                           (beta-eq in Q)
        9.    μ_S_Q = 0                                    (strangeness eq)
        10.   (1-χ)*n_B_H + χ*n_B_Q = n_B                  (baryon conservation)
        11.   μ_B_H = μ_B_Q                                (baryon chemical eq)
        12.   P_H + P_eL_H = P_Q + P_eL_Q                  (mechanical eq)
    """
    if sfho_params is None:
        sfho_params = get_sfho_nucleonic()
    if alphabag_params is None:
        alphabag_params = get_alphabag_default()
    if particles is None:
        particles = NUCLEONS
    
    if initial_guess is None:
        n_B_est = 0.5 if chi < 0.5 else 1.2
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,  # σ, ω, ρ, φ
            950.0, -50.0,           # μ_B_H, μ_C_H
            300.0, 350.0, 350.0,    # μu, μd, μs
            50.0, 50.0,             # μeL_H, μeL_Q
            n_B_est                 # n_B
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_B = x
        mu_S_H = 0.0
        
        # Hadronic thermodynamics (includes source terms)
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        
        # Get field residuals using source terms from had
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        # Quark thermodynamics
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        
        # Gluon thermodynamics (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        # Local electron thermodynamics
        eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        res = np.zeros(12)
        # Field equations (scaled)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        # Local neutrality
        res[4] = had.n_C - eleH.n  # n_C_H = n_eL_H
        res[5] = qua.n_C - eleQ.n  # n_C_Q = n_eL_Q
        # β-equilibrium
        res[6] = mu_C_H + mu_eL_H
        res[7] = qua.mu_C + mu_eL_Q
        # Strangeness equilibrium
        res[8] = qua.mu_S
        # Baryon conservation
        res[9] = (1 - chi) * had.n_B + chi * qua.n_B - n_B
        # Chemical equilibrium
        res[10] = (mu_B_H - qua.mu_B) / 1000.0
        # Mechanical equilibrium (with local electrons, gluons if included)
        res[11] = (had.P + eleH.P - qua.P - glu_P - eleQ.P) / 100.0
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 2000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_B = sol.x
    mu_S_H = 0.0
    
    # Final thermodynamics
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    # Consider converged if residual is small enough, even if solver didn't officially succeed
    max_err = np.max(np.abs(sol.fun))
    is_converged = sol.success or max_err < 1e-6
    
    return MixedPhaseResult(
        converged=is_converged,
        error=max_err,
        n_B=n_B, T=T, eta=1.0, chi=chi,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q,
        P_eL_H=eleH.P, e_eL_H=eleH.e, s_eL_H=eleH.s,
        P_eL_Q=eleQ.P, e_eL_Q=eleQ.e, s_eL_Q=eleQ.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


def solve_eta1_fixed_chi_yc(T: float, chi: float, Y_C: float,
                             sfho_params: SFHoParams = None,
                             alphabag_params: AlphaBagParams = None,
                             particles: List = None,
                             initial_guess: np.ndarray = None,
                             include_pseudoscalar_mesons: bool = True,
                             include_gluons: bool = True,
                             include_photons: bool = True,
                             include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Solve mixed phase at η=1 (Maxwell) with fixed chi and fixed Y_C.
    
    Maxwell construction: phases are locally charge-neutral but separated.
    Each phase has its own electrons with local neutrality.
    
    12 unknowns: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μ_u, μ_d, μ_s, μ_eL_H, μ_eL_Q, n_B]
    """
    if sfho_params is None:
        from sfho_parameters import get_sfho_2fam_phi
        sfho_params = get_sfho_2fam_phi()
    if alphabag_params is None:
        alphabag_params = AlphaBagParams("default", 180.0, 0.1 * np.pi / 2)
    if particles is None:
        from sfho_eos import BARYONS_NYD
        particles = BARYONS_NYD
    
    n_B_est = chi * 0.6 + (1 - chi) * 0.4
    
    if initial_guess is None:
        initial_guess = np.array([
            50.0, 100.0, 5.0, 0.0,   # σ, ω, ρ, φ
            1100.0, -50.0,           # μ_B_H, μ_C_H
            250.0, 400.0, 400.0,     # μ_u, μ_d, μ_s
            50.0, 50.0,              # μ_eL_H, μ_eL_Q
            n_B_est                  # n_B
        ])
    
    def equations(x):
        sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_B = x
        mu_S_H = 0.0
        
        had = compute_sfho_thermo_from_mu_fields(
            mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons
        )
        res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
            sigma, omega, rho, phi,
            had.src_sigma, had.src_omega, had.src_rho, had.src_phi, sfho_params
        )
        
        qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
        
        # Gluon thermodynamics (only if include_gluons)
        glu_P = gluon_thermo(T, alphabag_params.alpha).P if include_gluons else 0.0
        
        eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        res = np.zeros(12)
        res[0] = res_sigma / 1e6
        res[1] = res_omega / 1e6
        res[2] = res_rho / 1e6
        res[3] = res_phi / 1e6
        res[4] = had.n_C - eleH.n  # Local neutrality H
        res[5] = qua.n_C - eleQ.n  # Local neutrality Q
        res[6] = (1 - chi) * had.n_C + chi * qua.n_C - n_B * Y_C  # Fixed Y_C
        res[7] = qua.mu_S  # Strangeness equilibrium
        res[8] = (1 - chi) * had.n_B + chi * qua.n_B - n_B  # Baryon conservation
        res[9] = (mu_B_H - qua.mu_B) / 1000.0  # Chemical equilibrium
        res[10] = (had.P + eleH.P - qua.P - glu_P - eleQ.P) / 100.0  # Mechanical equilibrium
        res[11] = (mu_C_H - qua.mu_C) / 100.0  # Charge chemical potential
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 2000}, tol=1e-8)
    
    sigma, omega, rho, phi, mu_B_H, mu_C_H, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_B = sol.x
    mu_S_H = 0.0
    
    had = compute_sfho_thermo_from_mu_fields(
        mu_B_H, mu_C_H, mu_S_H, sigma, omega, rho, phi, T, particles, sfho_params,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons
    )
    qua = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, alphabag_params)
    eleH = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    
    # Conditionally compute optional contributions
    gamma = photon_thermo(T) if include_photons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    gluon = gluon_thermo(T, alphabag_params.alpha) if include_gluons else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    nu_th = neutrino_thermo(0.0, T, include_antiparticles=True) if include_thermal_neutrinos else type('obj', (object,), {'P': 0.0, 'e': 0.0, 's': 0.0})()
    
    max_err = np.max(np.abs(sol.fun))
    is_converged = sol.success or max_err < 1e-6
    
    return MixedPhaseResult(
        converged=is_converged,
        error=max_err,
        n_B=n_B, T=T, eta=1.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        sigma_H=sigma, omega_H=omega, rho_H=rho, phi_H=phi,
        mu_B_H=mu_B_H, mu_C_H=mu_C_H, mu_S_H=mu_S_H,
        n_B_H=had.n_B, n_C_H=had.n_C, n_S_H=had.n_S,
        P_H=had.P, e_H=had.e, s_H=had.s, f_H=had.f,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s,
        n_u_Q=qua.n_u, n_d_Q=qua.n_d, n_s_Q=qua.n_s,
        mu_B_Q=qua.mu_B, mu_C_Q=qua.mu_C, mu_S_Q=qua.mu_S,
        n_B_Q=qua.n_B, n_C_Q=qua.n_C, n_S_Q=qua.n_S,
        P_Q=qua.P, e_Q=qua.e, s_Q=qua.s, f_Q=qua.f,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q,
        P_eL_H=eleH.P, e_eL_H=eleH.e, s_eL_H=eleH.s,
        P_eL_Q=eleQ.P, e_eL_Q=eleQ.e, s_eL_Q=eleQ.s,
        P_gamma=gamma.P, e_gamma=gamma.e, s_gamma=gamma.s,
        P_gluon=gluon.P, e_gluon=gluon.e, s_gluon=gluon.s,
        P_nu_th=3.0*nu_th.P, e_nu_th=3.0*nu_th.e, s_nu_th=3.0*nu_th.s,
    )


def solve_fixed_chi(T: float, chi: float, eta: float,
                    sfho_params: SFHoParams = None,
                    alphabag_params: AlphaBagParams = None,
                    particles: List = None,
                    initial_guess: np.ndarray = None,
                    eq_mode: str = "beta",
                    Y_C: float = None,
                    include_pseudoscalar_mesons: bool = True,
                    include_gluons: bool = True,
                    include_photons: bool = True,
                    include_thermal_neutrinos: bool = True) -> MixedPhaseResult:
    """
    Dispatch to appropriate fixed-chi solver based on eta and equilibrium mode.
    
    For phase boundary finding: χ is fixed, n_B is the unknown.
    """
    physics_opts = dict(
        include_pseudoscalar_mesons=include_pseudoscalar_mesons,
        include_gluons=include_gluons,
        include_photons=include_photons,
        include_thermal_neutrinos=include_thermal_neutrinos
    )
    
    if eq_mode == "fixed_yc":
        if Y_C is None:
            raise ValueError("Y_C must be provided for fixed_yc mode")
        if abs(eta) < 1e-10:
            return solve_eta0_fixed_chi_yc(T, chi, Y_C, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        elif abs(eta - 1.0) < 1e-10:
            return solve_eta1_fixed_chi_yc(T, chi, Y_C, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        else:
            raise NotImplementedError(f"Fixed-chi solver for eta={eta} not yet implemented")
    elif eq_mode == "beta":
        if abs(eta) < 1e-10:
            return solve_eta0_fixed_chi_beta(T, chi, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        elif abs(eta - 1.0) < 1e-10:
            return solve_eta1_fixed_chi_beta(T, chi, sfho_params, alphabag_params, particles, initial_guess, **physics_opts)
        else:
            raise NotImplementedError(f"Fixed-chi solver for eta={eta} not yet implemented")
    else:
        raise ValueError(f"Unknown equilibrium mode: {eq_mode}")


# =============================================================================
# PHASE BOUNDARY FINDING
# =============================================================================

def result_to_guess(result: MixedPhaseResult, eta: float = 0.0) -> np.ndarray:
    """Convert MixedPhaseResult to guess array for fixed-chi solver.
    
    For η=0 (Gibbs): 11 unknowns [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, n_B]
    For η=1 (Maxwell): 12 unknowns [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeL_H, μeL_Q, n_B]
    """
    if abs(eta - 1.0) < 1e-10:
        # η=1: need local electron chemical potentials
        # result.mu_eG stores average of local electrons, need to estimate individuals
        mu_eL_H = getattr(result, 'mu_eL_H', result.mu_eG)
        mu_eL_Q = getattr(result, 'mu_eL_Q', result.mu_eG)
        return np.array([
            result.sigma_H, result.omega_H, result.rho_H, result.phi_H,
            result.mu_B_H, result.mu_C_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            mu_eL_H, mu_eL_Q,
            result.n_B
        ])
    else:
        # η=0: global electrons
        return np.array([
            result.sigma_H, result.omega_H, result.rho_H, result.phi_H,
            result.mu_B_H, result.mu_C_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            result.mu_eG,
            result.n_B
        ])


def extrapolate_guess(history: list, T_target: float, fallback: np.ndarray) -> np.ndarray:
    """
    Linear extrapolation of guess from T history.
    
    history: list of (T, guess_array) tuples
    """
    if len(history) < 2:
        return fallback.copy()
    
    # Get last two points
    T1, g1 = history[-2]
    T2, g2 = history[-1]
    
    if abs(T2 - T1) < 1e-10:
        return g2.copy()
    
    # Linear extrapolation
    slope = (g2 - g1) / (T2 - T1)
    return g2 + slope * (T_target - T2)


def find_phase_boundaries_single(T: float, eta: float,
                                 sfho_params: SFHoParams, alphabag_params: AlphaBagParams,
                                 particles: List,
                                 initial_guess_onset: np.ndarray = None,
                                 initial_guess_offset: np.ndarray = None,
                                 eq_mode: str = "beta", Y_C: float = None,
                                 verbose: bool = False,
                                 include_pseudoscalar_mesons: bool = True,
                                 include_gluons: bool = True,
                                 include_photons: bool = True,
                                 include_thermal_neutrinos: bool = True) -> PhaseBoundaryResult:
    """
    Find phase boundaries for a single T.
    
    Uses provided initial guesses (from T-marching) if available.
    """
    result = PhaseBoundaryResult(T=T, eta=eta, converged=False)
    
    # Default guess if none provided
    if initial_guess_onset is None:
        initial_guess_onset = np.array([
            50.0, 100.0, 5.0, 0.0,  # σ, ω, ρ, φ
            950.0, -50.0,           # μ_B_H, μ_C_H
            300.0, 350.0, 350.0,    # μu, μd, μs
            50.0,                   # μeG
            0.4                     # n_B (onset estimate)
        ])
    
    # --- ONSET (χ=0) ---
    onset_result = None
    try:
        onset_result = solve_fixed_chi(
            T, chi=0.0, eta=eta,
            sfho_params=sfho_params,
            alphabag_params=alphabag_params,
            particles=particles,
            initial_guess=initial_guess_onset,
            eq_mode=eq_mode,
            Y_C=Y_C,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons,
            include_gluons=include_gluons,
            include_photons=include_photons,
            include_thermal_neutrinos=include_thermal_neutrinos
        )
        
        if onset_result.converged and onset_result.error < 1e-4:
            result.n_B_onset = onset_result.n_B
            result.converged_onset = True
            result.sigma_H_onset = onset_result.sigma_H
            result.omega_H_onset = onset_result.omega_H
            result.rho_H_onset = onset_result.rho_H
            result.phi_H_onset = onset_result.phi_H
            result.mu_B_H_onset = onset_result.mu_B_H
            result.mu_C_H_onset = onset_result.mu_C_H
            result.mu_u_Q_onset = onset_result.mu_u_Q
            result.mu_d_Q_onset = onset_result.mu_d_Q
            result.mu_s_Q_onset = onset_result.mu_s_Q
            result.mu_eG_onset = onset_result.mu_eG
    except Exception as e:
        if verbose:
            print(f"Onset exception: {e}")
    
    # --- OFFSET (χ=1) ---
    if initial_guess_offset is None:
        if result.converged_onset:
            # Use onset solution with scaled n_B
            initial_guess_offset = result_to_guess(onset_result, eta)
            initial_guess_offset[-1] = result.n_B_onset * 2.5  # Offset higher
        else:
            initial_guess_offset = initial_guess_onset.copy()
            initial_guess_offset[-1] = 1.0  # Default offset estimate
    
    offset_result = None
    try:
        offset_result = solve_fixed_chi(
            T, chi=1.0, eta=eta,
            sfho_params=sfho_params,
            alphabag_params=alphabag_params,
            particles=particles,
            initial_guess=initial_guess_offset,
            eq_mode=eq_mode,
            Y_C=Y_C,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons,
            include_gluons=include_gluons,
            include_photons=include_photons,
            include_thermal_neutrinos=include_thermal_neutrinos
        )
        
        if offset_result.converged and offset_result.error < 1e-4:
            result.n_B_offset = offset_result.n_B
            result.converged_offset = True
            result.sigma_H_offset = offset_result.sigma_H
            result.omega_H_offset = offset_result.omega_H
            result.rho_H_offset = offset_result.rho_H
            result.phi_H_offset = offset_result.phi_H
            result.mu_B_H_offset = offset_result.mu_B_H
            result.mu_C_H_offset = offset_result.mu_C_H
            result.mu_u_Q_offset = offset_result.mu_u_Q
            result.mu_d_Q_offset = offset_result.mu_d_Q
            result.mu_s_Q_offset = offset_result.mu_s_Q
            result.mu_eG_offset = offset_result.mu_eG
    except Exception as e:
        if verbose:
            print(f"Offset exception: {e}")
    
    result.converged = result.converged_onset and result.converged_offset
    return result, onset_result, offset_result


def find_all_boundaries(T_array: np.ndarray, eta: float,
                        sfho_params: SFHoParams, alphabag_params: AlphaBagParams,
                        particles: List,
                        H_table: dict = None, Q_table: dict = None, n_B_values: np.ndarray = None,
                        eq_mode: str = "beta", Y_C: float = None,
                        verbose: bool = False,
                        include_pseudoscalar_mesons: bool = True,
                        include_gluons: bool = True,
                        include_photons: bool = True,
                        include_thermal_neutrinos: bool = True) -> List[PhaseBoundaryResult]:
    """
    Find phase boundaries for all T values using bidirectional T-marching.
    
    Strategy:
    1. Find a working start T (try middle, then search)
    2. March upward from working start
    3. March downward from working start
    
    Uses previous T solution as initial guess for next T (warm-starting).
    Returns list of PhaseBoundaryResult for all T values.
    """
    T_sorted = np.sort(T_array)
    n_T = len(T_sorted)
    
    # Result storage: dict by T for easy lookup
    results_by_T = {}
    
    # Helper function to build initial guess from H_table at transition region
    def build_default_guess(T):
        if H_table is None or Q_table is None or n_B_values is None:
            return None
        
        # Find n_B where P_H ≈ P_Q (pressure crossing)
        best_n_B = None
        min_delta_P = float('inf')
        best_H = None
        best_Q = None
        
        for n_B in n_B_values:
            H_result = H_table.get((n_B, T))
            Q_result = Q_table.get((n_B, T))
            
            if H_result is None or Q_result is None:
                continue
            if not getattr(H_result, 'converged', True) or not getattr(Q_result, 'converged', True):
                continue
            
            P_H = getattr(H_result, 'P_total', 0.0)
            P_Q = getattr(Q_result, 'P_total', 0.0)
            
            delta_P = abs(P_H - P_Q)
            if delta_P < min_delta_P:
                min_delta_P = delta_P
                best_n_B = n_B
                best_H = H_result
                best_Q = Q_result
        
        if best_n_B is None or best_H is None:
            # Fallback: just use middle of density range
            for n_B in n_B_values[len(n_B_values)//3:2*len(n_B_values)//3]:
                H_result = H_table.get((n_B, T))
                if H_result is not None and getattr(H_result, 'converged', True):
                    best_n_B = n_B
                    best_H = H_result
                    break
            if best_H is None:
                return None
        
        # Get H phase values
        sigma = getattr(best_H, 'sigma', 50.0)
        omega = getattr(best_H, 'omega', 100.0)
        rho = getattr(best_H, 'rho', 5.0)
        phi = getattr(best_H, 'phi', 0.0)
        mu_B_H = getattr(best_H, 'mu_B', 950.0)
        mu_C_H = getattr(best_H, 'mu_C', -50.0)
        mu_e_H = getattr(best_H, 'mu_e', 50.0)
        
        # Build quark chemical potentials from equilibrium relations
        # μ_B_Q = μ_B_H (baryon chemical equilibrium)
        # μ_S_Q = 0 (strangeness equilibrium) → μ_s = μ_d
        # Beta eq: μ_u - μ_d = -μ_e (charge equilibrium in Q sector)
        # μ_B_Q = 2μ_d + μ_u = 3μ_d - μ_e → μ_d = (μ_B_H + μ_e) / 3
        # μ_u = μ_d - μ_e
        mu_d = (mu_B_H + mu_e_H) / 3.0
        mu_u = mu_d - mu_e_H
        mu_s = mu_d  # μ_S = 0
        
        # Build guess: different structure for different eta
        if abs(eta - 1.0) < 1e-10:
            # η=1: 12 unknowns [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeL_H, μeL_Q, n_B]
            mu_eL_H = mu_e_H
            mu_eL_Q = mu_e_H
            return np.array([
                sigma, omega, rho, phi,
                mu_B_H, mu_C_H,
                mu_u, mu_d, mu_s,
                mu_eL_H, mu_eL_Q,
                best_n_B
            ])
        else:
            # η=0: 11 unknowns [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, n_B]
            mu_eG = mu_e_H
            return np.array([
                sigma, omega, rho, phi,
                mu_B_H, mu_C_H,
                mu_u, mu_d, mu_s,
                mu_eG,
                best_n_B
            ])
    
    # --- STEP 1: Find working start (try middle of T range first) ---
    idx_start = n_T // 2
    working_T = None
    working_guess = None
    working_result = None
    
    # Try from middle outward
    for delta in range(n_T):
        for direction in [0, 1]:  # Try both directions
            if direction == 0:
                idx = idx_start + delta
            else:
                idx = idx_start - delta - 1
            
            if idx < 0 or idx >= n_T:
                continue
            
            T = T_sorted[idx]
            guess = build_default_guess(T)
            
            if verbose and guess is not None:
                print(f"        [T={T:.1f}] Trying with guess: n_B={guess[-1]:.4f}, μ_B_H={guess[4]:.1f}, μu={guess[6]:.1f}, μd={guess[7]:.1f}, err=", end="", flush=True)
            elif verbose:
                print(f"        [T={T:.1f}] No guess built", flush=True)
                continue
            
            result, onset_result, offset_result = find_phase_boundaries_single(
                T, eta, sfho_params, alphabag_params, particles,
                initial_guess_onset=guess,
                eq_mode=eq_mode, Y_C=Y_C, verbose=False,
                include_pseudoscalar_mesons=include_pseudoscalar_mesons,
                include_gluons=include_gluons,
                include_photons=include_photons,
                include_thermal_neutrinos=include_thermal_neutrinos
            )
            
            if verbose:
                if onset_result is not None:
                    off_err = offset_result.error if offset_result is not None else float('nan')
                    print(f"onset_err={onset_result.error:.2e}, offset_err={off_err:.2e}, conv_on={result.converged_onset}, conv_off={result.converged_offset}")
                else:
                    print("N/A")
            
            if result.converged:
                working_T = T
                working_idx = idx
                working_result = result
                working_guess = result_to_guess(onset_result, eta) if onset_result else None
                working_offset_guess = result_to_guess(offset_result, eta) if offset_result else None
                results_by_T[T] = result
                if verbose:
                    print(f"        ✓ Found working start at T={T:.1f} MeV")
                break
        
        if working_T is not None:
            break
    
    if working_T is None:
        # No working start found, return empty results
        if verbose:
            print("      ✗ No working start found")
        return [PhaseBoundaryResult(T=T, eta=eta, converged=False) for T in T_sorted]
    
    # --- STEP 2: March upward from working start ---
    onset_history_up = [(working_T, working_guess.copy())] if working_guess is not None else []
    offset_history_up = [(working_T, working_offset_guess.copy())] if working_offset_guess is not None else []
    prev_onset = working_guess
    prev_offset = working_offset_guess
    
    for idx in range(working_idx + 1, n_T):
        T = T_sorted[idx]
        
        # Build guess from history
        if len(onset_history_up) >= 2:
            guess_onset = extrapolate_guess(onset_history_up, T, prev_onset)
        elif prev_onset is not None:
            guess_onset = prev_onset.copy()
        else:
            guess_onset = build_default_guess(T)
        
        if len(offset_history_up) >= 2:
            guess_offset = extrapolate_guess(offset_history_up, T, prev_offset)
        elif prev_offset is not None:
            guess_offset = prev_offset.copy()
        else:
            guess_offset = None
        
        result, onset_result, offset_result = find_phase_boundaries_single(
            T, eta, sfho_params, alphabag_params, particles,
            initial_guess_onset=guess_onset,
            initial_guess_offset=guess_offset,
            eq_mode=eq_mode, Y_C=Y_C, verbose=False,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons,
            include_gluons=include_gluons,
            include_photons=include_photons,
            include_thermal_neutrinos=include_thermal_neutrinos
        )
        
        results_by_T[T] = result
        
        if result.converged_onset and onset_result is not None:
            prev_onset = result_to_guess(onset_result, eta)
            onset_history_up.append((T, prev_onset.copy()))
        
        if result.converged_offset and offset_result is not None:
            prev_offset = result_to_guess(offset_result, eta)
            offset_history_up.append((T, prev_offset.copy()))
    
    # --- STEP 3: March downward from working start ---
    onset_history_down = [(working_T, working_guess.copy())] if working_guess is not None else []
    offset_history_down = [(working_T, working_offset_guess.copy())] if working_offset_guess is not None else []
    prev_onset = working_guess
    prev_offset = working_offset_guess
    
    for idx in range(working_idx - 1, -1, -1):
        T = T_sorted[idx]
        
        # Build guess from history (note: extrapolating backward)
        if len(onset_history_down) >= 2:
            guess_onset = extrapolate_guess(onset_history_down, T, prev_onset)
        elif prev_onset is not None:
            guess_onset = prev_onset.copy()
        else:
            guess_onset = build_default_guess(T)
        
        if len(offset_history_down) >= 2:
            guess_offset = extrapolate_guess(offset_history_down, T, prev_offset)
        elif prev_offset is not None:
            guess_offset = prev_offset.copy()
        else:
            guess_offset = None
        
        result, onset_result, offset_result = find_phase_boundaries_single(
            T, eta, sfho_params, alphabag_params, particles,
            initial_guess_onset=guess_onset,
            initial_guess_offset=guess_offset,
            eq_mode=eq_mode, Y_C=Y_C, verbose=False,
            include_pseudoscalar_mesons=include_pseudoscalar_mesons,
            include_gluons=include_gluons,
            include_photons=include_photons,
            include_thermal_neutrinos=include_thermal_neutrinos
        )
        
        results_by_T[T] = result
        
        if result.converged_onset and onset_result is not None:
            prev_onset = result_to_guess(onset_result, eta)
            onset_history_down.append((T, prev_onset.copy()))
        
        if result.converged_offset and offset_result is not None:
            prev_offset = result_to_guess(offset_result, eta)
            offset_history_down.append((T, prev_offset.copy()))
    
    # --- Collect and print results in order ---
    boundaries = []
    for i_T, T in enumerate(T_sorted):
        result = results_by_T.get(T, PhaseBoundaryResult(T=T, eta=eta, converged=False))
        boundaries.append(result)
        
        if verbose or (i_T % 10 == 0):
            if result.converged:
                print(f"      ✓ T={T:6.1f} MeV: onset={result.n_B_onset:.4f}, offset={result.n_B_offset:.4f}")
            else:
                onset_str = f"{result.n_B_onset:.4f}" if result.converged_onset else "---"
                offset_str = f"{result.n_B_offset:.4f}" if result.converged_offset else "---"
                print(f"      ✗ T={T:6.1f} MeV: onset={onset_str}, offset={offset_str}")
    
    return boundaries


# Keep old function signature for compatibility
def find_phase_boundaries(T: float, eta: float,
                          sfho_params: SFHoParams, alphabag_params: AlphaBagParams,
                          particles: List,
                          H_table: dict, Q_table: dict, n_B_values: np.ndarray,
                          eq_mode: str = "beta", Y_C: float = None,
                          verbose: bool = False,
                          initial_guess_onset: np.ndarray = None,
                          initial_guess_offset: np.ndarray = None,
                          include_pseudoscalar_mesons: bool = True,
                          include_gluons: bool = True,
                          include_photons: bool = True,
                          include_thermal_neutrinos: bool = True) -> PhaseBoundaryResult:
    """
    Find phase boundaries for a single T (legacy interface).
    
    For T-marching, use find_all_boundaries instead.
    """
    # Build guess from H_table if not provided
    if initial_guess_onset is None and H_table is not None:
        for n_B in n_B_values:
            H_result = H_table.get((n_B, T))
            if H_result is not None and getattr(H_result, 'converged', True):
                initial_guess_onset = np.array([
                    getattr(H_result, 'sigma', 50.0),
                    getattr(H_result, 'omega', 100.0),
                    getattr(H_result, 'rho', 5.0),
                    getattr(H_result, 'phi', 0.0),
                    getattr(H_result, 'mu_B', 950.0),
                    getattr(H_result, 'mu_C', -50.0),
                    300.0, 350.0, 350.0,
                    50.0,
                    n_B
                ])
                break
    
    result, _, _ = find_phase_boundaries_single(
        T, eta, sfho_params, alphabag_params, particles,
        initial_guess_onset=initial_guess_onset,
        initial_guess_offset=initial_guess_offset,
        eq_mode=eq_mode, Y_C=Y_C, verbose=verbose,
        include_pseudoscalar_mesons=include_pseudoscalar_mesons,
        include_gluons=include_gluons,
        include_photons=include_photons,
        include_thermal_neutrinos=include_thermal_neutrinos
    )
    return result


def generate_unified_table(n_B_values: np.ndarray, T_array: np.ndarray, eta: float,
                           sfho_params: SFHoParams, alphabag_params: AlphaBagParams,
                           particles: List,
                           H_table: dict, Q_table: dict,
                           boundaries: dict,
                           eq_mode: str = "beta", Y_C: float = None,
                           include_pseudoscalar_mesons: bool = True,
                           include_gluons: bool = True,
                           include_photons: bool = True,
                           include_thermal_neutrinos: bool = True,
                           verbose: bool = False) -> List:
    """
    Generate unified EOS table combining pure H, mixed, and pure Q phases.
    
    For each (n_B, T):
      - n_B < n_onset: use pure H from H_table
      - n_onset ≤ n_B ≤ n_offset: solve mixed phase
      - n_B > n_offset: use pure Q from Q_table
    """
    results = []
    import time as time_module
    
    # Physics options to pass to solvers
    physics_opts = dict(
        include_pseudoscalar_mesons=include_pseudoscalar_mesons,
        include_gluons=include_gluons,
        include_photons=include_photons,
        include_thermal_neutrinos=include_thermal_neutrinos
    )
    
    for i_T, T in enumerate(T_array):
        t_start_T = time_module.time()
        
        # Get boundaries for this T
        bounds = boundaries.get(T)
        n_onset = bounds['n_onset'] if bounds else None
        n_offset = bounds['n_offset'] if bounds else None
        
        # Counters for this T
        n_H = 0
        n_mixed = 0
        n_mixed_converged = 0
        n_Q = 0
        
        # Get onset solution for initial guess (if available)
        onset_guess = None
        if bounds and 'boundary' in bounds:
            boundary_result = bounds['boundary']
            if boundary_result.converged_onset:
                # Build guess from onset solution
                onset_guess = np.array([
                    boundary_result.sigma_H_onset,
                    boundary_result.omega_H_onset,
                    boundary_result.rho_H_onset,
                    boundary_result.phi_H_onset,
                    boundary_result.mu_B_H_onset,
                    boundary_result.mu_C_H_onset,
                    boundary_result.mu_u_Q_onset,
                    boundary_result.mu_d_Q_onset,
                    boundary_result.mu_s_Q_onset,
                    boundary_result.mu_eG_onset
                ])
        
        # Previous result for guess extrapolation
        prev_result = None
        first_mixed = True
        
        for n_B in n_B_values:
            # Determine which phase
            if n_onset is None or n_B < n_onset:
                # Pure H phase
                result = H_table.get((n_B, T))
                if result is None:
                    continue
                n_H += 1
            elif n_offset is not None and n_B > n_offset:
                # Pure Q phase
                result = Q_table.get((n_B, T))
                if result is None:
                    continue
                n_Q += 1
            else:
                # Mixed phase
                n_mixed += 1
                
                # Estimate χ from n_B position between onset and offset
                if n_onset is not None and n_offset is not None and n_offset > n_onset:
                    chi_est = (n_B - n_onset) / (n_offset - n_onset)
                    chi_est = max(0.01, min(0.99, chi_est))  # Keep in (0,1)
                else:
                    chi_est = 0.5
                
                # Build initial guess
                # η=0 needs 11 elements: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μeG, χ]
                # η=1 needs 12 elements: [σ, ω, ρ, φ, μ_B_H, μ_C_H, μu, μd, μs, μ_eL_H, μ_eL_Q, χ]
                if first_mixed and onset_guess is not None:
                    if abs(eta) < 1e-10:  # η=0
                        guess = np.append(onset_guess, chi_est)
                    else:  # η=1
                        # Insert local electron chemical potentials (use μeG for both)
                        mu_eL_H_est = onset_guess[9] if len(onset_guess) > 9 else 50.0
                        mu_eL_Q_est = onset_guess[9] if len(onset_guess) > 9 else 50.0
                        guess = np.concatenate([onset_guess[:9], [mu_eL_H_est, mu_eL_Q_est, chi_est]])
                    first_mixed = False
                elif prev_result is not None and hasattr(prev_result, 'sigma_H'):
                    if abs(eta) < 1e-10:  # η=0
                        guess = np.array([
                            prev_result.sigma_H, prev_result.omega_H,
                            prev_result.rho_H, prev_result.phi_H,
                            prev_result.mu_B_H, prev_result.mu_C_H,
                            prev_result.mu_u_Q, prev_result.mu_d_Q, prev_result.mu_s_Q,
                            prev_result.mu_eG,
                            chi_est
                        ])
                    else:  # η=1
                        mu_eL_H = getattr(prev_result, 'mu_eL_H', prev_result.mu_eG)
                        mu_eL_Q = getattr(prev_result, 'mu_eL_Q', prev_result.mu_eG)
                        guess = np.array([
                            prev_result.sigma_H, prev_result.omega_H,
                            prev_result.rho_H, prev_result.phi_H,
                            prev_result.mu_B_H, prev_result.mu_C_H,
                            prev_result.mu_u_Q, prev_result.mu_d_Q, prev_result.mu_s_Q,
                            mu_eL_H, mu_eL_Q,
                            chi_est
                        ])
                else:
                    guess = None
                
                try:
                    result = solve_mixed_phase(
                        n_B, T, eta,
                        sfho_params=sfho_params,
                        alphabag_params=alphabag_params,
                        particles=particles,
                        eq_mode=eq_mode,
                        Y_C=Y_C,
                        initial_guess=guess,
                        **physics_opts
                    )
                    if result.converged:
                        n_mixed_converged += 1
                        first_mixed = False
                except Exception as e:
                    if verbose:
                        print(f"[n_B={n_B:.3f} failed]", end="")
                    continue
            
            if result is not None:
                results.append(result)
                prev_result = result
        
        t_elapsed_T = time_module.time() - t_start_T
        n_total_T = n_H + n_mixed + n_Q
        ms_per_pt = (t_elapsed_T * 1000 / n_mixed) if n_mixed > 0 else 0
        conv_pct = (100.0 * n_mixed_converged / n_mixed) if n_mixed > 0 else 100.0
        
        if verbose or (i_T % 10 == 0):
            if bounds:
                print(f"      T={T:6.1f} MeV: H={n_H:3d}, mixed={n_mixed_converged:3d}/{n_mixed:3d} ({conv_pct:5.1f}%), Q={n_Q:3d} | {t_elapsed_T:.1f}s, {ms_per_pt:.1f}ms/pt")
            else:
                print(f"      T={T:6.1f} MeV: no boundaries (pure H only)")
    
    return results


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("SFHo + AlphaBag Mixed Phase EOS Test")
    print("=" * 60)
    
    # Test parameters
    n_B = 0.5  # fm^-3
    T = 10.0   # MeV
    
    print(f"\nTest: n_B = {n_B} fm⁻³, T = {T} MeV")
    print("-" * 40)
    
    result = solve_eta0_beta(n_B, T)
    
    print(f"Converged: {result.converged}")
    print(f"Error: {result.error:.2e}")
    print(f"χ = {result.chi:.4f}")
    print(f"n_B_tot = {result.n_B_tot:.4f} fm⁻³ (input: {n_B})")
    print(f"P_H = {result.P_H:.2f} MeV/fm³")
    print(f"P_Q = {result.P_Q:.2f} MeV/fm³")
    print(f"μ_B_H = {result.mu_B_H:.1f} MeV")
    print(f"μ_B_Q = {result.mu_B_Q:.1f} MeV")
    
    print("\n✓ Test complete")
