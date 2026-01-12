"""
zlvmit_mixed_phase_eos.py
=========================
ZL (nucleonic) + vMIT (quark) mixed phase EOS solver.

This module provides:
    - Mixed phase solvers for all η values (Gibbs, Maxwell, intermediate)
    - Phase boundary finding (onset at χ=0, offset at χ=1)
    - Unified EOS table generation
    - Isentrope computation

Equilibrium modes:
    - BETA: Beta equilibrium with electrons (charge neutrality)
    - FIXED_YC: Fixed charge fraction Y_C (with electrons for neutrality)
    - TRAPPED: Trapped neutrinos (fixed lepton fraction Y_L)

Solver unknowns:
    - η=0 (global total electric charge neutrality): 12 [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns, χ]
    - η=1 (local total electric charge neutrality): 13 [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, χ]
    - 0<η<1: 14 [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, χ]

Note: μi and χ are physical unknowns; ni are solved via consistency ni(μi,T) since μi_eff(μi, ni) due to interactions.

File structure:
    PART 1: Data structures (EquilibriumMode, MixedPhaseResult, PhaseBoundaryResult)
    PART 2: Core solvers (beta, fixed_yc, fixed-chi)
    PART 3: Phase boundary finding (T-marching)
    PART 4: Table generation
"""

import numpy as np
import os
import time
from scipy.optimize import root
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

# ZL (hadronic) modules
from zl_parameters import ZLParams, get_zl_default
from zl_thermodynamics_nucleons import compute_zl_thermo_from_mu_n
from zl_eos import (
    get_default_guess_beta_eq as get_zl_guess,
    solve_zl_trapped_neutrinos,
    result_to_guess as zl_result_to_guess,
    solve_zl_beta_eq as solve_pure_H_beta,
    solve_zl_fixed_yc as solve_pure_H_fixed_yc
)

# vMIT (quark) modules
from vmit_parameters import VMITParams, get_vmit_default
from vmit_thermodynamics_quarks import compute_vmit_thermo_from_mu_n
from vmit_eos import (
    get_default_guess_beta_eq as get_vmit_guess, 
    solve_vmit_trapped_neutrinos,
    result_to_guess as vmit_result_to_guess,
    solve_vmit_beta_eq as solve_pure_Q_beta,
    solve_vmit_fixed_yc as solve_pure_Q_fixed_yc
)

# General physics modules
from general_thermodynamics_leptons import electron_thermo, neutrino_thermo, photon_thermo
from general_physics_constants import hc, PI2
import general_particles


# =============================================================================
# =============================================================================
# PART 1: DATA STRUCTURES
# =============================================================================
# =============================================================================

# =============================================================================
# EQUILIBRIUM MODE ENUM
# =============================================================================
class EquilibriumMode(Enum):
    """Equilibrium condition for mixed phase calculation."""
    BETA = "beta"           # Beta equilibrium with electrons (charge neutrality)
    FIXED_YC = "fixed_yc"   # Fixed charge fraction (no electrons)
    TRAPPED = "trapped"     # Trapped neutrinos (fixed lepton fraction Y_L)



# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class MixedPhaseResult:
    """Result from mixed phase calculation."""
    converged: bool
    error: float
    
    # Inputs
    n_B: float
    T: float
    eta: float
    
    # Solution
    chi: float  # Quark volume fraction
    
    # Hadronic phase - particle chemical potentials
    mu_p_H: float
    mu_n_H: float
    n_p_H: float
    n_n_H: float
    P_H: float  # Hadronic pressure (WITHOUT electrons/photons - added separately)
    
    # Quark phase - particle chemical potentials
    mu_u_Q: float
    mu_d_Q: float
    mu_s_Q: float
    n_u_Q: float
    n_d_Q: float
    n_s_Q: float
    P_Q: float  # Quark pressure (WITHOUT electrons/photons - added separately)
    
    # ==========================================================================
    # Electrons
    # ==========================================================================
    mu_eL_H: float = 0.0  # Local electrons in H
    mu_eL_Q: float = 0.0  # Local electrons in Q
    mu_eG: float = 0.0    # Global electrons
    n_eL_H: float = 0.0
    n_eL_Q: float = 0.0
    n_eG: float = 0.0
    P_eL_H: float = 0.0   # Electron pressure in H
    P_eL_Q: float = 0.0   # Electron pressure in Q
    P_eG: float = 0.0     # Global electron pressure
    
    # ==========================================================================
    # Neutrinos (for trapped neutrino mode)
    # ==========================================================================
    mu_nuG: float = 0.0     # Global neutrino μ
    n_nuG: float = 0.0      # Global neutrino density
    P_nuG: float = 0.0      # Global neutrino pressure
    e_nuG: float = 0.0      # Global neutrino energy density
    s_nuG: float = 0.0      # Global neutrino entropy density
    
    # ==========================================================================
    # Equilibrium mode inputs
    # ==========================================================================
    eq_mode: str = "beta"   # "beta", "fixed_yc", or "trapped"
    Y_C_input: float = 0.0  # Input charge fraction (for fixed_yc mode)
    Y_L_input: float = 0.0  # Input lepton fraction (for trapped mode)
    
    # ==========================================================================
    # Conserved charge chemical potentials (B=baryon, C=charge, S=strangeness)
    # ==========================================================================
    # Hadronic phase: μ_B = μ_n, μ_C = μ_p - μ_n, μ_S = 0
    mu_B_H: float = 0.0
    mu_C_H: float = 0.0
    mu_S_H: float = 0.0
    
    # Quark phase: μ_B = (μ_u + 2μ_d), μ_C = (μ_u - μ_d), μ_S = (μ_s - μ_d)
    mu_B_Q: float = 0.0
    mu_C_Q: float = 0.0
    mu_S_Q: float = 0.0
    
    # ==========================================================================
    # Conserved charge densities
    # ==========================================================================
    # Hadronic phase
    n_B_H: float = 0.0   # n_p + n_n
    n_C_H: float = 0.0   # n_p (proton = charge carrier)
    n_S_H: float = 0.0   # 0 (no strangeness in H)
    
    # Quark phase  
    n_B_Q: float = 0.0   # (n_u + n_d + n_s) / 3
    n_C_Q: float = 0.0   # (2*n_u - n_d - n_s) / 3
    n_S_Q: float = 0.0   # n_s (strangeness = 1 for s quark)
    
    # ==========================================================================
    # Thermodynamic quantities per phase
    # ==========================================================================
    # Hadronic phase
    e_H: float = 0.0   # Energy density
    s_H: float = 0.0   # Entropy density
    f_H: float = 0.0   # Free energy density (f = e - Ts)
    
    # Quark phase
    e_Q: float = 0.0   # Energy density
    s_Q: float = 0.0   # Entropy density
    f_Q: float = 0.0   # Free energy density
    
    # Electron thermodynamics per phase
    e_eL_H: float = 0.0   # Electron energy density in H
    s_eL_H: float = 0.0   # Electron entropy density in H
    e_eL_Q: float = 0.0   # Electron energy density in Q
    s_eL_Q: float = 0.0   # Electron entropy density in Q
    e_eG: float = 0.0     # Global electron energy density
    s_eG: float = 0.0     # Global electron entropy density
    
    # Photon thermodynamics
    P_gamma: float = 0.0  # Photon pressure
    e_gamma: float = 0.0  # Photon energy density
    s_gamma: float = 0.0  # Photon entropy density
    
    # ==========================================================================
    # Total quantities (DERIVED via @property)
    # P_total = (1-χ)*P_H + χ*P_Q + P_e_tot + P_γ
    # where P_e_tot = η*(1-χ)*P_eL_H + η*χ*P_eL_Q + (1-η)*P_eG
    # Note: P_H and P_Q are WITHOUT electrons/photons to avoid double counting
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
    # Total neutrino quantities (for trapped mode)
    # ==========================================================================
    
    @property
    def P_nu_tot(self) -> float:
        """Total neutrino pressure (volume-weighted)."""
        return self.P_nuG
    
    @property
    def e_nu_tot(self) -> float:
        """Total neutrino energy density (volume-weighted)."""
        return self.e_nuG
    
    @property
    def s_nu_tot(self) -> float:
        """Total neutrino entropy density (volume-weighted)."""
        return self.s_nuG
    
    @property
    def n_nu_tot(self) -> float:
        """Total neutrino density (volume-weighted)."""
        return self.n_nuG
    
    @property
    def n_L_tot(self) -> float:
        """Total lepton density: n_e + n_nu (for trapped mode)."""
        return self.n_e_tot + self.n_nu_tot
    
    @property
    def Y_L_tot(self) -> float:
        """Lepton fraction: (n_e + n_nu) / n_B."""
        return self.n_L_tot / self.n_B 
    
    # ==========================================================================
    # Total thermodynamic quantities (including neutrinos)
    # ==========================================================================
    
    @property
    def P_total(self) -> float:
        """Total pressure: (1-χ)*P_H + χ*P_Q + P_e_tot + P_nu_tot + P_γ"""
        return ((1 - self.chi) * self.P_H + self.chi * self.P_Q + 
                self.P_e_tot + self.P_nu_tot + self.P_gamma)
    
    @property
    def e_total(self) -> float:
        """Total energy density (volume-weighted + leptons + photons)."""
        return ((1 - self.chi) * self.e_H + self.chi * self.e_Q + 
                self.e_e_tot + self.e_nu_tot + self.e_gamma)
    
    @property
    def s_total(self) -> float:
        """Total entropy density (volume-weighted + leptons + photons)."""
        return ((1 - self.chi) * self.s_H + self.chi * self.s_Q + 
                self.s_e_tot + self.s_nu_tot + self.s_gamma)
    
    @property
    def f_total(self) -> float:
        """Total free energy density: f = e - Ts"""
        return self.e_total - self.T * self.s_total
    
    
    # ==========================================================================
    # Per-phase particle fractions: Y_i_H = n_i_H / n_B_H, Y_i_Q = n_i_Q / n_B_Q
    # ==========================================================================
    
    # Hadronic phase fractions
    @property
    def Y_p_H(self) -> float:
        """Proton fraction in H phase: n_p_H / n_B_H"""
        return self.n_p_H / self.n_B_H 
    
    @property
    def Y_n_H(self) -> float:
        """Neutron fraction in H phase: n_n_H / n_B_H"""
        return self.n_n_H / self.n_B_H 
    
    # Quark phase fractions
    @property
    def Y_u_Q(self) -> float:
        """Up quark fraction in Q phase: n_u_Q / n_B_Q"""
        return self.n_u_Q / self.n_B_Q 
    
    @property
    def Y_d_Q(self) -> float:
        """Down quark fraction in Q phase: n_d_Q / n_B_Q"""
        return self.n_d_Q / self.n_B_Q 
    
    @property
    def Y_s_Q(self) -> float:
        """Strange quark fraction in Q phase: n_s_Q / n_B_Q"""
        return self.n_s_Q / self.n_B_Q 
    
    # ==========================================================================
    # Total particle fractions: Y_i_tot = volume-weighted contribution to n_B
    # ==========================================================================
    
    @property
    def Y_p_tot(self) -> float:
        """Total proton fraction: (1-χ)*n_p_H / n_B"""
        return (1 - self.chi) * self.n_p_H / self.n_B 
    
    @property
    def Y_n_tot(self) -> float:
        """Total neutron fraction: (1-χ)*n_n_H / n_B"""
        return (1 - self.chi) * self.n_n_H / self.n_B 
    
    @property
    def Y_u_tot(self) -> float:
        """Total up quark fraction: χ*n_u_Q / n_B"""
        return self.chi * self.n_u_Q / self.n_B 
    
    @property
    def Y_d_tot(self) -> float:
        """Total down quark fraction: χ*n_d_Q / n_B"""
        return self.chi * self.n_d_Q / self.n_B 
    
    @property
    def Y_s_tot(self) -> float:
        """Total strange quark fraction: χ*n_s_Q / n_B"""
        return self.chi * self.n_s_Q / self.n_B 
    
    @property
    def Y_e_tot(self) -> float:
        """Total electron fraction: n_e_tot / n_B"""
        return self.n_e_tot / self.n_B 
    
    # ==========================================================================
    # Per-phase charge fractions: Y_X_H = n_X_H / n_B_H, Y_X_Q = n_X_Q / n_B_Q
    # ==========================================================================
    
    @property
    def Y_C_H(self) -> float:
        """Charge fraction in H phase: n_C_H / n_B_H = n_p_H / n_B_H"""
        return self.n_C_H / self.n_B_H 
    
    @property
    def Y_S_H(self) -> float:
        """Strangeness fraction in H phase (should be 0)."""
        return self.n_S_H / self.n_B_H 
    
    @property
    def Y_C_Q(self) -> float:
        """Charge fraction in Q phase: n_C_Q / n_B_Q"""
        return self.n_C_Q / self.n_B_Q 
    
    @property
    def Y_S_Q(self) -> float:
        """Strangeness fraction in Q phase: n_S_Q / n_B_Q"""
        return self.n_S_Q / self.n_B_Q

    # ==========================================================================
    # Total charge densities
    # ==========================================================================
    
    @property
    def n_B_tot(self) -> float:
        """Total baryon density (should be equal to n_B)."""
        return ((1 - self.chi) * self.n_B_H + self.chi * self.n_B_Q) # for checking consistency
    
    @property
    def n_C_tot(self) -> float:
        """Total charge density."""
        return ((1 - self.chi) * self.n_C_H + self.chi * self.n_C_Q) 
    
    @property
    def n_S_tot(self) -> float:
        """Total strangeness density."""
        return ((1 - self.chi) * self.n_S_H + self.chi * self.n_S_Q)

    # ==========================================================================
    # Total charge fractions
    # ==========================================================================
    
    @property
    def Y_B_tot(self) -> float:
        """Total baryon fraction (should be 1)."""
        return self.n_B_tot / self.n_B 
    
    @property
    def Y_C_tot(self) -> float:
        """Total charge fraction."""
        return self.n_C_tot / self.n_B
    
    @property
    def Y_S_tot(self) -> float:
        """Total strangeness fraction."""
        return self.n_S_tot / self.n_B


# =============================================================================
# =============================================================================
# =============================================================================
# PART 2: CORE SOLVERS
# =============================================================================
# =============================================================================


# =============================================================================
# η=0 GCN beta equilibrium [Gibbs construction] (12 unknowns)
# =============================================================================
def solve_eta0_beta(n_B: float, T: float,
                    zl_params: ZLParams = None,
                    vmit_params: VMITParams = None,
                    initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 (Gibbs construction).
    
    12 unknowns: [μp, μn, μu, μd, μs, μe, np, nn, nu, nd, ns, χ]
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6. μ_C_H + μ_e = 0                            (β-equilibrium in H sector)
        7. μ_C_Q + μ_e = 0                            (β-equilibrium in Q sector)
        8. μ_S_Q = 0                                  (weak strangeness eq in Q sector)
        9. (1-χ)*n_B_H + χ*n_B_Q = n_B                (baryon number conservation)
        10. (1-χ)*n_C_H + χ*n_C_Q - n_e = 0           (global charge neutrality)
        11. μ_B_H = μ_B_Q                             (baryon chemical equilibrium)
        12. P_H = P_Q                                 (mechanical equilibrium)

        note: µ_eG_H=µ_eG_Q by construction (=µ_eG)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = np.array([
            700, 1300, 150, 500, 500, 150,  # μp, μn, μu, μd, μs, μe
            0.1, 0.3, 0.01, 0.7, 0.5, -0.05  # np, nn, nu, nd, ns, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = x

        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s

        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        n_S_H = had_sec.n_S
        n_S_Q = qua_sec.n_S

        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_H = had_sec.mu_S
        mu_S_Q = qua_sec.mu_S

        n_eG = ele_sec.n
    
        P_H = had_sec.P
        P_Q = qua_sec.P
        
        res = np.zeros(12)
        res[0] = n_p_calc - n_p  # p consistency
        res[1] = n_n_calc - n_n  # n consistency
        res[2] = n_u_calc - n_u  # u consistency
        res[3] = n_d_calc - n_d  # d consistency
        res[4] = n_s_calc - n_s  # s consistency
        res[5] = mu_C_H + mu_eG  # beta-eq in Hadronic sector
        res[6] = mu_C_Q + mu_eG  # beta-eq in Quark sector
        res[7] = mu_S_Q  # weak strangeness eq in Quark sector
        res[8] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon number global conservation
        res[9] = (1 - chi) * n_C_H + chi * n_C_Q - n_eG  # global charge neutrality
        res[10] = mu_B_H - mu_B_Q  # baryon chemical equilibrium
        res[11] = P_H - P_Q  # mechanical equilibrium
    
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 1000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 1000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = sol.x

    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n, 
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eG=mu_eG, n_eG=ele_sec.n, 
        P_eG=ele_sec.P, e_eG=ele_sec.e, s_eG=ele_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )

# =============================================================================
# η=1 LCN beta equilibrium [Maxwell construction] (13 unknowns)
# =============================================================================
def solve_eta1_beta(n_B: float, T: float,
                    zl_params: ZLParams = None,
                    vmit_params: VMITParams = None,
                    initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 (Maxwell construction).
    
    13 unknowns: [μp, μn, μu, μd, μs, μeH, μeQ, np, nn, nu, nd, ns, χ]
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6. (1-χ)n_B_H + χ*n_B_Q = nB              (baryon conservation)
        7. n_C_H = n_eH                            (local neutrality in H)
        8. n_C_Q = n_eQ                            (local neutrality in Q)
        9. μ_B_H = μ_B_Q                           (chemical equilibrium)
        10. μ_C_H + μ_eH = 0                       (beta-eq in H sector)
        11. μ_C_Q + μ_eQ = 0                       (beta-eq in Q sector)
        12. μ_S_Q = 0                              (strangeness equilibrium)
        13. P_H + P_eH = P_Q + P_eQ                (pressure equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        initial_guess = np.array([
            1334.0, 1714.5, 563.1, 575.7, 575.7, 380.5, 12.6,  # μp, μn, μu, μd, μs, μeL_H, μeL_Q
            0.242, 0.592, 1.084, 1.181, 0.988, -1.23  # np, nn, nu, nd, ns, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, chi  = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_H = had_sec.mu_S
        mu_S_Q = qua_sec.mu_S

        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eH = eleH_sec.P
        P_eQ = eleQ_sec.P
        
        res = np.zeros(13)
        res[0] = n_p_calc - n_p  # p consistency
        res[1] = n_n_calc - n_n  # n consistency
        res[2] = n_u_calc - n_u  # u consistency
        res[3] = n_d_calc - n_d  # d consistency
        res[4] = n_s_calc - n_s  # s consistency
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon global conservation
        res[6] = n_C_H - n_eL_H  # local neutrality H
        res[7] = n_C_Q - n_eL_Q  # local neutrality Q
        res[8] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[9] = mu_C_H + mu_eL_H  # beta-eq in H
        res[10] = mu_C_Q + mu_eL_Q  # beta-eq in Q
        res[11] = mu_S_Q  # strangeness eq
        res[12] = P_H + P_eH - P_Q - P_eQ  # pressure eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 1000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 1000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, chi = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=1.0, chi=chi,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )
    
# =============================================================================
# 0<η<1 New framework beta equilibrium (14 unknowns)
# =============================================================================
def solve_etaX_beta(n_B: float, T: float, eta: float,
                    zl_params: ZLParams = None,
                    vmit_params: VMITParams = None,
                    initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for 0 < η < 1.
    
    14 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, χ]
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6.   (1-χ)*n_B_H + χ*n_B_Q = n_B                    (baryon conservation)
        7.   n_C_H = n_eL_H                                 (local neutrality H)
        8.   n_C_Q = n_eL_Q                                 (local neutrality Q)
        9.   (1-χ)*n_C_H + χ*n_C_Q = n_eG                   (global neutrality)
        10.  μ_B_H = μ_B_Q                                  (baryon chemical equilibrium)
        11.  μ_C_H + η*μ_eL_H + (1-η)*μ_eG = 0              (beta-eq in H sector)
        12.  μ_C_Q + η*μ_eL_Q + (1-η)*μ_eG = 0              (beta-eq in Q sector)
        13.  μ_S_Q = 0                                      (strangeness equilibrium)
        14.  P_H + η*P_eL_H = P_Q + η*P_eL_Q                (mechanical equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        initial_guess = np.array([
            1389.9, 1619.1, 447.5, 585.8, 585.8, 418.7, -490.7, 418.8,  # μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG
            0.323, 0.527, 0.474, 1.354, 1.151, -0.0003  # np, nn, nu, nd, ns, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        n_eG = eleG_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P
        
        res = np.zeros(14)
        res[0] = n_p_calc - n_p  # p consistency
        res[1] = n_n_calc - n_n  # n consistency
        res[2] = n_u_calc - n_u  # u consistency
        res[3] = n_d_calc - n_d  # d consistency
        res[4] = n_s_calc - n_s  # s consistency
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = n_C_H - n_eL_H  # local neutrality H
        res[7] = n_C_Q - n_eL_Q  # local neutrality Q
        res[8] = (1 - chi) * n_C_H + chi * n_C_Q - n_eG  # global neutrality
        res[9] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[10] = mu_C_H + eta * mu_eL_H + (1 - eta) * mu_eG  # beta-eq H
        res[11] = mu_C_Q + eta * mu_eL_Q + (1 - eta) * mu_eG  # beta-eq Q
        res[12] = mu_S_Q  # weak strangeness eq in Q
        res[13] = P_H + eta * P_eL_H - P_Q - eta * P_eL_Q  # mechanical eq
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 1000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 1000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=eta, chi=chi,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, mu_eG=mu_eG,
        n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n, n_eG=eleG_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_eG=eleG_sec.P, e_eG=eleG_sec.e, s_eG=eleG_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# η=0 GCN FIXED Y_C (with electrons)
# =============================================================================
def solve_eta0_fixed_yc(n_B: float, Y_C: float, T: float,
                        zl_params: ZLParams = None,
                        vmit_params: VMITParams = None,
                        initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 with fixed charge fraction Y_C.

    Electrons are included to ensure charge neutrality:
    
    12 unknowns: [μp, μn, μu, μd, μs, μe, np, nn, nu, nd, ns, χ]
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6. (1-χ)*n_B_H + χ*n_B_Q = n_B             (baryon conservation)
        7. (1-χ)*n_C_H + χ*n_C_Q = n_B * Y_C       (charge global conservation)
        8. n_e = n_B * Y_C                         (global charge neutrality)
        9. μ_B_H = μ_B_Q                           (baryon chemical equilibrium)
        10. μ_C_H = μ_C_Q                          (charge chemical equilibrium)
        11. μ_S_Q = 0                              (strangeness weak equilibrium in Q)
        12. P_H = P_Q                              (mechanical equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = np.array([
            700, 1300, 150, 500, 500, 100,  # μp, μn, μu, μd, μs, μe
            Y_C * n_B, (1 - Y_C) * n_B, n_B, n_B, 0.5*n_B, 0.1  # np, nn, nu, nd, ns, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eG = ele_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B

        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C

        mu_S_H = had_sec.mu_S
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        
        # Target electron density from fixed Y_C
        n__target = Y_C * n_B
        
        res = np.zeros(12)
        res[0] = (n_p_calc - n_p)/max(abs(n_p), 1e-8)                                   # p consistency
        res[1] = (n_n_calc - n_n)/max(abs(n_n), 1e-8)                                   # n consistency
        res[2] = (n_u_calc - n_u)/max(abs(n_u), 1e-8)                                   # u consistency
        res[3] = (n_d_calc - n_d)/max(abs(n_d), 1e-8)                                   # d consistency
        res[4] = (n_s_calc - n_s)/max(abs(n_s), 1e-8)                                   # s consistency
        res[5] = ((1 - chi) * n_B_H + chi * n_B_Q - n_B)/max(abs(n_B), 1e-8)            # Baryon conservation
        res[6] = ((1 - chi) * n_C_H + chi * n_C_Q - n_B * Y_C)/max(abs(n_B * Y_C), 1e-8)      # C global conservation
        res[7] = (n_B * Y_C - n_eG)/max(abs(n_B * Y_C), 1e-8)                                   # Global electric charge neutrality
        res[8] = (mu_B_H - mu_B_Q)/max(abs(mu_B_H), 1e-8)                                 # Baryon chemical eq
        res[9] = (mu_C_H - mu_C_Q)/max(abs(mu_C_H), 1e-8)                                  # Charge chemical eq
        res[10] = (mu_S_Q)/max(abs(mu_B_Q), 1e-8)                                          # Strangeness weak chemical eq
        res[11] = (P_H - P_Q)/max(abs(P_H), 1e-8)                                       # Mechanical equilibrium
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eG=mu_eG, n_eG=ele_sec.n,
        P_eG=ele_sec.P, e_eG=ele_sec.e, s_eG=ele_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# 0 < η < 1 MIXED SURFACE TENSION - FIXED Y_C (with electrons)
# =============================================================================
def solve_etaX_fixed_yc(n_B: float, Y_C: float, T: float, eta: float,
                        zl_params: ZLParams = None,
                        vmit_params: VMITParams = None,
                        initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for 0 < η < 1 with fixed charge fraction Y_C.
    
    14 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, χ]
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6.   (1-χ)*n_B_H + χ*n_B_Q = n_B                     (baryon conservation)
        7.   (1-χ)*n_C_H + χ*n_C_Q = n_B * Y_C               (charge conservation)
        8.   n_C_H = n_eL_H                                  (local neutrality H)
        9.   n_C_Q = n_eL_Q                                  (local neutrality Q)
        10.  n_eG = n_B * Y_C                                (global neutrality)
        11.  μ_B_H = μ_B_Q                                   (baryon chemical eq)
        12.  μ_C_H + η*μ_eL_H = μ_C_Q + η*μ_eL_Q             (charge chemical eq)
        13.  μ_S_Q = 0                                       (weak strangeness eq in Q)
        14.  P_H + η*P_eL_H = P_Q + η*P_eL_Q                 (mechanical equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Check initial guess size - this solver needs 14 elements
    if initial_guess is not None and len(initial_guess) != 14:
        initial_guess = None
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = np.array([
            1389.9, 1619.1, 447.5, 585.8, 585.8, 418.7, -490.7, 418.8,  # μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG
            0.323, 0.527, 0.474, 1.354, 1.151, -0.0003  # np, nn, nu, nd, ns, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        n_eG = eleG_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P

        
        res = np.zeros(14)
        res[0] = n_p_calc - n_p  # p consistency
        res[1] = n_n_calc - n_n  # n consistency
        res[2] = n_u_calc - n_u  # u consistency
        res[3] = n_d_calc - n_d  # d consistency
        res[4] = n_s_calc - n_s  # s consistency
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = (1 - chi) * n_C_H + chi * n_C_Q - n_B * Y_C  # charge conservation
        res[7] = n_C_H - n_eL_H  # local neutrality H
        res[8] = n_C_Q - n_eL_Q  # local neutrality Q
        res[9] = n_B * Y_C - n_eG  # global neutrality
        res[10] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[11] = mu_C_H + eta * mu_eL_H - mu_C_Q - eta * mu_eL_Q  # charge chemical eq
        res[12] = mu_S_Q  # weak strangeness eq in Q
        res[13] = P_H + eta * P_eL_H - P_Q - eta * P_eL_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, chi = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=eta, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, mu_eG=mu_eG,
        n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n, n_eG=eleG_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_eG=eleG_sec.P, e_eG=eleG_sec.e, s_eG=eleG_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# η=1 LCN FIXED Y_C (13 unknowns)
# =============================================================================
def solve_eta1_fixed_yc(n_B: float, Y_C: float, T: float,
                        zl_params: ZLParams = None,
                        vmit_params: VMITParams = None,
                        initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 with fixed Y_C.
    
    13 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, χ]
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6.   (1-χ)*n_B_H + χ*n_B_Q = n_B                  (baryon conservation)
        7.   n_C_H = n_eL_H                               (local neutrality H)
        8.   n_C_Q = n_eL_Q                               (local neutrality Q)
        9.   μ_B_H = μ_B_Q                                (baryon chemical eq)
        10.  μ_C_H = μ_C_Q                                (charge chemical eq)
        11.  μ_S_Q = 0                                    (weak strangeness eq in Q)
        12.  P_H + P_eL_H = P_Q + P_eL_Q                  (mechanical equilibrium)
        13.  (1-χ)*n_eL_H + χ*n_eL_Q = n_B * Y_C         (fixed Y_C constraint)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Check initial guess size - this solver needs 13 elements
    if initial_guess is not None and len(initial_guess) != 13:
        initial_guess = None
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = np.array([
            1334.0, 1714.5, 563.1, 575.7, 575.7, 380.5, 12.6,  # μp, μn, μu, μd, μs, μeL_H, μeL_Q
            0.242, 0.592, 1.084, 1.181, 0.988, -1.23  # np, nn, nu, nd, ns, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, chi = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P
        
        res = np.zeros(13)
        res[0] = n_p_calc - n_p  # p consistency
        res[1] = n_n_calc - n_n  # n consistency
        res[2] = n_u_calc - n_u  # u consistency
        res[3] = n_d_calc - n_d  # d consistency
        res[4] = n_s_calc - n_s  # s consistency
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = (1 - chi) * n_C_H + chi * n_C_Q - n_B * Y_C  # charge conservation
        res[7] = n_C_H - n_eL_H  # local neutrality H
        res[8] = n_C_Q - n_eL_Q  # local neutrality Q
        res[9] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[10] = mu_C_H + mu_eL_H - mu_C_Q - mu_eL_Q  # charge chemical eq
        res[11] = mu_S_Q  # weak strangeness eq in Q
        res[12] = P_H + P_eL_H - P_Q - P_eL_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, chi = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=1.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# η=0 FIXED-CHI SOLVER (for finding phase boundaries)
# =============================================================================
def solve_eta0_fixed_chi_beta(T: float, chi: float,
                              zl_params: ZLParams = None,
                              vmit_params: VMITParams = None,
                         initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 with χ FIXED (Gibbs construction).
    
    Used to find phase boundaries:
    - χ=0: onset of mixed phase (n_B_onset)
    - χ=1: end of mixed phase (n_B_offset)
    
    12 unknowns: [μp, μn, μu, μd, μs, μe, np, nn, nu, nd, ns, n_B]
    Note: n_B is now the unknown, χ is the input!
    
    Same equations as solve_eta0_beta, but with χ fixed.
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        n_B_est = 0.48 if chi < 0.5 else 1.5
        initial_guess = get_default_guess(chi=chi, eta=0.0, n_B_est=n_B_est, T=T,
                                          zl_params=zl_params, vmit_params=vmit_params,
                                          fixed_chi=True)
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eG = ele_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        
        res = np.zeros(12)
        res[0] = n_p_calc - n_p  # p consistency
        res[1] = n_n_calc - n_n  # n consistency
        res[2] = n_u_calc - n_u  # u consistency
        res[3] = n_d_calc - n_d  # d consistency
        res[4] = n_s_calc - n_s  # s consistency
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = (1 - chi) * n_C_H + chi * n_C_Q - n_eG  # global neutrality
        res[7] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[8] = mu_C_H + mu_eG  # beta-eq
        res[9] = mu_C_Q + mu_eG  # charge chemical eq
        res[10] = mu_S_Q  # strangeness eq
        res[11] = P_H - P_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='hybr', options={'maxfev': 10000}, tol=1e-10)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = sol.x

    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eG=mu_eG, n_eG=ele_sec.n,
        P_eG=ele_sec.P, e_eG=ele_sec.e, s_eG=ele_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# η=0 FIXED-CHI SOLVER - FIXED Y_C (for finding phase boundaries)
# =============================================================================
def solve_eta0_fixed_chi_yc(T: float, chi: float, Y_C: float,
                            zl_params: ZLParams = None,
                            vmit_params: VMITParams = None,
                            initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=0 with χ FIXED and fixed charge fraction Y_C.
    
    Used to find phase boundaries for fixed Y_C case:
    - χ=0: onset of mixed phase (n_B_onset)
    - χ=1: end of mixed phase (n_B_offset)
    
    In fixed Y_C mode, electrons ARE included with constraint:
        n_e = n_B * Y_C  (electron density fixed by charge fraction)
    
    12 unknowns: [μp, μn, μu, μd, μs, μe, np, nn, nu, nd, ns, n_B]
    Note: n_B is now the unknown, χ is the input!
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6. (1-χ)(np+nn) + χ(nu+nd+ns)/3 = nB      (baryon conservation)
        7. (1-χ)n_C_H + χ*n_C_Q - n_e = 0         (global charge neutrality)
        8. n_e = n_B * Y_C                         (fixed charge fraction)
        9. μn = μu + 2μd                          (chemical equilibrium)
        10. μp = 2μu + μd                         (chemical equilibrium)
        11. P_H = P_Q                             (pressure equilibrium)
        12. μd = μs                               (strangeness equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        n_B_est = 0.48 if chi < 0.5 else 1.5
        initial_guess = get_default_guess(chi=chi, eta=0.0, n_B_est=n_B_est, T=T,
                                          zl_params=zl_params, vmit_params=vmit_params,
                                          fixed_chi=True)
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eG = ele_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        
        
        res = np.zeros(12)
        res[0] = n_p_calc - n_p
        res[1] = n_n_calc - n_n
        res[2] = n_u_calc - n_u
        res[3] = n_d_calc - n_d
        res[4] = n_s_calc - n_s
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = (1 - chi) * n_C_H + chi * n_C_Q - n_B * Y_C  # C global conservation
        res[7] = n_B * Y_C - n_eG  # global neutrality
        res[8] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[9] = mu_C_H - mu_C_Q  # charge chemical eq
        res[10] = mu_S_Q  # strangeness eq
        res[11] = P_H - P_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    ele_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eG=mu_eG, n_eG=ele_sec.n,
        P_eG=ele_sec.P, e_eG=ele_sec.e, s_eG=ele_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )





# =============================================================================
# η=1 FIXED-CHI SOLVER - FIXED Y_C (for phase boundaries)
# =============================================================================
def solve_eta1_fixed_chi_yc(T: float, chi: float, Y_C: float,
                            zl_params: ZLParams = None,
                            vmit_params: VMITParams = None,
                            initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 with χ FIXED and fixed charge fraction Y_C.
    
    Used to find phase boundaries for fixed Y_C case:
    - χ=0: onset of mixed phase (n_B_onset)
    - χ=1: end of mixed phase (n_B_offset)
    
    In η=1 Maxwell construction, each phase is locally neutral with its own electrons.
    
    13 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, n_B]
    
    Equations:
        1-5.  Self-consistency for np, nn, nu, nd, ns
        6.    (1-χ)*n_B_H + χ*n_B_Q = n_B                  (baryon conservation)
        7.    (1-χ)*n_C_H + χ*n_C_Q = n_B * Y_C            (charge conservation)
        8.    n_C_H = n_eL_H                               (local neutrality H)
        9.    n_C_Q = n_eL_Q                               (local neutrality Q)
        10.   μ_B_H = μ_B_Q                                (baryon chemical eq)
        11.   μ_C_H + μ_eL_H = μ_C_Q + μ_eL_Q              (charge chemical eq)
        12.   μ_S_Q = 0                                    (weak strangeness eq in Q)
        13.   P_H + P_eL_H = P_Q + P_eL_Q                  (mechanical equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        n_B_est = 0.48 if chi < 0.5 else 1.5
        initial_guess = get_default_guess(chi=chi, eta=1.0, n_B_est=n_B_est, T=T,
                                          zl_params=zl_params, vmit_params=vmit_params,
                                          fixed_chi=True)

    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, n_B = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P
        
        res = np.zeros(13)
        res[0] = n_p_calc - n_p
        res[1] = n_n_calc - n_n
        res[2] = n_u_calc - n_u
        res[3] = n_d_calc - n_d
        res[4] = n_s_calc - n_s
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = (1 - chi) * n_C_H + chi * n_C_Q - n_B * Y_C  # charge conservation
        res[7] = n_C_H - n_eL_H  # local neutrality H
        res[8] = n_C_Q - n_eL_Q  # local neutrality Q
        res[9] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[10] = mu_C_H + mu_eL_H - mu_C_Q - mu_eL_Q  # charge chemical eq
        res[11] = mu_S_Q  # strangeness eq
        res[12] = P_H + P_eL_H - P_Q - P_eL_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, n_B = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=1.0, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )





# =============================================================================
# η=1 FIXED-CHI SOLVER (for finding phase boundaries)
# =============================================================================
def solve_eta1_fixed_chi_beta(T: float, chi: float,
                              zl_params: ZLParams = None,
                              vmit_params: VMITParams = None,
                        initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for η=1 (Maxwell) with χ FIXED.
    
    Used to find phase boundaries:
    - χ=0: onset of mixed phase (n_B_onset)
    - χ=1: end of mixed phase (n_B_offset)
    
    13 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, n_B]
    Note: n_B is now the unknown, χ is the input!
    
    Equations:
        1-5.  Self-consistency for np, nn, nu, nd, ns
        6.    (1-χ)*n_B_H + χ*n_B_Q = n_B                  (baryon conservation)
        7.    n_C_H = n_eL_H                               (local neutrality H)
        8.    n_C_Q = n_eL_Q                               (local neutrality Q)
        9.    μ_B_H = μ_B_Q                                (baryon chemical eq)
        10.   μ_C_H + μ_eL_H = 0                           (beta-eq in H)
        11.   μ_C_Q + μ_eL_Q = 0                           (beta-eq in Q)
        12.   μ_S_Q = 0                                    (strangeness eq)
        13.   P_H + P_eL_H = P_Q + P_eL_Q                  (mechanical eq)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        n_B_est = 0.48 if chi < 0.5 else 1.5
        initial_guess = get_default_guess(chi=chi, eta=1.0, n_B_est=n_B_est, T=T,
                                          zl_params=zl_params, vmit_params=vmit_params,
                                          fixed_chi=True)
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, n_B = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P
        
        res = np.zeros(13)
        res[0] = n_p_calc - n_p
        res[1] = n_n_calc - n_n
        res[2] = n_u_calc - n_u
        res[3] = n_d_calc - n_d
        res[4] = n_s_calc - n_s
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = n_C_H - n_eL_H  # local neutrality H
        res[7] = n_C_Q - n_eL_Q  # local neutrality Q
        res[8] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[9] = mu_C_H + mu_eL_H  # beta-eq in H
        res[10] = mu_C_Q + mu_eL_Q  # beta-eq in Q
        res[11] = mu_S_Q  # strangeness eq
        res[12] = P_H + P_eL_H - P_Q - P_eL_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='hybr', options={'maxfev': 10000}, tol=1e-10)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, n_p, n_n, n_u, n_d, n_s, n_B = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=1.0, chi=chi,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )




# =============================================================================
# 0 < η < 1 FIXED-CHI SOLVER (for finding phase boundaries)
# =============================================================================
def solve_etaX_fixed_chi_beta(T: float, chi: float, eta: float,
                              zl_params: ZLParams = None,
                              vmit_params: VMITParams = None,
                        initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for 0 < η < 1 with χ FIXED.
    
    Used to find phase boundaries:
    - χ=0: onset of mixed phase (n_B_onset)
    - χ=1: end of mixed phase (n_B_offset)
    
    14 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, n_B]
    Note: n_B is now the unknown, χ is the input!
    
    Equations:
        1-5. Self-consistency for np, nn, nu, nd, ns
        6.   (1-χ)*n_B_H + χ*n_B_Q = n_B                    (baryon conservation)
        7.   n_C_H = n_eL_H                                 (local neutrality H)
        8.   n_C_Q = n_eL_Q                                 (local neutrality Q)
        9.   (1-χ)*n_C_H + χ*n_C_Q = n_eG                   (global neutrality)
        10.  μ_B_H = μ_B_Q                                  (baryon chemical equilibrium)
        11.  μ_C_H + η*μ_eL_H + (1-η)*μ_eG = 0              (beta-eq in H sector)
        12.  μ_C_Q + η*μ_eL_Q + (1-η)*μ_eG = 0              (beta-eq in Q sector)
        13.  μ_S_Q = 0                                      (strangeness equilibrium)
        14.  P_H + η*P_eL_H = P_Q + η*P_eL_Q                (mechanical equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        n_B_est = 0.48 if chi < 0.5 else 1.5
        initial_guess = get_default_guess(chi=chi, eta=eta, n_B_est=n_B_est, T=T,
                                          zl_params=zl_params, vmit_params=vmit_params,
                                          fixed_chi=True)
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        n_eG = eleG_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P
        
        res = np.zeros(14)
        res[0] = n_p_calc - n_p
        res[1] = n_n_calc - n_n
        res[2] = n_u_calc - n_u
        res[3] = n_d_calc - n_d
        res[4] = n_s_calc - n_s
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = n_C_H - n_eL_H  # local neutrality H
        res[7] = n_C_Q - n_eL_Q  # local neutrality Q
        res[8] = (1 - chi) * n_C_H + chi * n_C_Q - n_eG  # global neutrality
        res[9] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[10] = mu_C_H + eta * mu_eL_H + (1 - eta) * mu_eG  # beta-eq H
        res[11] = mu_C_Q + eta * mu_eL_Q + (1 - eta) * mu_eG  # beta-eq Q
        res[12] = mu_S_Q  # strangeness eq
        res[13] = P_H + eta * P_eL_H - P_Q - eta * P_eL_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        sol = root(equations, sol.x, method='lm', options={'maxiter': 5000}, tol=1e-8)
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-4:
        # Third attempt with different starting point
        sol = root(equations, sol.x, method='hybr', options={'maxfev': 10000}, tol=1e-10)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=eta, chi=chi,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, mu_eG=mu_eG,
        n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n, n_eG=eleG_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_eG=eleG_sec.P, e_eG=eleG_sec.e, s_eG=eleG_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# 0 < η < 1 FIXED-CHI SOLVER - FIXED Y_C (for phase boundaries)
# =============================================================================
def solve_etaX_fixed_chi_yc(T: float, chi: float, eta: float, Y_C: float,
                            zl_params: ZLParams = None,
                            vmit_params: VMITParams = None,
                            initial_guess: np.ndarray = None) -> MixedPhaseResult:
    """
    Solve mixed phase for 0 < η < 1 with χ FIXED and fixed charge fraction Y_C.
    
    Used to find phase boundaries for fixed Y_C case.
    Three electron populations (local H, local Q, global).
    
    14 unknowns: [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, n_B]
    
    Equations:
        1-5.  Self-consistency for np, nn, nu, nd, ns
        6.    (1-χ)*n_B_H + χ*n_B_Q = n_B                     (baryon conservation)
        7.    (1-χ)*n_C_H + χ*n_C_Q = n_B * Y_C               (charge conservation)
        8.    n_C_H = n_eL_H                                  (local neutrality H)
        9.    n_C_Q = n_eL_Q                                  (local neutrality Q)
        10.   n_eG = n_B * Y_C                                (global neutrality)
        11.   μ_B_H = μ_B_Q                                   (baryon chemical eq)
        12.   μ_C_H + η*μ_eL_H = μ_C_Q + η*μ_eL_Q             (charge chemical eq)
        13.   μ_S_Q = 0                                       (weak strangeness eq in Q)
        14.   P_H + η*P_eL_H = P_Q + η*P_eL_Q                 (mechanical equilibrium)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if initial_guess is None:
        n_B_est = 0.48 if chi < 0.5 else 1.5
        initial_guess = get_default_guess(chi=chi, eta=eta, n_B_est=n_B_est, T=T,
                                          zl_params=zl_params, vmit_params=vmit_params,
                                          eq_mode="fixed_yc", Y_C=Y_C, fixed_chi=True)
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
        eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
        eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_eL_H = eleH_sec.n
        n_eL_Q = eleQ_sec.n
        n_eG = eleG_sec.n
        
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        P_H = had_sec.P
        P_Q = qua_sec.P
        P_eL_H = eleH_sec.P
        P_eL_Q = eleQ_sec.P
        
        res = np.zeros(14)
        res[0] = n_p_calc - n_p
        res[1] = n_n_calc - n_n
        res[2] = n_u_calc - n_u
        res[3] = n_d_calc - n_d
        res[4] = n_s_calc - n_s
        res[5] = (1 - chi) * n_B_H + chi * n_B_Q - n_B  # baryon conservation
        res[6] = (1 - chi) * n_C_H + chi * n_C_Q - n_B * Y_C  # charge conservation
        res[7] = n_C_H - n_eL_H  # local neutrality H
        res[8] = n_C_Q - n_eL_Q  # local neutrality Q
        res[9] = n_B * Y_C - n_eG  # global neutrality
        res[10] = mu_B_H - mu_B_Q  # baryon chemical eq
        res[11] = mu_C_H + eta * mu_eL_H - mu_C_Q - eta * mu_eL_Q  # charge chemical eq
        res[12] = mu_S_Q  # strangeness eq
        res[13] = P_H + eta * P_eL_H - P_Q - eta * P_eL_Q  # mechanical eq
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_eL_H, mu_eL_Q, mu_eG, n_p, n_n, n_u, n_d, n_s, n_B = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    eleH_sec = electron_thermo(mu_eL_H, T, include_antiparticles=True)
    eleQ_sec = electron_thermo(mu_eL_Q, T, include_antiparticles=True)
    eleG_sec = electron_thermo(mu_eG, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=eta, chi=chi,
        eq_mode="fixed_yc", Y_C_input=Y_C,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eL_H=mu_eL_H, mu_eL_Q=mu_eL_Q, mu_eG=mu_eG,
        n_eL_H=eleH_sec.n, n_eL_Q=eleQ_sec.n, n_eG=eleG_sec.n,
        P_eL_H=eleH_sec.P, e_eL_H=eleH_sec.e, s_eL_H=eleH_sec.s,
        P_eL_Q=eleQ_sec.P, e_eL_Q=eleQ_sec.e, s_eL_Q=eleQ_sec.s,
        P_eG=eleG_sec.P, e_eG=eleG_sec.e, s_eG=eleG_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# UNIFIED SOLVER
# =============================================================================
def solve_mixed_phase(n_B: float, T: float, eta: float,
                      zl_params: ZLParams = None,
                      vmit_params: VMITParams = None,
                      initial_guess: np.ndarray = None,
                      eq_mode: str = "beta",
                      Y_C: float = None,
                      Y_L: float = None) -> MixedPhaseResult:
    """
    Unified solver for ZL+vMIT mixed phase.
    
    Dispatches to appropriate solver based on η value and equilibrium mode.
    
    Args:
        n_B: Baryon density in fm^-3
        T: Temperature in MeV
        eta: parameter controlling local-global total electric charge neutrality (0=Gibbs, 1=Maxwell)
        zl_params: ZL model parameters (default if None)
        vmit_params: vMIT model parameters (default if None)
        initial_guess: Initial guess array (size depends on eta and eq_mode)
        eq_mode: "beta" for beta-equilibrium, "fixed_yc" for fixed charge fraction, "trapped" for neutrino trapping
        Y_C: Charge fraction Y_C = n_C/n_B (required for fixed_yc mode)
        Y_L: Lepton fraction Y_L = (n_e + n_ν)/n_B (required for trapped mode)
    
    Returns:
        MixedPhaseResult with all thermodynamic quantities
    """
    if eq_mode == "trapped":
        if Y_L is None:
            raise ValueError("Y_L must be provided for trapped mode")
        from zlvmit_trapped_solvers import solve_eta0_trapped
        if abs(eta) < 1e-10:
            return solve_eta0_trapped(n_B, Y_L, T, zl_params, vmit_params, initial_guess)
        else:
            # For now, only η=0 trapped is implemented
            raise NotImplementedError(f"Trapped mode for eta={eta} not yet implemented (only eta=0 available)")
    elif eq_mode == "fixed_yc":
        if Y_C is None:
            raise ValueError("Y_C must be provided for fixed_yc mode")
        if abs(eta) < 1e-10:
            return solve_eta0_fixed_yc(n_B, Y_C, T, zl_params, vmit_params, initial_guess)
        elif abs(eta - 1.0) < 1e-10:
            return solve_eta1_fixed_yc(n_B, Y_C, T, zl_params, vmit_params, initial_guess)
        else:
            return solve_etaX_fixed_yc(n_B, Y_C, T, eta, zl_params, vmit_params, initial_guess)
    elif eq_mode == "beta": 
        if abs(eta) < 1e-10:
            return solve_eta0_beta(n_B, T, zl_params, vmit_params, initial_guess)
        elif abs(eta - 1.0) < 1e-10:
            return solve_eta1_beta(n_B, T, zl_params, vmit_params, initial_guess)
        else:
            return solve_etaX_beta(n_B, T, eta, zl_params, vmit_params, initial_guess)
    else:
        raise ValueError("Invalid eq_mode: must be 'fixed_yc', 'beta', or 'trapped'")




# =============================================================================
# UNIFIED FIXED-CHI SOLVER
# =============================================================================
def solve_fixed_chi(T: float, chi: float, eta: float,
                    zl_params: ZLParams = None,
                    vmit_params: VMITParams = None,
                    initial_guess: np.ndarray = None,
                    eq_mode: str = "beta",
                    Y_C: float = None,
                    Y_L: float = None) -> MixedPhaseResult:
    """
    Dispatch to appropriate fixed-chi solver based on eta and equilibrium mode.
    
    Args:
        eq_mode: "beta", "fixed_yc", or "trapped"
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
    
    Includes retry logic with perturbed initial guesses for robustness.
    """
    def solve_once(guess):
        if eq_mode == "trapped":
            if Y_L is None:
                raise ValueError("Y_L must be provided for trapped mode")
            from zlvmit_trapped_solvers import solve_eta0_fixed_chi_trapped
            if abs(eta) < 1e-10:
                return solve_eta0_fixed_chi_trapped(T, chi, Y_L, zl_params, vmit_params, guess)
            else:
                raise NotImplementedError(f"Trapped mode fixed-chi for eta={eta} not yet implemented")
        elif eq_mode == "fixed_yc":
            # Fixed Y_C mode - supports all eta values
            if abs(eta) < 1e-10:
                return solve_eta0_fixed_chi_yc(T, chi, Y_C, zl_params, vmit_params, guess)
            elif abs(eta - 1.0) < 1e-10:
                return solve_eta1_fixed_chi_yc(T, chi, Y_C, zl_params, vmit_params, guess)
            else:
                return solve_etaX_fixed_chi_yc(T, chi, eta, Y_C, zl_params, vmit_params, guess)
        elif eq_mode == "beta":
            if abs(eta) < 1e-10:
                return solve_eta0_fixed_chi_beta(T, chi, zl_params, vmit_params, guess)
            elif abs(eta - 1.0) < 1e-10:
                return solve_eta1_fixed_chi_beta(T, chi, zl_params, vmit_params, guess)
            else:
                return solve_etaX_fixed_chi_beta(T, chi, eta, zl_params, vmit_params, guess)
        else:
            raise ValueError("Invalid eq_mode: must be 'fixed_yc', 'beta', or 'trapped'")
    
    return solve_once(initial_guess)


# =============================================================================
# =============================================================================
# PART 3: PHASE BOUNDARY FINDING AND HELPER FUNCTIONS
# =============================================================================
# =============================================================================

# =============================================================================
# PHASE BOUNDARY RESULT DATACLASS
# =============================================================================
@dataclass
class PhaseBoundaryResult:
    """Result from phase boundary search at a given T."""
    T: float
    eta: float
    n_B_onset: float   # n_B at χ=0 (start of mixed phase)
    n_B_offset: float  # n_B at χ=1 (end of mixed phase)
    converged_onset: bool
    converged_offset: bool
    error_onset: float
    error_offset: float
    
    # Full solution at χ=0 (onset) - for warm-starting table generation
    mu_p_H_onset: float = 0.0
    mu_n_H_onset: float = 0.0
    mu_u_Q_onset: float = 0.0
    mu_d_Q_onset: float = 0.0
    mu_s_Q_onset: float = 0.0
    mu_eL_H_onset: float = 0.0  # Local electron μ in H phase
    mu_eL_Q_onset: float = 0.0  # Local electron μ in Q phase
    mu_eG_onset: float = 0.0    # Global electron μ
    n_p_H_onset: float = 0.0
    n_n_H_onset: float = 0.0
    n_u_Q_onset: float = 0.0
    n_d_Q_onset: float = 0.0
    n_s_Q_onset: float = 0.0
    n_eL_H_onset: float = 0.0
    n_eL_Q_onset: float = 0.0
    n_eG_onset: float = 0.0

    # Full solution at χ=1 (offset) - if needed
    mu_p_H_offset: float = 0.0
    mu_n_H_offset: float = 0.0
    mu_u_Q_offset: float = 0.0
    mu_d_Q_offset: float = 0.0
    mu_s_Q_offset: float = 0.0
    mu_eL_H_offset: float = 0.0  # Local electron μ in H phase
    mu_eL_Q_offset: float = 0.0  # Local electron μ in Q phase
    mu_eG_offset: float = 0.0    # Global electron μ
    n_p_H_offset: float = 0.0
    n_n_H_offset: float = 0.0
    n_u_Q_offset: float = 0.0
    n_d_Q_offset: float = 0.0
    n_s_Q_offset: float = 0.0
    n_eL_H_offset: float = 0.0
    n_eL_Q_offset: float = 0.0
    n_eG_offset: float = 0.0

    


# =============================================================================
# FILE I/O
# =============================================================================
def get_boundary_filename(eta: float, output_dir: str = "output",
                          zl_params: ZLParams = None, vmit_params: VMITParams = None,
                          eq_mode: str = "beta") -> str:
    """Get filename for phase boundary table.
    
    One file per eta contains all composition values (Y_C or Y_L as first column).
    
    Args:
        eta: parameter controlling local-global total electric charge neutrality
        output_dir: Directory for output files
        zl_params: ZL model parameters
        vmit_params: vMIT model parameters
        eq_mode: Equilibrium mode ("beta", "fixed_yc", or "trapped")
    """
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Build filename based on equilibrium mode (but NOT on Y_C/Y_L)
    base = f"phase_boundaries_eta{eta:.2f}"
    
    # Add mode suffix for non-beta modes
    if eq_mode == "fixed_yc":
        base += "_fixedYC"
    elif eq_mode == "trapped":
        base += "_trapped"
    # Beta equilibrium: no mode suffix (default)
    
    base += f"_B{int(vmit_params.B4)}_a{vmit_params.a}.dat"
    
    return os.path.join(output_dir, base)



def save_boundaries_to_file(boundaries: List[PhaseBoundaryResult], 
                            eta: float, 
                            output_dir: str = "output",
                            zl_params: ZLParams = None,
                            vmit_params: VMITParams = None,
                            eq_mode: str = "beta",
                            Y_C: float = None,
                            Y_L: float = None):
    """Save phase boundaries to .dat file with full onset solution for warm-starting.
    
    For fixed_yc/trapped modes, appends to existing file (or creates new one).
    Y_C or Y_L is included as the first column for these modes.
    
    Args:
        boundaries: List of phase boundary results
        eta: parameter controlling local-global total electric charge neutrality
        output_dir: Directory for output files
        zl_params: ZL model parameters
        vmit_params: vMIT model parameters
        eq_mode: Equilibrium mode ("beta", "fixed_yc", or "trapped")
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = get_boundary_filename(eta, output_dir, zl_params, vmit_params,
                                     eq_mode=eq_mode)
    
    # For fixed_yc/trapped: load existing data if file exists
    existing_data = []
    if eq_mode in ("fixed_yc", "trapped") and os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                existing_data.append(line)
    
    with open(filename, 'w') as f:
        f.write(f"# ZL+vMIT Phase Boundaries\n")
        f.write(f"# eta = {eta:.2f}\n")
        f.write(f"# eq_mode = {eq_mode}\n")
        if zl_params is not None:
            f.write(f"# ZL params: {zl_params.name}\n")
        if vmit_params is not None:
            f.write(f"# vMIT params: B^1/4={vmit_params.B4} MeV, a={vmit_params.a} fm^2\n")
        
        # Header with all columns
        if eq_mode == "fixed_yc":
            f.write("# Columns: Y_C T n_B_onset n_B_offset conv mu_p_H mu_n_H mu_u_Q mu_d_Q mu_s_Q ")
        elif eq_mode == "trapped":
            f.write("# Columns: Y_L T n_B_onset n_B_offset conv mu_p_H mu_n_H mu_u_Q mu_d_Q mu_s_Q ")
        else:  # beta
            f.write("# Columns: T n_B_onset n_B_offset conv mu_p_H mu_n_H mu_u_Q mu_d_Q mu_s_Q ")
        f.write("mu_eL_H mu_eL_Q mu_eG n_p_H n_n_H n_u_Q n_d_Q n_s_Q\n")
        
        # Write existing data first (for other Y_C values)
        comp_value = Y_C if eq_mode == "fixed_yc" else (Y_L if eq_mode == "trapped" else None)
        for line in existing_data:
            parts = line.split()
            if len(parts) > 0:
                existing_comp = float(parts[0])
                # Skip lines with the same composition (will be replaced by new data)
                if comp_value is not None and abs(existing_comp - comp_value) < 1e-6:
                    continue
            f.write(line)
        
        # Write new data
        for b in boundaries:
            conv = 1 if (b.converged_onset and b.converged_offset) else 0
            
            # For fixed_yc/trapped: write composition as first column
            if eq_mode == "fixed_yc" and Y_C is not None:
                f.write(f"{Y_C:8.4f} ")
            elif eq_mode == "trapped" and Y_L is not None:
                f.write(f"{Y_L:8.4f} ")
            
            # Write T, boundaries, convergence, then all onset solution values
            f.write(f"{b.T:10.4f} {b.n_B_onset:12.6e} {b.n_B_offset:12.6e} {conv:2d} ")
            f.write(f"{b.mu_p_H_onset:12.4f} {b.mu_n_H_onset:12.4f} ")
            f.write(f"{b.mu_u_Q_onset:12.4f} {b.mu_d_Q_onset:12.4f} {b.mu_s_Q_onset:12.4f} ")
            f.write(f"{b.mu_eL_H_onset:12.4f} {b.mu_eL_Q_onset:12.4f} {b.mu_eG_onset:12.4f} ")
            f.write(f"{b.n_p_H_onset:12.6e} {b.n_n_H_onset:12.6e} ")
            f.write(f"{b.n_u_Q_onset:12.6e} {b.n_d_Q_onset:12.6e} {b.n_s_Q_onset:12.6e}\n")
    
    return filename


def load_boundaries_from_file(filename: str, eta: float,
                              eq_mode: str = "beta",
                              Y_C: float = None,
                              Y_L: float = None) -> List[PhaseBoundaryResult]:
    """Load phase boundaries from .dat file.
    
    For fixed_yc/trapped modes, filters to only return results matching Y_C/Y_L.
    Supports both old format (no composition column) and new format (with composition).
    
    Args:
        filename: Path to boundary file
        eta: parameter controlling local-global total electric charge neutrality (used to tag results)
        eq_mode: Equilibrium mode ("beta", "fixed_yc", or "trapped")
        Y_C: Charge fraction to filter by (for fixed_yc mode)
        Y_L: Lepton fraction to filter by (for trapped mode)
        
    Returns:
        List of PhaseBoundaryResult for matching composition
    """
    boundaries = []
    comp_value = Y_C if eq_mode == "fixed_yc" else (Y_L if eq_mode == "trapped" else None)
    has_comp_column = eq_mode in ("fixed_yc", "trapped")
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            
            # Determine column offset based on format
            # New format for fixed_yc/trapped: first column is Y_C or Y_L
            # Old format or beta: first column is T
            offset = 0
            row_comp = None
            
            if has_comp_column and len(parts) >= 18:
                # New format with composition column
                row_comp = float(parts[0])
                offset = 1
                
                # Filter by composition if specified
                if comp_value is not None and abs(row_comp - comp_value) > 1e-6:
                    continue
            
            T = float(parts[offset + 0])
            n_B_onset = float(parts[offset + 1])
            n_B_offset = float(parts[offset + 2])
            conv = int(parts[offset + 3]) == 1
            
            # Check if we have the extended format with onset solution (17 columns after offset)
            if len(parts) >= offset + 17:
                boundaries.append(PhaseBoundaryResult(
                    T=T, eta=eta,
                    n_B_onset=n_B_onset, n_B_offset=n_B_offset,
                    converged_onset=conv, converged_offset=conv,
                    error_onset=0.0, error_offset=0.0,
                    # Onset solution values
                    mu_p_H_onset=float(parts[offset + 4]),
                    mu_n_H_onset=float(parts[offset + 5]),
                    mu_u_Q_onset=float(parts[offset + 6]),
                    mu_d_Q_onset=float(parts[offset + 7]),
                    mu_s_Q_onset=float(parts[offset + 8]),
                    mu_eL_H_onset=float(parts[offset + 9]),
                    mu_eL_Q_onset=float(parts[offset + 10]),
                    mu_eG_onset=float(parts[offset + 11]),
                    n_p_H_onset=float(parts[offset + 12]),
                    n_n_H_onset=float(parts[offset + 13]),
                    n_u_Q_onset=float(parts[offset + 14]),
                    n_d_Q_onset=float(parts[offset + 15]),
                    n_s_Q_onset=float(parts[offset + 16])
                ))
            else:
                # Old format without onset solution
                boundaries.append(PhaseBoundaryResult(
                    T=T, eta=eta,
                    n_B_onset=n_B_onset, n_B_offset=n_B_offset,
                    converged_onset=conv, converged_offset=conv,
                    error_onset=0.0, error_offset=0.0
                ))
    return boundaries





def result_to_guess(result: MixedPhaseResult, eta: float, fixed_chi: bool = True) -> np.ndarray:
    """Convert result to guess array based on eta.
    
    Args:
        result: MixedPhaseResult to extract values from
        eta: parameter controlling local-global total electric charge neutrality
        fixed_chi: If True, returns guess for fixed-chi solver (n_B is last unknown)
                   If False, returns guess for fixed-n_B solver (χ is last unknown)
    
    Note: n_e is computed from μ_e, not an unknown.
    """
    last_unknown = result.n_B if fixed_chi else result.chi
    
    if abs(eta) < 1e-10:
        # η=0: 12 unknowns [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns, n_B or χ]
        return np.array([
            result.mu_p_H, result.mu_n_H, result.mu_u_Q, result.mu_d_Q, result.mu_s_Q, result.mu_eG,
            result.n_p_H, result.n_n_H, result.n_u_Q, result.n_d_Q, result.n_s_Q, last_unknown
        ])
    elif abs(eta - 1.0) < 1e-10:
        # η=1: 13 unknowns [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, n_B or χ]
        return np.array([
            result.mu_p_H, result.mu_n_H, result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            result.mu_eL_H, result.mu_eL_Q,
            result.n_p_H, result.n_n_H, result.n_u_Q, result.n_d_Q, result.n_s_Q, last_unknown
        ])
    else:
        # 0<η<1: 14 unknowns [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, n_B or χ]
        return np.array([
            result.mu_p_H, result.mu_n_H, result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            result.mu_eL_H, result.mu_eL_Q, result.mu_eG,
            result.n_p_H, result.n_n_H, result.n_u_Q, result.n_d_Q, result.n_s_Q, last_unknown
        ])


# =============================================================================
# CHI-CONTINUATION FOR OFFSET
# =============================================================================
def solve_offset_by_continuation(T: float, eta: float,
                                  onset_result: 'MixedPhaseResult',
                                  zl_params: ZLParams = None,
                                  vmit_params: VMITParams = None,
                                  chi_step: float = 0.05,
                                  verbose: bool = False,
                                  eq_mode: str = "beta",
                                  Y_C: float = None,
                                  Y_L: float = None) -> 'MixedPhaseResult':
    """
    Solve for offset (χ=1) by gradual continuation from onset (χ=0).
    
    Strategy:
    1. Start from converged onset solution (χ=0)
    2. Step χ from 0 to 1 in small increments
    3. Use each converged solution as initial guess for next step
    
    Args:
        T: Temperature in MeV
        eta: parameter controlling local-global total electric charge neutrality
        onset_result: Converged MixedPhaseResult at χ=0
        chi_step: Step size for χ (default 0.05)
        eq_mode: "beta", "fixed_yc", or "trapped"
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
        
    Returns:
        MixedPhaseResult at χ=1 (or failed result if continuation breaks)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if not onset_result.converged:
        # Can't continue from non-converged onset
        return onset_result
    
    current_guess = result_to_guess(onset_result, eta, eq_mode)
    
    # Step from χ=0 to χ=1
    chi_values = np.arange(chi_step, 1.0 + chi_step/2, chi_step)
    chi_values = np.minimum(chi_values, 1.0)  # Ensure we don't exceed 1.0
    
    last_good_result = onset_result
    
    for chi in chi_values:
        result = solve_fixed_chi(T, chi=chi, eta=eta, 
                                  zl_params=zl_params, vmit_params=vmit_params,
                                  initial_guess=current_guess,
                                  eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
        
        if result.converged and result.error < 1e-4:
            current_guess = result_to_guess(result, eta, eq_mode)
            last_good_result = result
        else:
            # Try with smaller step
            if chi_step > 0.02:
                # Recursively try with smaller step from last good point
                return solve_offset_by_continuation(
                    T, eta, last_good_result, zl_params, vmit_params,
                    chi_step=chi_step/2, verbose=verbose,
                    eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L
                )
            else:
                if verbose:
                    print(f"    χ-continuation failed at χ={chi:.2f}")
                # Return last good result (not at χ=1)
                return MixedPhaseResult(
                    converged=False, error=1.0,
                    n_B=result.n_B, T=T, eta=eta, chi=1.0,
                    mu_p_H=0, mu_n_H=0, n_p_H=0, n_n_H=0, P_H=0,
                    mu_u_Q=0, mu_d_Q=0, mu_s_Q=0, n_u_Q=0, n_d_Q=0, n_s_Q=0, P_Q=0
                )
    
    # Reached χ=1 successfully
    return last_good_result


def find_offset_by_nB_sweep(T: float, eta: float,
                             onset_result: 'MixedPhaseResult',
                             zl_params: ZLParams = None,
                             vmit_params: VMITParams = None,
                             n_B_step: float = 0.01,
                             chi_threshold: float = 0.99,
                             max_n_B: float = 3.0,
                             verbose: bool = False,
                             eq_mode: str = "beta",
                             Y_C: float = None,
                             Y_L: float = None) -> 'MixedPhaseResult':
    """
    Find offset boundary by sweeping n_B from onset and finding where chi reaches threshold.
    
    This approach is consistent with how solve_mixed_phase works in the table generator,
    ensuring the offset boundary matches where chi actually reaches ~1.0 during the sweep.
    
    Strategy:
    1. Start from converged onset solution (χ=0)
    2. Sweep n_B in small increments, solving mixed phase at each point
    3. Stop when χ >= chi_threshold (default 0.99)
    
    Args:
        T: Temperature in MeV
        eta: parameter controlling local-global charge neutrality
        onset_result: Converged MixedPhaseResult at onset (χ=0)
        n_B_step: Step size for n_B sweep (default 0.01 fm^-3)
        chi_threshold: Chi value to consider as offset (default 0.99)
        max_n_B: Maximum n_B to search (default 3.0 fm^-3)
        eq_mode: "beta", "fixed_yc", or "trapped"
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
        
    Returns:
        MixedPhaseResult at the offset (where χ reaches chi_threshold)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    if not onset_result.converged:
        return onset_result
    
    # Start from onset n_B and sweep upward
    n_B_start = onset_result.n_B
    n_B_values = np.arange(n_B_start + n_B_step, max_n_B, n_B_step)
    
    # Initial guess from onset
    current_guess = result_to_guess_mixed(onset_result, eta)
    last_good_result = onset_result
    converged_history = [(n_B_start, current_guess.copy())]
    
    for n_B in n_B_values:
        # Use extrapolation if we have history
        if len(converged_history) >= 2:
            n_B1, guess1 = converged_history[-2]
            n_B2, guess2 = converged_history[-1]
            if abs(n_B2 - n_B1) > 1e-10:
                slope = (guess2 - guess1) / (n_B2 - n_B1)
                guess_to_use = guess2 + slope * (n_B - n_B2)
            else:
                guess_to_use = current_guess
        else:
            guess_to_use = current_guess
        
        result = solve_mixed_phase(
            n_B, T, eta,
            zl_params=zl_params,
            vmit_params=vmit_params,
            initial_guess=guess_to_use,
            eq_mode=eq_mode,
            Y_C=Y_C
        )
        
        if result.converged and result.error < 1e-4:
            current_guess = result_to_guess_mixed(result, eta)
            converged_history.append((n_B, current_guess.copy()))
            if len(converged_history) > 5:
                converged_history.pop(0)
            last_good_result = result
            
            # Check if we've reached the offset threshold
            if result.chi >= chi_threshold:
                if verbose:
                    print(f"    n_B sweep found offset: n_B={n_B:.4f} fm^-3, chi={result.chi:.4f}")
                return result
        else:
            # If non-converged but chi was close to 1, use last good result
            if last_good_result.chi >= 0.95:
                if verbose:
                    print(f"    n_B sweep: solver failed at n_B={n_B:.4f}, using last good chi={last_good_result.chi:.4f}")
                return last_good_result
            
            # Try fallback with previous guess (no extrapolation)
            result_fallback = solve_mixed_phase(
                n_B, T, eta,
                zl_params=zl_params,
                vmit_params=vmit_params,
                initial_guess=current_guess,
                eq_mode=eq_mode,
                Y_C=Y_C
            )
            if result_fallback.converged and result_fallback.error < 1e-4:
                if result_fallback.chi >= chi_threshold:
                    return result_fallback
                current_guess = result_to_guess_mixed(result_fallback, eta)
                last_good_result = result_fallback
    
    # Reached max_n_B without finding offset
    if verbose:
        print(f"    n_B sweep: reached max_n_B={max_n_B} without finding offset, last chi={last_good_result.chi:.4f}")
    return last_good_result


def result_to_guess_mixed(result: 'MixedPhaseResult', eta: float) -> np.ndarray:
    """Convert MixedPhaseResult to guess array for solve_mixed_phase.
    
    solve_mixed_phase expects:
    - eta=0: 11 unknowns [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns]
    - eta=1: 12 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, np, nn, nu, nd, ns]
    - 0<eta<1: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, χ]
    """
    if abs(eta) < 1e-10:
        # eta=0: 11 unknowns [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns]
        return np.array([
            result.mu_p_H, result.mu_n_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            result.mu_eG,
            result.n_p_H, result.n_n_H,
            result.n_u_Q, result.n_d_Q, result.n_s_Q
        ])
    elif abs(eta - 1.0) < 1e-10:
        # eta=1: 12 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, np, nn, nu, nd, ns]
        mu_eH = getattr(result, 'mu_eL_H', result.mu_eG)
        mu_eQ = getattr(result, 'mu_eL_Q', result.mu_eG)
        return np.array([
            result.mu_p_H, result.mu_n_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            mu_eH, mu_eQ,
            result.n_p_H, result.n_n_H,
            result.n_u_Q, result.n_d_Q, result.n_s_Q
        ])
    else:
        # 0<eta<1: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, χ]
        mu_eH = getattr(result, 'mu_eL_H', result.mu_eG)
        mu_eQ = getattr(result, 'mu_eL_Q', result.mu_eG)
        return np.array([
            result.mu_p_H, result.mu_n_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            mu_eH, mu_eQ, result.mu_eG,
            result.n_p_H, result.n_n_H,
            result.n_u_Q, result.n_d_Q, result.n_s_Q,
            result.chi
        ])


def get_default_guess(chi: float, eta: float, n_B_est: float, T: float,
                      zl_params: ZLParams = None, 
                      vmit_params: VMITParams = None,
                      eq_mode: str = "beta",
                      Y_C: float = None,
                      Y_L: float = None,
                      fixed_chi: bool = False) -> np.ndarray:
    """
    Get initial guess for any eta and chi.
    
    For χ=0 (onset): Use pure H solution to get μ's
    For χ=1 (offset): Use pure Q solution to get μ's
    
    Args:
        eq_mode: "beta", "fixed_yc", or "trapped"
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
        fixed_chi: If True, return guess for fixed-chi solver (fewer unknowns)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    B4_scale = vmit_params.B4 / 180.0
    
    # For fixed_yc with small Y_C, use fast analytical guess instead of calling pure phase solver
    # This avoids expensive pure phase solver calls that struggle with small Y_C
    if eq_mode == "fixed_yc" and Y_C is not None and Y_C < 0.1:
        # Quick analytical guess scales for small Y_C
        mu_e_scale = max(Y_C * 3.0, 0.3)
        n_p_guess = max(Y_C * n_B_est, 0.02)
        n_n_guess = max((1 - Y_C) * n_B_est, 0.1)
        
        # Use empirical values for chemical potentials (scaled by B4)
        mu_p_est = 850.0 * B4_scale
        mu_n_est = 1100.0 * B4_scale
        mu_e_est = 250.0 * B4_scale * mu_e_scale
        mu_u_est = 200.0 * B4_scale
        mu_d_est = 450.0 * B4_scale
        mu_s_est = mu_d_est
        
        if abs(eta) < 1e-10:
            return np.array([mu_p_est, mu_n_est, mu_u_est, mu_d_est, mu_s_est, mu_e_est,
                            n_p_guess, n_n_guess, 0.04, 0.8, 0.6, n_B_est])
        elif abs(eta - 1.0) < 1e-10:
            return np.array([mu_p_est, mu_n_est, mu_u_est, mu_d_est, mu_s_est,
                            mu_e_est, 10.0 * mu_e_scale,
                            n_p_guess, n_n_guess, 1.0, 1.1, 0.9, n_p_guess, 1e-5, n_B_est])
        else:
            return np.array([mu_p_est, mu_n_est, mu_u_est, mu_d_est, mu_s_est,
                            mu_e_est, 10.0 * mu_e_scale, mu_e_est,
                            n_p_guess, n_n_guess, 0.5, 1.0, 0.8, n_p_guess, n_p_guess, n_B_est])
    
    # Try to get better guesses from pure phase solutions
    try:
        if chi < 0.5:
            # χ≈0: mostly hadronic phase
            # Select appropriate pure phase solver based on equilibrium mode
            if eq_mode == "trapped":
                H = solve_pure_H_trapped(n_B_est, T, Y_L, zl_params)
            elif eq_mode == "fixed_yc":
                H = solve_pure_H_fixed_yc(n_B_est, T, Y_C, zl_params)
            else:  # beta
                H = solve_pure_H_beta(n_B_est, T, zl_params)
            if not H.converged:
                raise ValueError("Pure H failed")
                
            # Estimate quark μ's from hadronic values
            mu_u_est = (H.mu_p_H - H.mu_eG) / 3.0
            mu_d_est = mu_u_est + H.mu_eG
            mu_s_est = mu_d_est
            
            # Estimate quark densities from n_B (based on Mathematica patterns)
            # At onset, quark densities are: n_u ~ 0.09, n_d ~ 0.9, n_s ~ 0.73 for n_B ~ 0.47
            n_u_est = 0.1 * n_B_est / 0.47
            n_d_est = 0.9 * n_B_est / 0.47
            n_s_est = 0.73 * n_B_est / 0.47
            
            if abs(eta) < 1e-10:
                # η=0: 12 unknowns (same for all eq_modes)
                return np.array([
                    H.mu_p_H, H.mu_n_H, mu_u_est, mu_d_est, mu_s_est, H.mu_eG,
                    H.n_p_H, H.n_n_H, n_u_est, n_d_est, n_s_est, n_B_est
                ])
            elif abs(eta - 1.0) < 1e-10:
                # η=1: 15 unknowns for both beta and fixed_yc
                n_C_Q_est = 0.01
                return np.array([
                    H.mu_p_H, H.mu_n_H, mu_u_est, mu_d_est, mu_s_est,
                    H.mu_eG, 10.0,  # μ_eH, μ_eQ
                    H.n_p_H, H.n_n_H, 0.5, 1.0, 0.5,  # densities
                    H.n_p_H, n_C_Q_est, n_B_est  # n_eH, n_eQ, n_B
                ])
            else:
                # 0 < η < 1: different sizes for beta (17) vs fixed_yc (16)
                n_C_Q_est = 0.01
                if eq_mode == "fixed_yc":
                    if fixed_chi:
                        # Fixed-χ solver: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, nB]
                        return np.array([
                            H.mu_p_H, H.mu_n_H, mu_u_est, mu_d_est, mu_s_est,
                            H.mu_eG, 10.0, H.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            H.n_p_H, H.n_n_H, 0.5, 1.0, 0.5, n_B_est  # np, nn, nu, nd, ns, n_B
                        ])
                    else:
                        # Regular solver: 16 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, neH, neG, nB]
                        return np.array([
                            H.mu_p_H, H.mu_n_H, mu_u_est, mu_d_est, mu_s_est,
                            H.mu_eG, 10.0, H.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            H.n_p_H, H.n_n_H, 0.5, 1.0, 0.5,  # np, nn, nu, nd, ns
                            H.n_p_H, H.n_p_H, n_B_est  # n_eH, n_eG, n_B
                        ])
                else:
                    # Beta equilibrium 0 < η < 1
                    if fixed_chi:
                        # Fixed-χ solver: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, nB]
                        return np.array([
                            H.mu_p_H, H.mu_n_H, mu_u_est, mu_d_est, mu_s_est,
                            H.mu_eG, 10.0, H.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            H.n_p_H, H.n_n_H, 0.5, 1.0, 0.5, n_B_est  # np, nn, nu, nd, ns, n_B
                        ])
                    else:
                        # Regular solver: 17 unknowns (includes n_eH, n_eQ, n_eG)
                        return np.array([
                            H.mu_p_H, H.mu_n_H, mu_u_est, mu_d_est, mu_s_est,
                            H.mu_eG, 10.0, H.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            H.n_p_H, H.n_n_H, 0.5, 1.0, 0.5,  # densities
                            H.n_p_H, n_C_Q_est, H.n_p_H, n_B_est  # n_eH, n_eQ, n_eG, n_B
                        ])
        else:
            # χ≈1: mostly quark phase
            # Select appropriate pure phase solver based on equilibrium mode
            if eq_mode == "trapped":
                Q = solve_pure_Q_trapped(n_B_est, T, Y_L, vmit_params)
            elif eq_mode == "fixed_yc":
                Q = solve_pure_Q_fixed_yc(n_B_est, T, Y_C, vmit_params)
            else:  # beta
                Q = solve_pure_Q_beta(n_B_est, T, vmit_params)
            if not Q.converged:
                raise ValueError("Pure Q failed")
            
            mu_p_est = 2 * Q.mu_u_Q + Q.mu_d_Q
            mu_n_est = Q.mu_u_Q + 2 * Q.mu_d_Q
            n_C_Q = (2*Q.n_u_Q - Q.n_d_Q - Q.n_s_Q) / 3.0
            
            if abs(eta) < 1e-10:
                # η=0: 12 unknowns (same for all eq_modes)
                return np.array([
                    mu_p_est, mu_n_est, Q.mu_u_Q, Q.mu_d_Q, Q.mu_s_Q, Q.mu_eG,
                    0.01, 0.3, Q.n_u_Q, Q.n_d_Q, Q.n_s_Q, n_B_est
                ])
            elif abs(eta - 1.0) < 1e-10:
                # η=1: 15 unknowns for both beta and fixed_yc
                return np.array([
                    mu_p_est, mu_n_est, Q.mu_u_Q, Q.mu_d_Q, Q.mu_s_Q,
                    200.0, Q.mu_eG,  # μ_eH, μ_eQ
                    0.01, 0.3, Q.n_u_Q, Q.n_d_Q, Q.n_s_Q,  # densities
                    0.01, max(n_C_Q, 1e-6), n_B_est  # n_eH, n_eQ, n_B
                ])
            else:
                # 0 < η < 1: different sizes for beta (17) vs fixed_yc (16)
                if eq_mode == "fixed_yc":
                    if fixed_chi:
                        # Fixed-χ solver: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, nB]
                        return np.array([
                            mu_p_est, mu_n_est, Q.mu_u_Q, Q.mu_d_Q, Q.mu_s_Q,
                            200.0, Q.mu_eG, Q.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            0.01, 0.3, Q.n_u_Q, Q.n_d_Q, Q.n_s_Q, n_B_est
                        ])
                    else:
                        # Regular solver: 16 unknowns
                        return np.array([
                            mu_p_est, mu_n_est, Q.mu_u_Q, Q.mu_d_Q, Q.mu_s_Q,
                            200.0, Q.mu_eG, Q.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            0.01, 0.3, Q.n_u_Q, Q.n_d_Q, Q.n_s_Q,  # densities
                            0.01, max(n_C_Q, 1e-6), n_B_est  # n_eH, n_eG, n_B
                        ])
                else:
                    # Beta equilibrium
                    if fixed_chi:
                        # Fixed-χ solver: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, nB]
                        return np.array([
                            mu_p_est, mu_n_est, Q.mu_u_Q, Q.mu_d_Q, Q.mu_s_Q,
                            200.0, Q.mu_eG, Q.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            0.01, 0.3, Q.n_u_Q, Q.n_d_Q, Q.n_s_Q, n_B_est
                        ])
                    else:
                        # Regular solver: 17 unknowns (includes n_eH, n_eQ, n_eG)
                        return np.array([
                            mu_p_est, mu_n_est, Q.mu_u_Q, Q.mu_d_Q, Q.mu_s_Q,
                            200.0, Q.mu_eG, Q.mu_eG,  # μ_eH, μ_eQ, μ_eG
                            0.01, 0.3, Q.n_u_Q, Q.n_d_Q, Q.n_s_Q,  # densities
                            0.01, max(n_C_Q, 1e-6), max(n_C_Q, 1e-6), n_B_est  # n_eH, n_eQ, n_eG, n_B
                        ])
    except:
        pass
    
    # Fallback to hardcoded values - scaled by B4
    # Lower B4 → lower chemical potentials at transition
    B4_scale = vmit_params.B4 / 180.0  # 0.917 for B4=165, 1.0 for B4=180
    
    # For fixed_yc mode, scale μ_e and n_p based on Y_C
    # Small Y_C means fewer protons and lower electron chemical potential
    if eq_mode == "fixed_yc" and Y_C is not None:
        # Scale μ_e with Y_C (small Y_C → small μ_e)
        mu_e_scale = max(Y_C * 3.0, 0.3)  # Scale down for small Y_C, min 0.3
        # n_p should be proportional to Y_C
        n_p_guess = max(Y_C * n_B_est, 0.02)
        n_n_guess = max((1 - Y_C) * n_B_est, 0.1)
    else:
        mu_e_scale = 1.0
        n_p_guess = 0.01
        n_n_guess = 0.35
    
    if abs(eta) < 1e-10:
        # 12 unknowns for all eq_modes
        # [μp, μn, μu, μd, μs, μe, np, nn, nu, nd, ns, n_B]
        return np.array([
            850.0 * B4_scale, 1100.0 * B4_scale, 
            200.0 * B4_scale, 450.0 * B4_scale, 450.0 * B4_scale, 
            250.0 * B4_scale * mu_e_scale,  # μe scaled for small Y_C
            n_p_guess, n_n_guess, 0.04, 0.8, 0.6, n_B_est
        ])
    elif abs(eta - 1.0) < 1e-10:
        # 15 unknowns for both beta and fixed_yc
        # [μp, μn, μu, μd, μs, μeH, μeQ, np, nn, nu, nd, ns, neH, neQ, n_B]
        return np.array([
            1334.0 * B4_scale, 1714.5 * B4_scale, 
            563.1 * B4_scale, 575.7 * B4_scale, 575.7 * B4_scale, 
            380.5 * B4_scale * mu_e_scale, 12.6 * B4_scale * mu_e_scale,
            n_p_guess, n_n_guess, 1.084, 1.181, 0.988, n_p_guess, 8.7e-6, n_B_est
        ])
    else:
        if eq_mode == "fixed_yc":
            if fixed_chi:
                # Fixed-χ solver: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, nB]
                return np.array([
                    1389.9 * B4_scale, 1619.1 * B4_scale, 
                    447.5 * B4_scale, 585.8 * B4_scale, 585.8 * B4_scale, 
                    418.7 * B4_scale * mu_e_scale, 15.0 * B4_scale * mu_e_scale, 418.8 * B4_scale * mu_e_scale,
                    n_p_guess, n_n_guess, 0.474, 1.354, 1.151, n_B_est
                ])
            else:
                # Regular solver: 16 unknowns
                return np.array([
                    1389.9 * B4_scale, 1619.1 * B4_scale, 
                    447.5 * B4_scale, 585.8 * B4_scale, 585.8 * B4_scale, 
                    418.7 * B4_scale * mu_e_scale, 15.0 * B4_scale * mu_e_scale, 418.8 * B4_scale * mu_e_scale,
                    n_p_guess, n_n_guess, 0.474, 1.354, 1.151, n_p_guess, n_p_guess, n_B_est
                ])
        else:
            # Beta equilibrium
            if fixed_chi:
                # Fixed-χ solver: 14 unknowns [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, nB]
                return np.array([
                    1389.9 * B4_scale, 1619.1 * B4_scale, 
                    447.5 * B4_scale, 585.8 * B4_scale, 585.8 * B4_scale, 
                    418.7 * B4_scale, 15.0 * B4_scale, 418.8 * B4_scale,
                    0.323, 0.527, 0.474, 1.354, 1.151, n_B_est
                ])
            else:
                # Regular solver: 17 unknowns
                return np.array([
                    1389.9 * B4_scale, 1619.1 * B4_scale, 
                    447.5 * B4_scale, 585.8 * B4_scale, 585.8 * B4_scale, 
                    418.7 * B4_scale, -490.7 * B4_scale, 418.8 * B4_scale,
                    0.323, 0.527, 0.474, 1.354, 1.151, 0.323, -0.519, 0.323, n_B_est
                ])


def extrapolate_nB_boundaries(history: list, T_target: float, default: float) -> float:
    """
    Extrapolate n_B to target T using converged history.
    
    Uses linear extrapolation from last 2 points, otherwise returns default.
    """
    if len(history) < 2:
        return history[-1][1] if history else default
    
    # Linear extrapolation from last 2 points
    T1, nB1 = history[-2]
    T2, nB2 = history[-1]
    if abs(T2 - T1) > 1e-10:
        slope = (nB2 - nB1) / (T2 - T1)
        n_B_extrap = nB2 + slope * (T_target - T2)
        return n_B_extrap
    
    return default


def extrapolate_guess_boundaries(history: list, T_target: float, default_guess: np.ndarray) -> np.ndarray:
    """
    Extrapolate all unknowns to target T using converged history.
    
    Each element of the guess vector is extrapolated independently using linear
    extrapolation from the last 2 converged points.
    
    Args:
        history: List of (T, guess_vector) tuples from converged solutions
        T_target: Target temperature to extrapolate to
        default_guess: Fallback guess if history is insufficient
        
    Returns:
        Extrapolated guess vector for T_target
    """
    if len(history) == 1:
        # Not enough history for extrapolation, use last converged or default
        return history[-1][1]
    
    # Linear extrapolation from last 2 points for each element
    T1, guess1 = history[-2]
    T2, guess2 = history[-1]
    
    # Extrapolate each element of the guess vector
    dT = T_target - T2
    slope = (guess2 - guess1) / (T2 - T1)
    guess_extrap = guess2 + slope * dT
    
    return guess_extrap


# =============================================================================
# PURE PHASES TABLE INTERPOLATION FOR MIXED PHASE BOUNDARIES GUESSES
# =============================================================================
def build_table_interpolators(table: dict, phase: str = "H") -> dict:
    """
    Build scipy RegularGridInterpolator objects for all quantities in a table.
    
    Args:
        table: Dict of {(n_B, T): result} from pre-computed pure phase table
        phase: "H" (hadronic) or "Q" (quark) - determines which attributes to extract
        
    Returns:
        Dict of {quantity_name: RegularGridInterpolator} for interpolating at any (n_B, T)
    """
    if table is None or len(table) == 0:
        return None
    
    # Extract unique n_B and T values from table keys
    n_B_vals = sorted(set(k[0] for k in table.keys()))
    T_vals = sorted(set(k[1] for k in table.keys()))
    
    # Define attributes to interpolate based on phase type
    if phase == "H":
        attrs = ['mu_p', 'mu_n', 'n_p', 'n_n', 'mu_e', 'n_e', 'e_total', 's_total']
    else:  # Q
        attrs = ['mu_u', 'mu_d', 'mu_s', 'n_u', 'n_d', 'n_s', 'mu_e', 'n_e', 'e_total', 's_total']
    
    # Build 2D arrays for each attribute
    interpolators = {}
    for attr in attrs:
        # Create 2D array: shape (len(n_B_vals), len(T_vals))
        data = np.full((len(n_B_vals), len(T_vals)), np.nan)
        
        for i, n_B in enumerate(n_B_vals):
            for j, T in enumerate(T_vals):
                result = table.get((n_B, T))
                if result is not None and result.converged and hasattr(result, attr):
                    data[i, j] = getattr(result, attr)
        
        # Create interpolator (bounds_error=False allows extrapolation with fill_value)
        interpolators[attr] = RegularGridInterpolator(
            (np.array(n_B_vals), np.array(T_vals)), data,
            method='linear', bounds_error=False, fill_value=None
        )
    
    # Store grid info for reference
    interpolators['_n_B_vals'] = np.array(n_B_vals)
    interpolators['_T_vals'] = np.array(T_vals)
    
    return interpolators


def find_n_B_crossing(H_interp: dict, Q_interp: dict, T: float, 
                       n_B_search: np.ndarray = None) -> float:
    """
    Find n_B where free energy density f_H = f_Q at temperature T.
    
    Free energy density: f = e - T*s
    
    Args:
        H_interp: Interpolators from build_table_interpolators for H phase
        Q_interp: Interpolators from build_table_interpolators for Q phase
        T: Temperature in MeV
        n_B_search: Array of n_B values to search (default: use grid from tables)
        
    Returns:
        n_B where f_H = f_Q, or None if no crossing found
    """
    if H_interp is None or Q_interp is None:
        print("Error: H_interp or Q_interp is None")#####-#####
        return None
    
    # Default search grid: combine both table grids
    if n_B_search is None:
        n_B_H = H_interp['_n_B_vals']
        n_B_Q = Q_interp['_n_B_vals']
        n_B_search = np.unique(np.concatenate([n_B_H, n_B_Q]))
    
    # Compute f_H - f_Q at each n_B
    diffs = []
    valid_n_B = []
    for n_B in n_B_search:
        try:
            e_H = float(H_interp['e_total']((n_B, T)))
            s_H = float(H_interp['s_total']((n_B, T)))
            e_Q = float(Q_interp['e_total']((n_B, T)))
            s_Q = float(Q_interp['s_total']((n_B, T)))
            
            f_H = e_H - T * s_H
            f_Q = e_Q - T * s_Q
            diff = f_H - f_Q
            if not np.isnan(diff):
                diffs.append(diff)
                valid_n_B.append(n_B)
        except:
            continue
    
    # Find sign change (crossing point)
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i+1] < 0:
            # Use root to find exact crossing
            n1, n2 = valid_n_B[i], valid_n_B[i+1]
            d1, d2 = diffs[i], diffs[i+1]
            
            n_B_guess = (n1 + n2) / 2
            
            def f_diff(n_B):
                e_H = float(H_interp['e_total']((n_B, T)))
                s_H = float(H_interp['s_total']((n_B, T)))
                e_Q = float(Q_interp['e_total']((n_B, T)))
                s_Q = float(Q_interp['s_total']((n_B, T)))
                return (e_H - T * s_H) - (e_Q - T * s_Q)
            
            sol = root(f_diff, n_B_guess, method='lm')
            if sol.success:
                return float(sol.x[0])
            else:
                return n_B_guess  # Fallback to linear estimate
    
    return None


def estimate_boundary_n_B(H_interp: dict, Q_interp: dict, T: float, 
                           eta: float, boundary: str = "onset") -> float:
    """
    Estimate n_B for onset or offset boundary based on free energy crossing.
    
    Args:
        H_interp: Interpolators for H phase
        Q_interp: Interpolators for Q phase
        T: Temperature
        eta: parameter controlling local-global total electric charge neutrality (affects width of mixed phase)
        boundary: "onset" (χ=0) or "offset" (χ=1)
        
    Returns:
        Estimated n_B for the boundary
    """
    n_B_crossing = find_n_B_crossing(H_interp, Q_interp, T)
    
    if n_B_crossing is None:
        return None
    
    # Width depends on η: measured from data
    # η=0: mixed phase width ~1.3 fm⁻³ (onset ~0.48, offset ~1.78)
    # η=1: mixed phase width ~0.25 fm⁻³ (onset ~0.83, offset ~1.08)
    # At crossing f_H=f_Q, onset is slightly below, offset is slightly above
    # Offset is further from crossing than onset for η=0
    
    if boundary == "onset":
        # Onset: slightly below crossing, similar for all η
        delta_onset = 0.15 + 0.10 * (1.0 - eta)
        return max(0.1, n_B_crossing - delta_onset)
    else:
        # Offset: much further above crossing for low η
        delta_offset = 0.15 + 0.60 * (1.0 - eta)  # 0.15 for η=1, 0.75 for η=0
        return n_B_crossing + delta_offset


def build_initial_guess_boundaries_given_nB(interp: dict, n_B: float, T: float,
                                            eta: float, eq_mode: str,
                                            phase: str = "H", Y_C: float = None) -> np.ndarray:
    """
    Build initial guess array for mixed-phase solver using interpolated pure phase values.
    
    Uses exact mixed phase equilibrium relations to compute quark chemical potentials
    from hadronic values (for onset) or vice versa (for offset).
    
    Mapping particle chemical potentials into strong charges:
        - μ_B_H = μ_n (hadronic baryon chemical potential)
        - μ_B_Q = 2μ_d + μ_u (quark baryon chemical potential)
        - μ_C_H = μ_p - μ_n (hadronic charge chemical potential)
        - μ_C_Q = μ_u - μ_d (quark charge chemical potential)
        - μ_S_Q = μ_s - μ_d (quark strangeness chemical potential)
    
    For beta equilibrium (general η):
        μ_B_H = μ_B_Q                                    (baryon chemical eq)
        μ_C_H + η*μ_eL_H + (1-η)*μ_eG = 0               (beta-eq in H sector)
        μ_C_Q + η*μ_eL_Q + (1-η)*μ_eG = 0               (beta-eq in Q sector)
        μ_S_Q = 0                                        (strangeness eq)
    For fixed_yc (general η):
        μ_B_H = μ_B_Q                                    (baryon chemical eq)
        μ_C_H + η*μ_eL_H = μ_C_Q + η*μ_eL_Q           (charge chemical eq)
        μ_S_Q = 0                                        (strangeness eq)
    
    Args:
        interp: Interpolators from build_table_interpolators
        n_B: Target baryon density
        T: Temperature
        eta: parameter controlling local-global total electric charge neutrality
        eq_mode: "beta", "fixed_yc", or "trapped"
        phase: "H" (use for onset) or "Q" (use for offset)
        Y_C: Charge fraction (for fixed_yc mode)
        
    Returns:
        numpy array of initial guess values with correct size for given eta/eq_mode
    """
    if interp is None:
        return None
    
    point = (n_B, T)
    
    if phase == "H":
        # =================================================================
        # ONSET (χ=0): H values from pure H table, Q values from equilibrium
        # At onset the mixed phase H sector ≈ pure H at same (n_B, T)
        # =================================================================
        
        # 1. Get H phase quantities from pure H table
        mu_p = float(interp['mu_p'](point))
        mu_n = float(interp['mu_n'](point))
        n_p = float(interp['n_p'](point))
        n_n = float(interp['n_n'](point))
        mu_e_pure = float(interp['mu_e'](point))  # μ_e from pure H table
        
        # 2. Conserved charge chemical potentials in H
        mu_B_H = mu_n                    # μ_B_H = μ_n
        mu_C_H = mu_p - mu_n             # μ_C_H = μ_p - μ_n
        
        # 3. Electron guesses: use pure H values, assume local ≈ global for initial guess
        mu_eL_H = mu_e_pure
        mu_eG = mu_e_pure
        
        # 4. Local neutrality: n_eL_H = n_C_H = n_p
        n_eL_H = n_p
        n_eG = n_p  # At χ=0, global = H sector
        
        # 5. Quark chemical potentials from equilibrium relations
        # μ_B_H = μ_B_Q → μ_n = 2μ_d + μ_u
        # μ_S_Q = 0 → μ_s = μ_d
        # For beta eq: use μ_C_Q + μ_e = 0 → μ_u - μ_d = -μ_e
        # For fixed_yc: use μ_C_H = μ_C_Q → μ_u - μ_d = μ_C_H
        # General formula: μ_u = μ_B_H/3 + 2μ_C_Q/3, μ_d = μ_B_H/3 - μ_C_Q/3
        
        if eq_mode == "beta" or eq_mode == "trapped":
            # Beta eq: μ_C_Q = -μ_e
            mu_C_Q = -mu_eG
        else:  # fixed_yc
            # Fixed Y_C: μ_C_H = μ_C_Q
            mu_C_Q = mu_C_H
        
        mu_u = mu_B_H / 3.0 + 2.0 * mu_C_Q / 3.0
        mu_d = mu_B_H / 3.0 - mu_C_Q / 3.0
        mu_s = mu_d  # μ_S_Q = 0 → μ_s = μ_d
        
        # 6. Quark electron (guess μ_eL_Q ≈ μ_eG for Gibbs approx)
        mu_eL_Q = mu_eG
        
        # 7. Quark densities: At onset (χ=0), n_B_H ≈ n_B, but we still need Q densities
        # From data analysis (onset boundary results):
        # η=0: n_u/n_B ≈ 0.18, n_d/n_B ≈ 1.95, n_s/n_B ≈ 1.58
        # η=1: n_u/n_B ≈ 1.30, n_d/n_B ≈ 1.42, n_s/n_B ≈ 1.18
        # Interpolate between these based on η
        n_u_ratio = 0.18 + 1.12 * eta  # 0.18 at η=0, 1.30 at η=1
        n_d_ratio = 1.95 - 0.53 * eta  # 1.95 at η=0, 1.42 at η=1  
        n_s_ratio = 1.58 - 0.40 * eta  # 1.58 at η=0, 1.18 at η=1
        n_u = n_u_ratio * n_B
        n_d = n_d_ratio * n_B
        n_s = n_s_ratio * n_B
        
        # 8. Quark phase electron density (local neutrality gives n_eL_Q = n_C_Q)
        # n_C_Q = (2n_u - n_d - n_s)/3
        n_eL_Q = max(0.01, (2*n_u - n_d - n_s) / 3.0)
        
    else:  # phase == "Q"
        # =================================================================
        # OFFSET (χ≈1): Q values from pure Q table, H values from equilibrium
        # At offset the mixed phase Q sector ≈ pure Q at same (n_B, T)
        # =================================================================
        
        # 1. Get Q phase quantities from pure Q table
        mu_u = float(interp['mu_u'](point))
        mu_d = float(interp['mu_d'](point))
        mu_s = float(interp['mu_s'](point))
        n_u = float(interp['n_u'](point))
        n_d = float(interp['n_d'](point))
        n_s = float(interp['n_s'](point))
        mu_e_pure = float(interp['mu_e'](point))  # μ_e from pure Q table
        
        # 2. Conserved charge chemical potentials in Q
        mu_B_Q = 2.0 * mu_d + mu_u    # μ_B_Q = 2μ_d + μ_u
        mu_C_Q = mu_u - mu_d          # μ_C_Q = μ_u - μ_d
        
        # 3. Electron guesses: use pure Q values
        mu_eL_Q = mu_e_pure
        mu_eG = mu_e_pure
        
        # 4. Local neutrality: n_eL_Q = n_C_Q = (2n_u - n_d - n_s)/3
        n_C_Q = (2.0 * n_u - n_d - n_s) / 3.0
        n_eL_Q = n_C_Q
        n_eG = n_C_Q  # At χ≈1, global ≈ Q sector
        
        # 5. Hadronic chemical potentials from equilibrium
        # μ_B_H = μ_B_Q → μ_n = μ_B_Q
        mu_n = mu_B_Q
        
        if eq_mode == "beta" or eq_mode == "trapped":
            # Beta eq: μ_C_H + η*μ_eL_H + (1-η)*μ_eG = 0
            # At offset (χ≈1), the H phase has μ_n >> μ_p for neutron-rich matter
            # From actual converged offset data:
            #   η=0.00: μ_p≈2082, μ_n≈2093, μeG≈11 (μeL_H=0)
            #   η=0.10: μ_p≈1803, μ_n≈1860, μeL_H≈473, μeG≈11
            #   η=0.30: μ_p≈1755, μ_n≈1906, μeL_H≈475, μeG≈11
            #   η=0.60: μ_p≈1481, μ_n≈1742, μeL_H≈427, μeG≈12
            #   η=1.00: μ_p≈1258, μ_n≈1610, μeL_H≈353, μeG≈0 (μeQ≈13)
            # Pattern: μ_p/μ_n ratio decreases with η
            #   η=0: μ_p/μ_n ≈ 0.995
            #   η=0.1: μ_p/μ_n ≈ 0.97
            #   η=0.3: μ_p/μ_n ≈ 0.92
            #   η=0.6: μ_p/μ_n ≈ 0.85  
            #   η=1: μ_p/μ_n ≈ 0.78
            mu_p_ratio = 1.0 - 0.22 * eta  # 1.0 at η=0, 0.78 at η=1
            mu_p = mu_p_ratio * mu_n
            
            # For μeL_H: at η>0, it's large (~400-500 MeV) to enforce local neutrality
            # μeL_H ≈ 0.25-0.30 * μ_n for intermediate η, but ~0 for η=0 and η=1
            # Peak around η≈0.3
            mu_eL_H_factor = 4.0 * eta * (1.0 - eta)  # Peaks at η=0.5
            mu_eL_H = max(mu_eG, mu_eL_H_factor * 0.25 * mu_n)
        else:  # fixed_yc
            # Fixed Y_C: μ_C_H = μ_C_Q
            mu_p = mu_n + mu_C_Q
            mu_eL_H = mu_eG
        
        # 7. Hadronic densities: At offset (χ≈1), n_B_H is NOT equal to system n_B
        # At offset, the H phase is at high chemical potential equilibrium with Q
        # From actual converged offset data:
        #   η=0.00: Y_p≈0.48, n_B_H/n_B≈0.76
        #   η=0.10: Y_p≈0.49, n_B_H/n_B≈0.80  
        #   η=0.30: Y_p≈0.43, n_B_H/n_B≈0.78
        #   η=0.60: Y_p≈0.40, n_B_H/n_B≈0.80
        #   η=1.00: Y_p≈0.28, n_B_H/n_B≈0.78
        # Interpolate based on η
        Y_p_ratio = 0.48 - 0.20 * eta  # 0.48 at η=0, 0.28 at η=1
        n_B_H_ratio = 0.78  # H phase n_B is ~78% of system n_B at offset
        n_B_H_est = n_B_H_ratio * n_B
        n_p = max(0.01, Y_p_ratio * n_B_H_est)
        n_n = max(0.01, (1.0 - Y_p_ratio) * n_B_H_est)
        
        # 8. Hadronic electron density (local neutrality: n_eL_H = n_C_H = n_p)
        n_eL_H = max(0.001, n_p)
    
    # =================================================================
    # Build guess array based on eta and eq_mode
    # Note: n_e is computed from μ_e, not an unknown
    # Note: χ is replaced by n_B for boundary finding (we solve for n_B at χ=0 or χ=1)
    # =================================================================
    if abs(eta) < 1e-10:
        # η=0 (Gibbs): 12 unknowns [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns, n_B]
        return np.array([mu_p, mu_n, mu_u, mu_d, mu_s, mu_eG,
                        n_p, n_n, n_u, n_d, n_s, n_B])
    
    elif abs(eta - 1.0) < 1e-10:
        # η=1 (Maxwell): 13 unknowns [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, n_B]
        return np.array([mu_p, mu_n, mu_u, mu_d, mu_s,
                        mu_eL_H, mu_eL_Q,
                        n_p, n_n, n_u, n_d, n_s, n_B])
    
    else:
        # 0 < η < 1: 14 unknowns [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, n_B]
        return np.array([mu_p, mu_n, mu_u, mu_d, mu_s,
                        mu_eL_H, mu_eL_Q, mu_eG,
                        n_p, n_n, n_u, n_d, n_s, n_B])


# =============================================================================
# FIND WORKING START POINT
# =============================================================================
def find_working_start(T_values: np.ndarray, eta: float, chi: float,
                        H_interp: dict, Q_interp: dict,
                        zl_params: ZLParams, vmit_params: VMITParams,
                        eq_mode: str = "beta", Y_C: float = None, Y_L: float = None,
                        T_start_default: float = 40.0, verbose: bool = False):
    """
    Find a working starting point (T, n_B) for boundary marching.
    
    Strategy:
    1. Start from T_start_default (default 40 MeV)
    2. Estimate n_B from free energy crossing at that T
    3. Try to solve; if fails, search other T values
    
    Args:
        T_values: Array of temperatures to search
        eta: parameter controlling local-global total electric charge neutrality
        chi: Volume fraction (0 for onset, 1 for offset)
        H_interp: Interpolators for H phase
        Q_interp: Interpolators for Q phase
        zl_params, vmit_params: Model parameters
        eq_mode: Equilibrium mode
        Y_C, Y_L: Composition parameters
        T_start_default: Starting temperature to try first (default 40 MeV)
        verbose: Print progress
        
    Returns:
        (T_working, n_B_est, guess, result, idx_start) or (None, None, None, None, -1) if failed
    """
    T_sorted = np.sort(T_values)
    
    # Find index closest to T_start_default
    idx_start = np.argmin(np.abs(T_sorted - T_start_default))
    
    boundary = "onset" if chi < 0.5 else "offset"
    phase = "H" if chi < 0.5 else "Q"
    interp = H_interp if chi < 0.5 else Q_interp
    
    # Try starting from T_start_default, then search if needed
    search_order = [idx_start]
    # Add nearby indices in alternating order (±1, ±2, ±3, ...)
    for offset in range(1, len(T_sorted)):
        if idx_start + offset < len(T_sorted):
            search_order.append(idx_start + offset)
        if idx_start - offset >= 0:
            search_order.append(idx_start - offset)
    
    for idx in search_order:
        T = T_sorted[idx]
        
        # Estimate n_B from free energy crossing
        n_B_est = estimate_boundary_n_B(H_interp, Q_interp, T, eta, boundary=boundary)
        
        # Build initial guess from interpolators
        guess = build_initial_guess_boundaries_given_nB(
            interp, n_B_est, T, eta, eq_mode, phase=phase, Y_C=Y_C
        )
        
        # Debug: show initial guess
        if verbose and guess is not None:
            if abs(eta) < 1e-10:
                # η=0: 12 unknowns [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns, n_B]
                print(f"    Trying T={T:.1f} n_B_est={n_B_est:.4f}: guess μp={guess[0]:.1f} μn={guess[1]:.1f} μu={guess[2]:.1f} μd={guess[3]:.1f} μs={guess[4]:.1f} μeG={guess[5]:.1f} np={guess[6]:.4f} nn={guess[7]:.4f}")
            elif abs(eta - 1.0) < 1e-10:
                # η=1: 13 unknowns
                print(f"    Trying T={T:.1f} n_B_est={n_B_est:.4f}: guess μp={guess[0]:.1f} μn={guess[1]:.1f} μu={guess[2]:.1f} μd={guess[3]:.1f} μeH={guess[5]:.1f} μeQ={guess[6]:.1f}")
            else:
                # 0<η<1: 14 unknowns
                print(f"    Trying T={T:.1f} n_B_est={n_B_est:.4f}: guess μp={guess[0]:.1f} μn={guess[1]:.1f} μu={guess[2]:.1f} μd={guess[3]:.1f} μeH={guess[5]:.1f} μeQ={guess[6]:.1f} μeG={guess[7]:.1f}")
        
        # Try to solve
        result = solve_fixed_chi(T, chi=chi, eta=eta, zl_params=zl_params,
                                  vmit_params=vmit_params, initial_guess=guess,
                                  eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
        if result.converged and result.error < 1e-4:
            if verbose:
                print(f"  Found working start: T={T:.1f} MeV, n_B_est={n_B_est:.4f}, n_B={result.n_B:.4f}")
            return T, n_B_est, guess, result, idx
    
    # No working point found
    if verbose:
        print(f"  Warning: No working start found for χ={chi}")
    return None, None, None, None, -1


# =============================================================================
# COMPUTE PHASE BOUNDARIES
# =============================================================================
def find_boundaries(T_values: np.ndarray, eta: float,
                                   T_start: float = None,
                                   n_B_onset_init: float = None,
                                   n_B_offset_init: float = None,
                                   zl_params: ZLParams = None,
                                   vmit_params: VMITParams = None,
                                   verbose: bool = True,
                                   debug: bool = False,
                                   eq_mode: str = "beta",
                                   Y_C: float = None,
                                   Y_L: float = None,
                                   H_table_lookup: dict = None,
                                   Q_table_lookup: dict = None) -> List[PhaseBoundaryResult]:
    """
    Find phase boundaries for given eta using bidirectional T-marching.
    
    Strategy: Start from T_start (where pure phase guesses work well),
    then march both up and down in T using warm-start.
    
    Args:
        eq_mode: "beta", "fixed_yc", or "trapped"
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
        H_table_lookup: Dict of {(n_B, T): ZLEOSResult} from pre-computed pure H table
                       Used to get better initial guesses (μ_p, μ_n) for boundary finding
        Q_table_lookup: Dict of {(n_B, T): VMITEOSResult} from pre-computed pure Q table
                       Used to get better initial guesses (μ_u, μ_d, μ_s) for boundary finding
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Build interpolators once (reused for all T values)
    # H_interp/Q_interp are needed for initial guess generation in all modes
    H_interp = build_table_interpolators(H_table_lookup, phase="H") if H_table_lookup else None
    Q_interp = build_table_interpolators(Q_table_lookup, phase="Q") if Q_table_lookup else None
    
    T_sorted = np.sort(T_values)
    
    # Store full result objects to enable warm-starting in table generation
    onset_results = {}   # T -> MixedPhaseResult
    offset_results = {}  # T -> MixedPhaseResult
    
    # Default fallback values for n_B estimates
    T_start_default = T_start if T_start is not None else 40.0
    
    # --- Find onset (χ=0) ---
    if verbose:
        print(f"\n--- Onset (χ=0, η={eta:.2f}) ---")
    
    # Find working start point for onset
    T_0, n_B_onset_init, guess, result, idx_start = find_working_start(
        T_values, eta, chi=0.0, H_interp=H_interp, Q_interp=Q_interp,
        zl_params=zl_params, vmit_params=vmit_params,
        eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L,
        T_start_default=T_start_default, verbose=verbose
    )
    
    # If no working start found, all T values failed
    if T_0 is None:
        if verbose:
            print(f"  ✗ No working start found for onset")
        # Return empty results
        return []
    
    onset_results[T_0] = result  # Store full result for warm-starting
    converged = result.converged and result.error < 1e-4
    if verbose:
        status = "✓" if converged else "✗"
        print(f"  {status} T={T_0:6.2f} n_B={result.n_B:.4f} μp={result.mu_p_H:.1f} μn={result.mu_n_H:.1f} μu={result.mu_u_Q:.1f} μd={result.mu_d_Q:.1f} μs={result.mu_s_Q:.1f} μeH={result.mu_eL_H:.1f} μeQ={result.mu_eL_Q:.1f} μeG={result.mu_eG:.1f} np={result.n_p_H:.4f} nn={result.n_n_H:.4f} nu={result.n_u_Q:.4f} nd={result.n_d_Q:.4f} ns={result.n_s_Q:.4f} (err={result.error:.1e})")
    
    current_guess = result_to_guess(result, eta, eq_mode) if converged else guess
    # Store full guess vectors in history for better extrapolation
    converged_onset_history = [(T_0, current_guess.copy())] if converged else []
    
    # March upward
    guess_up = current_guess.copy()
    for i in range(idx_start + 1, len(T_sorted)):
        T = T_sorted[i]
        
        # Track all results to pick best if none converge
        all_results = []
        
        # Strategy 1: Use previous converged solution directly (most stable)
        if len(converged_onset_history) >= 1:
            result = solve_fixed_chi(T, chi=0.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_up.copy(),
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        else:
            converged = False
        
        # Strategy 2: Extrapolate ALL unknowns from history (if previous didn't work)
        if not converged and len(converged_onset_history) >= 2:
            guess_extrap = extrapolate_guess_boundaries(converged_onset_history, T, guess_up)
            result = solve_fixed_chi(T, chi=0.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_extrap,
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        
        # Strategy 3: Use n_B from free energy crossing (n_B(f_H=f_Q) - delta)
        if not converged:
            n_B_crossing_est = estimate_boundary_n_B(H_interp, Q_interp, T, eta, boundary="onset")
            if n_B_crossing_est is None:
                n_B_crossing_est = n_B_onset_init
            guess_crossing = build_initial_guess_boundaries_given_nB(
                H_interp, n_B_crossing_est, T, eta, eq_mode, phase="H", Y_C=Y_C
            )
            if guess_crossing is not None:
                result = solve_fixed_chi(T, chi=0.0, eta=eta, zl_params=zl_params,
                                          vmit_params=vmit_params, initial_guess=guess_crossing,
                                          eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
                all_results.append(result)
                converged = result.converged and result.error < 1e-4
        
        # If none converged, pick result with lowest error
        best_guess = None
        if not converged and all_results:
            result = min(all_results, key=lambda r: r.error)
        
        # Debug: print last successful guess and result
        if debug:
            # Get the guess that was used for the best result
            if len(converged_onset_history) >= 2:
                best_guess = extrapolate_guess_boundaries(converged_onset_history, T, guess_up)
            elif len(converged_onset_history) >= 1:
                best_guess = guess_up.copy()
            else:
                best_guess = guess_crossing if 'guess_crossing' in dir() and guess_crossing is not None else guess_up.copy()

        
        onset_results[T] = result  # Store full result
        if verbose:
            status = "✓" if converged else "✗"
            print(f"  {status} T={T:6.2f} n_B={result.n_B:.4f} μp={result.mu_p_H:.1f} μn={result.mu_n_H:.1f} μu={result.mu_u_Q:.1f} μd={result.mu_d_Q:.1f} μs={result.mu_s_Q:.1f} μeH={result.mu_eL_H:.1f} μeQ={result.mu_eL_Q:.1f} μeG={result.mu_eG:.1f} np={result.n_p_H:.4f} nn={result.n_n_H:.4f} nu={result.n_u_Q:.4f} nd={result.n_d_Q:.4f} ns={result.n_s_Q:.4f} (err={result.error:.1e})")
        if converged:
            guess_up = result_to_guess(result, eta, eq_mode)
            converged_onset_history.append((T, guess_up.copy()))
    
    # March downward (same strategy but reversed history tracking)
    guess_down = current_guess.copy()
    # Store full guess vectors in history for better extrapolation
    converged_onset_history_down = [(T_0, current_guess.copy())] if T_0 in onset_results and onset_results[T_0].converged else []
    
    for i in range(idx_start - 1, -1, -1):
        T = T_sorted[i]
        
        # Track all results to pick best if none converge
        all_results = []
        
        # Strategy 1: Extrapolate ALL unknowns from history (better for T extrapolation)
        if len(converged_onset_history_down) >= 2:
            guess_extrap = extrapolate_guess_boundaries(converged_onset_history_down, T, guess_down)
            result = solve_fixed_chi(T, chi=0.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_extrap,
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        else:
            converged = False
        
        # Strategy 2: Use previous converged solution directly (fallback)
        if not converged and len(converged_onset_history_down) >= 1:
            result = solve_fixed_chi(T, chi=0.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_down.copy(),
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        
        # Strategy 3: Use n_B from free energy crossing (n_B(f_H=f_Q) - delta)
        if not converged:
            n_B_crossing_est = estimate_boundary_n_B(H_interp, Q_interp, T, eta, boundary="onset")
            if n_B_crossing_est is None:
                n_B_crossing_est = n_B_onset_init
            guess_crossing = build_initial_guess_boundaries_given_nB(
                H_interp, n_B_crossing_est, T, eta, eq_mode, phase="H", Y_C=Y_C
            )
            if guess_crossing is not None:
                result = solve_fixed_chi(T, chi=0.0, eta=eta, zl_params=zl_params,
                                          vmit_params=vmit_params, initial_guess=guess_crossing,
                                          eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
                all_results.append(result)
                converged = result.converged and result.error < 1e-4
        
        # If none converged, pick result with lowest error
        if not converged and all_results:
            result = min(all_results, key=lambda r: r.error)

        onset_results[T] = result  # Store full result
        if verbose:
            status = "✓" if converged else "✗"
            print(f"  {status} T={T:6.2f} n_B={result.n_B:.4f} μp={result.mu_p_H:.1f} μn={result.mu_n_H:.1f} μu={result.mu_u_Q:.1f} μd={result.mu_d_Q:.1f} μs={result.mu_s_Q:.1f} μeH={result.mu_eL_H:.1f} μeQ={result.mu_eL_Q:.1f} μeG={result.mu_eG:.1f} np={result.n_p_H:.4f} nn={result.n_n_H:.4f} nu={result.n_u_Q:.4f} nd={result.n_d_Q:.4f} ns={result.n_s_Q:.4f} (err={result.error:.1e})")
        if converged:
            guess_down = result_to_guess(result, eta, eq_mode)
            converged_onset_history_down.append((T, guess_down.copy()))
    
    # --- Find offset (χ=1) ---
    if verbose:
        print(f"\n--- Offset (χ=1, η={eta:.2f}) ---")
    
    # Find working start point for offset
    T_0_offset, n_B_offset_est, guess_offset, result_offset, offset_idx_start = find_working_start(
        T_values, eta, chi=1.0, H_interp=H_interp, Q_interp=Q_interp,
        zl_params=zl_params, vmit_params=vmit_params,
        eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L,
        T_start_default=T_start_default, verbose=verbose
    )
    
    # If no working start found, all T values failed
    if T_0_offset is None:
        if verbose:
            print(f"  ✗ No working start found for offset")
        # Return onset-only results (offset will be NaN)
        boundaries = []
        for T in T_sorted:
            onset_result = onset_results.get(T, None)
            if onset_result is not None:
                boundaries.append(PhaseBoundaryResult(
                    T=T, eta=eta,
                    n_B_onset=onset_result.n_B,
                    n_B_offset=np.nan,
                    converged_onset=onset_result.converged and onset_result.error < 1e-4,
                    converged_offset=False,
                    error_onset=onset_result.error,
                    error_offset=1.0
                ))
        return boundaries
    
    T_0 = T_0_offset
    result = result_offset
    guess = guess_offset
    
    offset_results[T_0] = result  # Store full result
    converged = result.converged and result.error < 1e-4
    if verbose:
        status = "✓" if converged else "✗"
        print(f"  {status} T={T_0:6.2f} n_B={result.n_B:.4f} μp={result.mu_p_H:.1f} μn={result.mu_n_H:.1f} μu={result.mu_u_Q:.1f} μd={result.mu_d_Q:.1f} μs={result.mu_s_Q:.1f} μeH={result.mu_eL_H:.1f} μeQ={result.mu_eL_Q:.1f} μeG={result.mu_eG:.1f} np={result.n_p_H:.4f} nn={result.n_n_H:.4f} nu={result.n_u_Q:.4f} nd={result.n_d_Q:.4f} ns={result.n_s_Q:.4f} (err={result.error:.1e})")
    
    current_guess = result_to_guess(result, eta, eq_mode) if converged else guess
    # Store full guess vectors in history for better extrapolation
    converged_offset_history = [(T_0, current_guess.copy())] if converged else []
    
    # March upward from offset starting point
    guess_up = current_guess.copy()
    for i in range(offset_idx_start + 1, len(T_sorted)):
        T = T_sorted[i]
        
        # Track all results to pick best if none converge
        all_results = []
        
        # Strategy 1: Use previous converged solution directly (most stable)
        if len(converged_offset_history) >= 1:
            result = solve_fixed_chi(T, chi=1.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_up.copy(),
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        else:
            converged = False
        
        # Strategy 2: Extrapolate ALL unknowns from history (if previous didn't work)
        if not converged and len(converged_offset_history) >= 2:
            guess_extrap = extrapolate_guess_boundaries(converged_offset_history, T, guess_up)
            result = solve_fixed_chi(T, chi=1.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_extrap,
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        
        # Strategy 3: Use n_B from free energy crossing (n_B(f_H=f_Q) + delta)
        if not converged:
            n_B_crossing_est = estimate_boundary_n_B(H_interp, Q_interp, T, eta, boundary="offset")
            if n_B_crossing_est is None:
                n_B_crossing_est = n_B_offset_init
            guess_crossing = build_initial_guess_boundaries_given_nB(
                Q_interp, n_B_crossing_est, T, eta, eq_mode, phase="Q", Y_C=Y_C
            )
            if guess_crossing is not None:
                result = solve_fixed_chi(T, chi=1.0, eta=eta, zl_params=zl_params,
                                          vmit_params=vmit_params, initial_guess=guess_crossing,
                                          eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
                all_results.append(result)
                converged = result.converged and result.error < 1e-4
        
        # Strategy 4: nB sweep from onset (consistent with table generator approach)
        if not converged:
            onset_result_T = onset_results.get(T, None)
            if onset_result_T is not None and onset_result_T.converged:
                result = find_offset_by_nB_sweep(
                    T, eta=eta, onset_result=onset_result_T,
                    zl_params=zl_params, vmit_params=vmit_params,
                    n_B_step=0.005, chi_threshold=0.999, verbose=False,
                    eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L
                )
                all_results.append(result)
                converged = result.converged and result.chi >= 0.99
        # If none converged, pick result with lowest error
        if not converged and all_results:
            result = min(all_results, key=lambda r: r.error)

        
        offset_results[T] = result  # Store full result
        if verbose:
            status = "✓" if converged else "✗"
            print(f"  {status} T={T:6.2f} n_B={result.n_B:.4f} μp={result.mu_p_H:.1f} μn={result.mu_n_H:.1f} μu={result.mu_u_Q:.1f} μd={result.mu_d_Q:.1f} μs={result.mu_s_Q:.1f} μeH={result.mu_eL_H:.1f} μeQ={result.mu_eL_Q:.1f} μeG={result.mu_eG:.1f} np={result.n_p_H:.4f} nn={result.n_n_H:.4f} nu={result.n_u_Q:.4f} nd={result.n_d_Q:.4f} ns={result.n_s_Q:.4f} (err={result.error:.1e})")
        if converged:
            guess_up = result_to_guess(result, eta, eq_mode)
            converged_offset_history.append((T, guess_up.copy()))
    
    # March downward from offset starting point (same strategy)
    guess_down = current_guess.copy()
    # Store full guess vectors in history for better extrapolation
    converged_offset_history_down = [(T_0, current_guess.copy())] if T_0 in offset_results and offset_results[T_0].converged else []
    
    for i in range(offset_idx_start - 1, -1, -1):
        T = T_sorted[i]
        
        # Track all results to pick best if none converge
        all_results = []
        
        # Strategy 1: Extrapolate ALL unknowns from history (better for T extrapolation)
        if len(converged_offset_history_down) >= 2:
            guess_extrap = extrapolate_guess_boundaries(converged_offset_history_down, T, guess_down)
            result = solve_fixed_chi(T, chi=1.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_extrap,
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        else:
            converged = False
        
        # Strategy 2: Use previous converged solution directly (fallback)
        if not converged and len(converged_offset_history_down) >= 1:
            result = solve_fixed_chi(T, chi=1.0, eta=eta, zl_params=zl_params,
                                      vmit_params=vmit_params, initial_guess=guess_down.copy(),
                                      eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
            all_results.append(result)
            converged = result.converged and result.error < 1e-4
        
        # Strategy 3: Use n_B from free energy crossing (n_B(f_H=f_Q) + delta)
        if not converged:
            n_B_crossing_est = estimate_boundary_n_B(H_interp, Q_interp, T, eta, boundary="offset")
            if n_B_crossing_est is None:
                n_B_crossing_est = n_B_offset_init
            guess_crossing = build_initial_guess_boundaries_given_nB(
                Q_interp, n_B_crossing_est, T, eta, eq_mode, phase="Q", Y_C=Y_C
            )
            if guess_crossing is not None:
                result = solve_fixed_chi(T, chi=1.0, eta=eta, zl_params=zl_params,
                                          vmit_params=vmit_params, initial_guess=guess_crossing,
                                          eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
                all_results.append(result)
                converged = result.converged and result.error < 1e-4
        
        # Strategy 4: nB sweep from onset (consistent with table generator approach)
        if not converged:
            onset_result_T = onset_results.get(T, None)
            if onset_result_T is not None and onset_result_T.converged:
                result = find_offset_by_nB_sweep(
                    T, eta=eta, onset_result=onset_result_T,
                    zl_params=zl_params, vmit_params=vmit_params,
                    n_B_step=0.005, chi_threshold=0.999, verbose=False,
                    eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L
                )
                all_results.append(result)
                converged = result.converged and result.chi >= 0.99
        # If none converged, pick result with lowest error
        if not converged and all_results:
            result = min(all_results, key=lambda r: r.error)
            
        offset_results[T] = result  # Store full result
        
        if verbose:
            status = "✓" if converged else "✗"
            print(f"  {status} T={T:6.2f} n_B={result.n_B:.4f} μp={result.mu_p_H:.1f} μn={result.mu_n_H:.1f} μu={result.mu_u_Q:.1f} μd={result.mu_d_Q:.1f} μs={result.mu_s_Q:.1f} μeH={result.mu_eL_H:.1f} μeQ={result.mu_eL_Q:.1f} μeG={result.mu_eG:.1f} np={result.n_p_H:.4f} nn={result.n_n_H:.4f} nu={result.n_u_Q:.4f} nd={result.n_d_Q:.4f} ns={result.n_s_Q:.4f} (err={result.error:.1e})")
        if converged:
            guess_down = result_to_guess(result, eta, eq_mode)
            converged_offset_history_down.append((T, guess_down.copy()))
    
    # Combine results - extract onset solution for warm-starting
    boundaries = []
    for T in T_sorted:
        onset_result = onset_results.get(T, None)
        offset_result = offset_results.get(T, None)
        
        # Extract onset values
        if onset_result is not None:
            n_on = onset_result.n_B
            conv_on = onset_result.converged and onset_result.error < 1e-4
            err_on = onset_result.error
        else:
            n_on, conv_on, err_on = np.nan, False, 1.0
        
        # Extract offset values
        if offset_result is not None:
            n_off = offset_result.n_B
            conv_off = offset_result.converged and offset_result.error < 1e-4
            err_off = offset_result.error
        else:
            n_off, conv_off, err_off = np.nan, False, 1.0
        
        # Build PhaseBoundaryResult with onset solution for warm-starting
        boundary = PhaseBoundaryResult(
            T=T, eta=eta,
            n_B_onset=n_on, n_B_offset=n_off,
            converged_onset=conv_on, converged_offset=conv_off,
            error_onset=err_on, error_offset=err_off
        )
        
        # Add onset solution if available
        if onset_result is not None and conv_on:
            boundary.mu_p_H_onset = onset_result.mu_p_H
            boundary.mu_n_H_onset = onset_result.mu_n_H
            boundary.mu_u_Q_onset = onset_result.mu_u_Q
            boundary.mu_d_Q_onset = onset_result.mu_d_Q
            boundary.mu_s_Q_onset = onset_result.mu_s_Q
            boundary.mu_eL_H_onset = onset_result.mu_eL_H
            boundary.mu_eL_Q_onset = onset_result.mu_eL_Q
            boundary.mu_eG_onset = onset_result.mu_eG
            boundary.n_p_H_onset = onset_result.n_p_H
            boundary.n_n_H_onset = onset_result.n_n_H
            boundary.n_u_Q_onset = onset_result.n_u_Q
            boundary.n_d_Q_onset = onset_result.n_d_Q
            boundary.n_s_Q_onset = onset_result.n_s_Q
            boundary.n_eL_H_onset = onset_result.n_eL_H
            boundary.n_eL_Q_onset = onset_result.n_eL_Q
            boundary.n_eG_onset = onset_result.n_eG
        
        boundaries.append(boundary)
    
    return boundaries



# =============================================================================
# HIGH-LEVEL API
# =============================================================================
def get_or_compute_boundaries(eta: float, T_values: np.ndarray,
                               output_dir: str = "output",
                               force_recompute: bool = False,
                               zl_params: ZLParams = None,
                               vmit_params: VMITParams = None,
                               verbose: bool = True,
                               debug: bool = False,
                               H_table_lookup: dict = None,
                               Q_table_lookup: dict = None,
                               eq_mode: str = "beta",
                               Y_C: float = None,
                               Y_L: float = None) -> List[PhaseBoundaryResult]:
    """
    Get phase boundaries from file if exists, otherwise compute and save.
    
    Args:
        eta: parameter controlling local-global total electric charge neutrality
        T_values: Temperature array
        output_dir: Output directory for boundary files
        force_recompute: Force recomputation even if file exists
        zl_params: ZL model parameters
        vmit_params: vMIT model parameters
        verbose: Print progress
        debug: Print detailed guess and result values for each T
        H_table_lookup: Optional dict of pure H phase results for initial guesses
        Q_table_lookup: Optional dict of pure Q phase results for initial guesses
        eq_mode: Equilibrium mode ("beta", "fixed_yc", or "trapped")
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
    """
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    filename = get_boundary_filename(eta, output_dir, zl_params, vmit_params,
                                     eq_mode=eq_mode)
    
    # Check if we can load existing data
    if not force_recompute and os.path.exists(filename):
        # Try to load boundaries for this specific composition
        existing = load_boundaries_from_file(filename, eta, eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
        if existing:
            if verbose:
                comp_str = ""
                if eq_mode == "fixed_yc" and Y_C is not None:
                    comp_str = f" (Y_C={Y_C:.4f})"
                elif eq_mode == "trapped" and Y_L is not None:
                    comp_str = f" (Y_L={Y_L:.4f})"
                print(f"Loading boundaries from {os.path.basename(filename)}{comp_str}: {len(existing)} T points")
            return existing
        # File exists but doesn't have this composition - need to compute and append
    
    if verbose:
        mode_str = eq_mode
        if eq_mode == "fixed_yc" and Y_C is not None:
            mode_str = f"fixed_yc (Y_C={Y_C:.4f})"
        elif eq_mode == "trapped" and Y_L is not None:
            mode_str = f"trapped (Y_L={Y_L:.4f})"
        print(f"Computing boundaries for η={eta:.2f} [{mode_str}]...")
    
    boundaries = find_boundaries(
        T_values, eta, 
        zl_params=zl_params,
        vmit_params=vmit_params,
        verbose=verbose,
        debug=debug,
        H_table_lookup=H_table_lookup,
        Q_table_lookup=Q_table_lookup,
        eq_mode=eq_mode,
        Y_C=Y_C,
        Y_L=Y_L
    )
    save_boundaries_to_file(boundaries, eta, output_dir, zl_params, vmit_params,
                           eq_mode=eq_mode, Y_C=Y_C, Y_L=Y_L)
    
    if verbose:
        print(f"Saved boundaries to {os.path.basename(filename)}")
    
    return boundaries



# =============================================================================
# =============================================================================
# PART 5: TABLE GENERATION
# =============================================================================
# =============================================================================

def generate_unified_table(n_B_values: np.ndarray, T_values: np.ndarray, 
                           eta: float, zl_params, vmit_params,
                           H_table: dict, Q_table: dict,
                           boundaries: dict = None,
                           verbose: bool = True,
                           eq_mode: str = "beta",
                           Y_C: float = None,
                           Y_L: float = None) -> list:
    """
    Generate unified EOS table combining all phases:
    - Pure H for n_B < n_onset (from precomputed full H table)
    - Mixed phase for n_onset <= n_B <= n_offset (computed on-demand)
    - Pure Q for n_B > n_offset (from precomputed full Q table)
    
    Args:
        n_B_values: Baryon density grid
        T_values: Temperature grid
        eta: parameter controlling local-global total electric charge neutrality
        zl_params: ZL model parameters
        vmit_params: vMIT model parameters
        H_table: Precomputed pure H table keyed by (n_B, T)
        Q_table: Precomputed pure Q table keyed by (n_B, T)
        boundaries: Pre-loaded boundaries dict (if provided, boundary_dir is ignored)
        verbose: Print progress
        eq_mode: Equilibrium mode ("beta", "fixed_yc", or "trapped")
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
    
    Returns a single list of MixedPhaseResult covering the full density range.
    """
    
    if boundaries is None:
        print(f"  ERROR: No phase boundary table found for η={eta:.2f}")
        print(f"  Please run find_phase_boundaries.py first.")
        return []
    
    # =========================================================================
    # Assemble unified table for each T
    # =========================================================================
    if verbose:
        mode_str = eq_mode
        if eq_mode == "fixed_yc" and Y_C is not None:
            mode_str = f"fixed_yc (Y_C={Y_C:.4f})"
        elif eq_mode == "trapped" and Y_L is not None:
            mode_str = f"trapped (Y_L={Y_L:.4f})"
        print(f"  Assembling unified table [{mode_str}]...")
    
    results = []
    
    for T in T_values:
        # Find closest T in boundaries
        boundary_Ts = sorted(boundaries.keys())
        closest_T = min(boundary_Ts, key=lambda t: abs(t - T))
        
        if abs(closest_T - T) > 5.0:
            print(f"  Warning: No boundary for T={T:.1f} MeV, using closest T={closest_T:.1f}")
        
        bdata = boundaries[closest_T]
        n_onset = bdata['n_onset']
        n_offset = bdata['n_offset']
        onset_guess = bdata.get('onset_guess', None)
        
        n_H = n_mixed = n_Q = n_mixed_conv = 0
        mixed_guess = onset_guess
        first_mixed_point = True
        T_start_time = time.time()
        
        # Track converged history for extrapolation: list of (n_B, guess_array) tuples
        converged_history = []
        
        # Flag to track when we've transitioned to pure Q (chi >= 1)
        in_pure_Q = False
        
        for n_B in n_B_values:
            key = (n_B, T)
            
            if n_B < n_onset:
                # Use precomputed pure H result
                if key in H_table:
                    result = H_table[key]
                    result.eta = eta
                    results.append(result)
                    n_H += 1
            
            elif in_pure_Q:
                # Already transitioned to pure Q, use Q table
                if key in Q_table:
                    result = Q_table[key]
                    result.eta = eta
                    results.append(result)
                    n_Q += 1
                    
            else:
                # In mixed phase region - solve until chi reaches 1
                
                # Use extrapolation as primary guess when we have history
                use_extrap = False
                guess_to_use = mixed_guess
                if len(converged_history) >= 2:
                    n_B1, guess1 = converged_history[-2]
                    n_B2, guess2 = converged_history[-1]
                    if abs(n_B2 - n_B1) > 1e-10:
                        slope = (guess2 - guess1) / (n_B2 - n_B1)
                        guess_to_use = guess2 + slope * (n_B - n_B2)
                        use_extrap = True
                
                result = solve_mixed_phase(
                    n_B, T, eta,
                    zl_params=zl_params,
                    vmit_params=vmit_params,
                    initial_guess=guess_to_use,
                    eq_mode=eq_mode,
                    Y_C=Y_C
                )
                
                # Debug: show first mixed point guess vs result
                if first_mixed_point and verbose and mixed_guess is not None:
                    print(f"      First mixed @ n_B={n_B:.4f}, conv={result.converged}, err={result.error:.2e}")
                    first_mixed_point = False
                elif first_mixed_point:
                    first_mixed_point = False
                
                # If extrapolation failed, try previous solution as fallback
                if not result.converged and use_extrap and mixed_guess is not None:
                    result_fallback = solve_mixed_phase(
                        n_B, T, eta,
                        zl_params=zl_params,
                        vmit_params=vmit_params,
                        initial_guess=mixed_guess,
                        eq_mode=eq_mode,
                        Y_C=Y_C
                    )
                    if result_fallback.converged or result_fallback.error < result.error:
                        result = result_fallback
                
                # Update guess and history for next point
                if result.converged:
                    mixed_guess = result_to_guess_for_unified(result, eta)
                    converged_history.append((n_B, mixed_guess.copy()))
                    # Keep history bounded to last 5 points
                    if len(converged_history) > 5:
                        converged_history.pop(0)
                
                # Check if chi has reached 1 - transition to pure Q
                if result.chi >= 1.0 and key in Q_table:
                    # Use pure Q for this point and all subsequent points
                    result = Q_table[key]
                    result.eta = eta
                    n_Q += 1
                    in_pure_Q = True  # All remaining points will be pure Q
                else:
                    n_mixed += 1
                    if result.converged:
                        n_mixed_conv += 1
                
                results.append(result)
        
        T_elapsed_time = time.time() - T_start_time
        avg_time_per_nB = T_elapsed_time / len(n_B_values) * 1000
        if verbose:
            conv_pct = n_mixed_conv/n_mixed*100 if n_mixed > 0 else 0.0
            print(f"    T={T:.1f} MeV: n_on={n_onset:.4f}, n_off={n_offset:.4f} → {n_H} H, {n_mixed} mixed ({conv_pct:.2f}% conv), {n_Q} Q  [{T_elapsed_time:.2f}s total, {avg_time_per_nB:.1f}ms/n_B]")
    
    return results


def result_to_guess_for_unified(result, eta: float) -> np.ndarray:
    """Extract initial guess from a MixedPhaseResult based on eta value.
    
    Used by generate_unified_table for warm-starting.
    """
    if abs(eta) < 1e-10:
        # η=0 (Gibbs): 12 unknowns
        return np.array([
            result.mu_p_H, result.mu_n_H, result.mu_u_Q, result.mu_d_Q, result.mu_s_Q, result.mu_eG,
            result.n_p_H, result.n_n_H, result.n_u_Q, result.n_d_Q, result.n_s_Q, result.chi
        ])
    elif abs(eta - 1.0) < 1e-10:
        # η=1 (Maxwell): 13 unknowns
        return np.array([
            result.mu_p_H, result.mu_n_H, result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            result.mu_eL_H, result.mu_eL_Q,
            result.n_p_H, result.n_n_H, result.n_u_Q, result.n_d_Q, result.n_s_Q, result.chi
        ])
    else:
        # 0<η<1: 14 unknowns
        return np.array([
            result.mu_p_H, result.mu_n_H, result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            result.mu_eL_H, result.mu_eL_Q, result.mu_eG,
            result.n_p_H, result.n_n_H, result.n_u_Q, result.n_d_Q, result.n_s_Q, result.chi
        ])

