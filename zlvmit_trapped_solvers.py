#!/usr/bin/env python3
"""
zlvmit_trapped_solvers.py
=========================
Trapped neutrino mode solvers for ZL+vMIT mixed phase EOS.

This module provides mixed phase solvers for the trapped neutrino case,
where the lepton fraction Y_L = (n_e + n_ν) / n_B is fixed.

Physics:
    - Electrons and electron neutrinos are in equilibrium: μ_e - μ_ν = μ_C
    - Lepton number is conserved globally: n_e + n_ν = Y_L * n_B
    - Global charge neutrality still applies
    
Solver structure (η=0 GCN):
    14 unknowns: [μp, μn, μu, μd, μs, μe, μν, np, nn, nu, nd, ns, nν, χ]
"""

import numpy as np
from scipy.optimize import root
from dataclasses import dataclass
from typing import Optional

# Import from parent modules
from zl_parameters import ZLParams, get_zl_default
from zl_thermodynamics_nucleons import compute_zl_thermo_from_mu_n

from vmit_parameters import VMITParams, get_vmit_default
from vmit_thermodynamics_quarks import compute_vmit_thermo_from_mu_n

from general_thermodynamics_leptons import electron_thermo, neutrino_thermo, photon_thermo


# =============================================================================
# η=0 TRAPPED NEUTRINO MODE (Gibbs construction)
# =============================================================================

def solve_eta0_trapped(
    n_B: float, 
    Y_L: float, 
    T: float,
    zl_params: ZLParams = None,
    vmit_params: VMITParams = None,
    initial_guess: np.ndarray = None
):
    """
    Solve mixed phase for η=0 (Gibbs construction) with trapped neutrinos.
    
    14 unknowns: [μp, μn, μu, μd, μs, μe, μν, np, nn, nu, nd, ns, nν, χ]
    
    Equations:
        1-5.  Self-consistency for np, nn, nu, nd, ns
        6.    (1-χ)*n_B_H + χ*n_B_Q = n_B                  (baryon conservation)
        7.    (1-χ)*n_C_H + χ*n_C_Q = n_e                  (global charge neutrality)
        8.    n_e + n_ν = n_B * Y_L                        (lepton number conservation)
        9.    μ_B_H = μ_B_Q                                (baryon chemical equilibrium)
        10.   μ_C_H + μ_e - μ_ν = 0                        (beta eq with neutrinos in H)
        11.   μ_C_Q + μ_e - μ_ν = 0                        (beta eq with neutrinos in Q)
        12.   μ_S_Q = 0                                    (strangeness equilibrium)
        13.   P_H = P_Q                                    (mechanical equilibrium)
        14.   n_ν_calc - n_ν = 0                           (neutrino self-consistency)
    
    Args:
        n_B: Baryon density (fm⁻³)
        Y_L: Lepton fraction (n_e + n_ν) / n_B
        T: Temperature (MeV)
        zl_params: ZL hadronic parameters
        vmit_params: vMIT quark parameters
        initial_guess: Initial guess array (14 elements)
        
    Returns:
        MixedPhaseResult with trapped neutrino quantities
    """
    # Import here to avoid circular import
    from zlvmit_mixed_phase_eos import MixedPhaseResult
    
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Default initial guess
    if initial_guess is None:
        # Estimate initial densities
        n_e_est = Y_L * n_B * 0.6  # Most leptons are electrons
        n_nu_est = Y_L * n_B * 0.4
        initial_guess = np.array([
            700, 1300, 150, 500, 500, 100, 50,  # μp, μn, μu, μd, μs, μe, μν
            Y_L * n_B * 0.4, (1 - Y_L * 0.4) * n_B,  # np, nn
            n_B, n_B, 0.5 * n_B,  # nu, nd, ns
            n_nu_est, 0.1  # nν, χ
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_e, mu_nu, n_p, n_n, n_u, n_d, n_s, n_nu, chi = x
        
        # Compute phase thermodynamics
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        ele_sec = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_sec = neutrino_thermo(mu_nu, T, include_antiparticles=True)
        
        # Self-consistency densities
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_nu_calc = nu_sec.n
        n_e = ele_sec.n
        
        # Phase densities
        n_B_H = had_sec.n_B
        n_B_Q = qua_sec.n_B
        n_C_H = had_sec.n_C
        n_C_Q = qua_sec.n_C
        
        # Chemical potentials
        mu_B_H = had_sec.mu_B
        mu_B_Q = qua_sec.mu_B
        mu_C_H = had_sec.mu_C
        mu_C_Q = qua_sec.mu_C
        mu_S_Q = qua_sec.mu_S
        
        # Pressures
        P_H = had_sec.P
        P_Q = qua_sec.P
        
        # Build residual vector
        res = np.zeros(14)
        
        # Particle consistency (normalized)
        res[0] = (n_p_calc - n_p) / max(abs(n_p), 1e-8)
        res[1] = (n_n_calc - n_n) / max(abs(n_n), 1e-8)
        res[2] = (n_u_calc - n_u) / max(abs(n_u), 1e-8)
        res[3] = (n_d_calc - n_d) / max(abs(n_d), 1e-8)
        res[4] = (n_s_calc - n_s) / max(abs(n_s), 1e-8)
        
        # Baryon conservation
        res[5] = ((1 - chi) * n_B_H + chi * n_B_Q - n_B) / max(abs(n_B), 1e-8)
        
        # Global charge neutrality: total hadronic charge = electron density
        n_C_tot = (1 - chi) * n_C_H + chi * n_C_Q
        res[6] = (n_C_tot - n_e) / max(abs(n_e), abs(n_C_tot), 1e-8)
        
        # Lepton number conservation
        n_L_tot = n_e + n_nu
        res[7] = (n_L_tot - n_B * Y_L) / max(abs(n_B * Y_L), 1e-8)
        
        # Baryon chemical equilibrium
        res[8] = (mu_B_H - mu_B_Q) / max(abs(mu_B_H), 1e-8)
        
        # Beta equilibrium with neutrinos (in H and Q sectors)
        # d → u + e⁻ + ν̄_e implies μ_d = μ_u + μ_e - μ_ν
        # For conserved charges: μ_C + μ_e - μ_ν = 0
        res[9] = (mu_C_H + mu_e - mu_nu) / max(abs(mu_C_H), abs(mu_e), 1e-8)
        res[10] = (mu_C_Q + mu_e - mu_nu) / max(abs(mu_C_Q), abs(mu_e), 1e-8)
        
        # Strangeness equilibrium in Q phase
        res[11] = mu_S_Q / max(abs(mu_B_Q), 1e-8)
        
        # Mechanical equilibrium
        res[12] = (P_H - P_Q) / max(abs(P_H), 1e-8)
        
        # Neutrino self-consistency
        res[13] = (n_nu_calc - n_nu) / max(abs(n_nu), 1e-8)
        
        return res
    
    # Solve
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    # Extract solution
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_e, mu_nu, n_p, n_n, n_u, n_d, n_s, n_nu, chi = sol.x
    
    # Recompute thermodynamics at solution
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    ele_sec = electron_thermo(mu_e, T, include_antiparticles=True)
    nu_sec = neutrino_thermo(mu_nu, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        eq_mode="trapped", Y_L_input=Y_L,
        # Hadronic phase
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        # Quark phase
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        # Electrons (global in η=0)
        mu_eG=mu_e, n_eG=ele_sec.n,
        P_eG=ele_sec.P, e_eG=ele_sec.e, s_eG=ele_sec.s,
        # Neutrinos (global)
        mu_nuG=mu_nu, n_nuG=n_nu,
        P_nuG=nu_sec.P, e_nuG=nu_sec.e, s_nuG=nu_sec.s,
        # Photons
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# η=0 FIXED-CHI TRAPPED (for boundary finding)
# =============================================================================

def solve_eta0_fixed_chi_trapped(
    T: float,
    chi: float,
    Y_L: float,
    zl_params: ZLParams = None,
    vmit_params: VMITParams = None,
    initial_guess: np.ndarray = None
):
    """
    Solve mixed phase for η=0 with fixed χ and trapped neutrinos.
    
    Used for finding phase boundaries (χ=0 for onset, χ=1 for offset).
    
    13 unknowns: [μp, μn, μu, μd, μs, μe, μν, np, nn, nu, nd, ns, nν]
    (n_B is determined from the solution)
    
    Returns:
        MixedPhaseResult with n_B determined from the solution
    """
    from zlvmit_mixed_phase_eos import MixedPhaseResult
    
    if zl_params is None:
        zl_params = get_zl_default()
    if vmit_params is None:
        vmit_params = get_vmit_default()
    
    # Default guess
    if initial_guess is None:
        initial_guess = np.array([
            700, 1300, 150, 500, 500, 100, 50,  # μp, μn, μu, μd, μs, μe, μν
            0.2, 0.4,  # np, nn
            0.5, 0.5, 0.3,  # nu, nd, ns
            0.1  # nν
        ])
    
    def equations(x):
        mu_p, mu_n, mu_u, mu_d, mu_s, mu_e, mu_nu, n_p, n_n, n_u, n_d, n_s, n_nu = x
        
        had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
        qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
        ele_sec = electron_thermo(mu_e, T, include_antiparticles=True)
        nu_sec = neutrino_thermo(mu_nu, T, include_antiparticles=True)
        
        n_p_calc = had_sec.n_p
        n_n_calc = had_sec.n_n
        n_u_calc = qua_sec.n_u
        n_d_calc = qua_sec.n_d
        n_s_calc = qua_sec.n_s
        n_nu_calc = nu_sec.n
        n_e = ele_sec.n
        
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
        
        # Determine n_B from chi
        n_B = (1 - chi) * n_B_H + chi * n_B_Q
        
        # Total lepton number
        n_L = n_e + n_nu
        
        res = np.zeros(13)
        res[0] = (n_p_calc - n_p) / max(abs(n_p), 1e-8)
        res[1] = (n_n_calc - n_n) / max(abs(n_n), 1e-8)
        res[2] = (n_u_calc - n_u) / max(abs(n_u), 1e-8)
        res[3] = (n_d_calc - n_d) / max(abs(n_d), 1e-8)
        res[4] = (n_s_calc - n_s) / max(abs(n_s), 1e-8)
        
        # Global charge neutrality
        n_C_tot = (1 - chi) * n_C_H + chi * n_C_Q
        res[5] = (n_C_tot - n_e) / max(abs(n_e), abs(n_C_tot), 1e-8)
        
        # Lepton fraction constraint
        res[6] = (n_L - n_B * Y_L) / max(abs(n_B * Y_L), 1e-8) if n_B > 1e-10 else n_L
        
        # Chemical equilibria
        res[7] = (mu_B_H - mu_B_Q) / max(abs(mu_B_H), 1e-8)
        res[8] = (mu_C_H + mu_e - mu_nu) / max(abs(mu_C_H), abs(mu_e), 1e-8)
        res[9] = (mu_C_Q + mu_e - mu_nu) / max(abs(mu_C_Q), abs(mu_e), 1e-8)
        res[10] = mu_S_Q / max(abs(mu_B_Q), 1e-8)
        
        # Mechanical equilibrium
        res[11] = (P_H - P_Q) / max(abs(P_H), 1e-8)
        
        # Neutrino consistency
        res[12] = (n_nu_calc - n_nu) / max(abs(n_nu), 1e-8)
        
        return res
    
    sol = root(equations, initial_guess, method='hybr', options={'maxfev': 2000}, tol=1e-6)
    if not sol.success:
        sol = root(equations, initial_guess, method='lm', options={'maxiter': 2000}, tol=1e-6)
    
    mu_p, mu_n, mu_u, mu_d, mu_s, mu_e, mu_nu, n_p, n_n, n_u, n_d, n_s, n_nu = sol.x
    
    had_sec = compute_zl_thermo_from_mu_n(mu_p, mu_n, n_p, n_n, T, zl_params)
    qua_sec = compute_vmit_thermo_from_mu_n(mu_u, mu_d, mu_s, n_u, n_d, n_s, T, vmit_params)
    ele_sec = electron_thermo(mu_e, T, include_antiparticles=True)
    nu_sec = neutrino_thermo(mu_nu, T, include_antiparticles=True)
    gamma_thermo = photon_thermo(T)
    
    n_B = (1 - chi) * had_sec.n_B + chi * qua_sec.n_B
    
    return MixedPhaseResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=n_B, T=T, eta=0.0, chi=chi,
        eq_mode="trapped", Y_L_input=Y_L,
        mu_p_H=mu_p, mu_n_H=mu_n, n_p_H=n_p, n_n_H=n_n,
        P_H=had_sec.P, e_H=had_sec.e, s_H=had_sec.s, f_H=had_sec.f,
        n_B_H=had_sec.n_B, n_C_H=had_sec.n_C, n_S_H=had_sec.n_S,
        mu_B_H=had_sec.mu_B, mu_C_H=had_sec.mu_C, mu_S_H=had_sec.mu_S,
        mu_u_Q=mu_u, mu_d_Q=mu_d, mu_s_Q=mu_s, n_u_Q=n_u, n_d_Q=n_d, n_s_Q=n_s,
        P_Q=qua_sec.P, e_Q=qua_sec.e, s_Q=qua_sec.s, f_Q=qua_sec.f,
        n_B_Q=qua_sec.n_B, n_C_Q=qua_sec.n_C, n_S_Q=qua_sec.n_S,
        mu_B_Q=qua_sec.mu_B, mu_C_Q=qua_sec.mu_C, mu_S_Q=qua_sec.mu_S,
        mu_eG=mu_e, n_eG=ele_sec.n,
        P_eG=ele_sec.P, e_eG=ele_sec.e, s_eG=ele_sec.s,
        mu_nuG=mu_nu, n_nuG=n_nu,
        P_nuG=nu_sec.P, e_nuG=nu_sec.e, s_nuG=nu_sec.s,
        P_gamma=gamma_thermo.P, e_gamma=gamma_thermo.e, s_gamma=gamma_thermo.s,
    )


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Trapped Neutrino Solvers Test")
    print("=" * 50)
    
    # Test parameters
    n_B = 0.5  # fm⁻³
    Y_L = 0.4  # Lepton fraction
    T = 30.0   # MeV
    
    print(f"\nTest point: n_B={n_B} fm⁻³, Y_L={Y_L}, T={T} MeV")
    
    # Test η=0 trapped solver
    print("\n1. Testing solve_eta0_trapped...")
    try:
        result = solve_eta0_trapped(n_B, Y_L, T)
        print(f"   converged={result.converged}, error={result.error:.2e}")
        print(f"   χ={result.chi:.4f}")
        print(f"   n_e={result.n_eG:.4f} fm⁻³, n_ν={result.n_nuG:.4f} fm⁻³")
        print(f"   Y_L_check={(result.n_eG + result.n_nuG)/n_B:.4f} (target: {Y_L})")
        print(f"   P_total={result.P_total:.2f} MeV/fm³")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    # Test fixed-chi solver (onset, chi=0)
    print("\n2. Testing solve_eta0_fixed_chi_trapped (χ=0 onset)...")
    try:
        result = solve_eta0_fixed_chi_trapped(T, chi=0.0, Y_L=Y_L)
        print(f"   converged={result.converged}, error={result.error:.2e}")
        print(f"   n_B_onset={result.n_B:.4f} fm⁻³")
        print(f"   P_total={result.P_total:.2f} MeV/fm³")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\nOK!" if result.converged else "\nSome tests failed")
