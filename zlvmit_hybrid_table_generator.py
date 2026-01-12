#!/usr/bin/env python3
"""
zlvmit_hybrid_table_generator.py
======================
Hybrid EOS Table Generator Script for ZL+vMIT mixed phase calculations.

This script orchestrates the entire computation workflow:
1. Initialize parameters
2. Load/compute phase boundaries for each η
3. Compute pure H and Q tables (once, reused for all η)
4. Compute mixed phase in transition region for each η
5. Assemble unified EOS tables (H → Mixed → Q)
6. Save results with comprehensive thermodynamic quantities

Usage:
    python zlvmit_hybrid_table_generator.py
"""

import numpy as np
import os
import time
from datetime import datetime


# =============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Density grid (fm^-3)
n0 = 0.16  # Nuclear saturation density
n_B_min = 0.1 * n0        # Start at low density
n_B_max = 12.0 * n0       # End at high density  
n_B_steps = 300           # Number of density points

# Temperature grid (MeV)
T_values = np.concatenate([[0.1], np.arange(2.5, 122.5, 2.5)])

# Surface tension parameter η: 0 = Gibbs, 1 = Maxwell, intermediate = hybrid
eta_values = [0.,0.1,0.3,0.6,1.0]

# Equilibrium mode: "beta", "fixed_yc", or "trapped"
#   - "beta": Beta equilibrium with electrons (charge neutrality)
#   - "fixed_yc": Fixed charge fraction Y_C (no electrons)
#   - "trapped": Trapped neutrinos (fixed lepton fraction Y_L)
EQUILIBRIUM_MODE = "beta"

# Charge fractions (used if EQUILIBRIUM_MODE = "fixed_yc")
# Output: one table per eta, each table contains all (n_B, T, Y_C) combinations
Y_C_values = [0.01,0.1,0.2,0.3, 0.4,0.5]

# Lepton fractions (used if EQUILIBRIUM_MODE = "trapped")
# Output: one table per eta, each table contains all (n_B, T, Y_L) combinations
Y_L_values = [0.3, 0.4, 0.5]

# vMIT bag model parameters
B4 = 165.0   # Bag constant B^{1/4} in MeV
a = 0.2      # Vector coupling in fm²

# Output directories (absolute paths)
OUTPUT_DIR = "/Users/mircoguerrini/Desktop/Research/Python_codes/output/zlvmit_hybrid_outputs"
BOUNDARY_DIR = "/Users/mircoguerrini/Desktop/Research/Python_codes/output/zlvmit_hybrid_outputs"

# Control flags
FORCE_RECOMPUTE_BOUNDARIES = True    # If True, recompute boundaries even if files exist
FORCE_RECOMPUTE_PURE_PHASES = False  # If True, recompute pure H/Q tables even if files exist
VERBOSE = True                       # Print detailed progress


# =============================================================================
# MIXED PHASE SOLVER DISPATCHER FOR FIXED Y_C
# =============================================================================

def boundary_result_to_onset_guess(b, eta: float) -> np.ndarray:
    """
    Convert PhaseBoundaryResult onset fields to a guess array for the mixed phase solver.
    
    The first mixed phase point (just above n_onset) should use the onset (χ=0) solution
    as its initial guess, with χ set to a small value instead of 0.
    
    Args:
        b: PhaseBoundaryResult with onset solution fields
        eta: parameter controlling local-global charge neutrality
        
    Returns:
        numpy array suitable as initial_guess for solve_mixed_phase
    """
    # Check if boundary has onset solution fields
    if not hasattr(b, 'mu_p_H_onset') or b.mu_p_H_onset is None:
        return None
    
    # Initial chi for first mixed point (slightly above onset)
    chi_init = 0.01
    
    if abs(eta) < 1e-10:
        # η=0: 12 unknowns [μp, μn, μu, μd, μs, μeG, np, nn, nu, nd, ns, χ]
        return np.array([
            b.mu_p_H_onset, b.mu_n_H_onset,
            b.mu_u_Q_onset, b.mu_d_Q_onset, b.mu_s_Q_onset,
            b.mu_eG_onset,
            b.n_p_H_onset, b.n_n_H_onset,
            b.n_u_Q_onset, b.n_d_Q_onset, b.n_s_Q_onset,
            chi_init
        ])
    elif abs(eta - 1.0) < 1e-10:
        # η=1: 13 unknowns [μp, μn, μu, μd, μs, μeL_H, μeL_Q, np, nn, nu, nd, ns, χ]
        return np.array([
            b.mu_p_H_onset, b.mu_n_H_onset,
            b.mu_u_Q_onset, b.mu_d_Q_onset, b.mu_s_Q_onset,
            b.mu_eL_H_onset, b.mu_eL_Q_onset,
            b.n_p_H_onset, b.n_n_H_onset,
            b.n_u_Q_onset, b.n_d_Q_onset, b.n_s_Q_onset,
            chi_init
        ])
    else:
        # 0<η<1: 14 unknowns [μp, μn, μu, μd, μs, μeL_H, μeL_Q, μeG, np, nn, nu, nd, ns, χ]
        return np.array([
            b.mu_p_H_onset, b.mu_n_H_onset,
            b.mu_u_Q_onset, b.mu_d_Q_onset, b.mu_s_Q_onset,
            b.mu_eL_H_onset, b.mu_eL_Q_onset, b.mu_eG_onset,
            b.n_p_H_onset, b.n_n_H_onset,
            b.n_u_Q_onset, b.n_d_Q_onset, b.n_s_Q_onset,
            chi_init
        ])

def result_to_guess_for_eta(result, eta: float) -> np.ndarray:
    """
    Convert a MixedPhaseResult to an initial guess array for the specified eta.
    
    Different eta values need different guess sizes:
    - eta=0: 12 elements [μp, μn, μu, μd, μs, μe, np, nn, nu, nd, ns, χ]
    - 0<eta<1: 16 elements [μp, μn, μu, μd, μs, μeH, μeQ, μeG, np, nn, nu, nd, ns, χ]
    - eta=1: 14 elements [μp, μn, μu, μd, μs, μeH, μeQ, np, nn, nu, nd, ns, χ]
    """
    # Get common values from result
    mu_e = getattr(result, 'mu_eG', 0) or getattr(result, 'mu_eL_H', 100)
    n_e = getattr(result, 'n_eG', 0) or getattr(result, 'n_eL_H', 0.01)
    
    if abs(eta) < 1e-10:
        # eta=0: 12 elements
        return np.array([
            result.mu_p_H, result.mu_n_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            mu_e,
            result.n_p_H, result.n_n_H,
            result.n_u_Q, result.n_d_Q, result.n_s_Q,
            result.chi
        ])
    elif abs(eta - 1.0) < 1e-10:
        # eta=1: 14 elements
        return np.array([
            result.mu_p_H, result.mu_n_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            mu_e, mu_e,  # μeH, μeQ
            result.n_p_H, result.n_n_H,
            result.n_u_Q, result.n_d_Q, result.n_s_Q,
            result.n_p_H,  # n_eH ≈ n_p
            result.chi
        ])
    else:
        # 0<eta<1: 16 elements
        return np.array([
            result.mu_p_H, result.mu_n_H,
            result.mu_u_Q, result.mu_d_Q, result.mu_s_Q,
            mu_e, mu_e, mu_e,  # μeH, μeQ, μeG
            result.n_p_H, result.n_n_H,
            result.n_u_Q, result.n_d_Q, result.n_s_Q,
            result.n_p_H, n_e,  # n_eH, n_eG
            result.chi
        ])



# =============================================================================
# THERMODYNAMIC OUTPUT - SPLIT INTO PRIMARY AND COMPLETE FILES
# =============================================================================

def results_to_dict_primary(results: list, eta_override: float = None) -> dict:
    """
    Convert list of MixedPhaseResult to dict with PRIMARY subset of quantities.
    
    Primary file contains:
    - Input: n_B, T, eta
    - Volume fraction: chi
    - Chemical potentials: mu_B_H, mu_C_H, mu_S_H, mu_B_Q, mu_C_Q, mu_S_Q
    - Conserved charge densities: n_B_H, n_C_H, n_S_H, n_B_Q, n_C_Q, n_S_Q
    
    Args:
        results: List of MixedPhaseResult or ZLEOSResult objects
        eta_override: If provided, use this eta value for all entries (for consistent table output)
    """
    if not results:
        return {}
    
    data = {
        # Input quantities
        'n_B': [], 'T': [], 'eta': [],
        # Convergence status
        'converged': [],
        # Volume fraction
        'chi': [],
        # Hadronic phase - conserved charge chemical potentials
        'mu_B_H': [], 'mu_C_H': [], 'mu_S_H': [],
        # Quark phase - conserved charge chemical potentials
        'mu_B_Q': [], 'mu_C_Q': [], 'mu_S_Q': [],
        # Electron chemical potentials
        'mu_eL_H': [], 'mu_eL_Q': [], 'mu_eG': [],
        # Hadronic phase - conserved charge densities
        'n_B_H': [], 'n_C_H': [], 'n_S_H': [],
        # Quark phase - conserved charge densities
        'n_B_Q': [], 'n_C_Q': [], 'n_S_Q': [],
    }
    
    for r in results:
        # Input - use eta_override if provided, otherwise try to get from result
        data['n_B'].append(r.n_B)
        data['T'].append(r.T)
        if eta_override is not None:
            data['eta'].append(eta_override)
        else:
            data['eta'].append(getattr(r, 'eta', 0.0))
        # Convergence - assume pure phases are converged
        data['converged'].append(1 if getattr(r, 'converged', True) else 0)
        
        # Determine phase type:
        # - MixedPhaseResult has 'chi' attribute
        # - Pure Q (VMITEOSResult) has 'mu_u' but not 'mu_p_H' -> chi=1
        # - Pure H (ZLEOSResult) has 'mu_p' but not 'mu_u' -> chi=0
        is_mixed = hasattr(r, 'chi')
        is_pure_Q = hasattr(r, 'mu_u') and not hasattr(r, 'mu_p_H')
        is_pure_H = hasattr(r, 'mu_p') and not hasattr(r, 'mu_u')
        
        # Volume fraction
        if is_mixed:
            data['chi'].append(r.chi)
        elif is_pure_Q:
            data['chi'].append(1.0)
        else:
            data['chi'].append(0.0)
        
        # Handle each phase type differently
        if is_pure_Q:
            # Pure Q phase (VMITEOSResult): Q values from result, H values = 0
            # Hadronic quantities = 0
            data['mu_B_H'].append(0.0)
            data['mu_C_H'].append(0.0)
            data['mu_S_H'].append(0.0)
            data['n_B_H'].append(0.0)
            data['n_C_H'].append(0.0)
            data['n_S_H'].append(0.0)
            data['mu_eL_H'].append(0.0)
            # Quark quantities from VMITEOSResult
            data['mu_B_Q'].append(getattr(r, 'mu_B', 0.0))
            data['mu_C_Q'].append(getattr(r, 'mu_C', 0.0))
            data['mu_S_Q'].append(getattr(r, 'mu_S', 0.0))
            # n_B_Q = n_B (entire baryon density is in Q phase)
            data['n_B_Q'].append(r.n_B)
            # n_C_Q = (2/3)*n_u - (1/3)*n_d - (1/3)*n_s
            n_u = getattr(r, 'n_u', 0.0)
            n_d = getattr(r, 'n_d', 0.0)
            n_s = getattr(r, 'n_s', 0.0)
            n_C_Q = (2.0/3.0) * n_u - (1.0/3.0) * n_d - (1.0/3.0) * n_s
            data['n_C_Q'].append(n_C_Q)
            # n_S_Q = n_s (number of strange quarks)
            data['n_S_Q'].append(n_s)
            data['mu_eL_Q'].append(getattr(r, 'mu_e', 0.0))
            data['mu_eG'].append(getattr(r, 'mu_e', 0.0))
        elif is_pure_H:
            # Pure H phase (ZLEOSResult): H values from result, Q values = 0
            data['mu_B_H'].append(getattr(r, 'mu_B', 0.0))
            data['mu_C_H'].append(getattr(r, 'mu_C', 0.0))
            data['mu_S_H'].append(0.0)  # No strangeness in hadronic phase
            data['n_B_H'].append(r.n_B)
            data['n_C_H'].append(getattr(r, 'n_p', 0.0))  # n_C_H = n_p in hadronic phase
            data['n_S_H'].append(0.0)
            data['mu_eL_H'].append(getattr(r, 'mu_e', 0.0))
            # Quark quantities = 0
            data['mu_B_Q'].append(0.0)
            data['mu_C_Q'].append(0.0)
            data['mu_S_Q'].append(0.0)
            data['n_B_Q'].append(0.0)
            data['n_C_Q'].append(0.0)
            data['n_S_Q'].append(0.0)
            data['mu_eL_Q'].append(0.0)
            data['mu_eG'].append(getattr(r, 'mu_e', 0.0))
        else:
            # Mixed phase (MixedPhaseResult): both H and Q quantities from result
            data['mu_B_H'].append(getattr(r, 'mu_B_H', 0.0))
            data['mu_C_H'].append(getattr(r, 'mu_C_H', 0.0))
            data['mu_S_H'].append(getattr(r, 'mu_S_H', 0.0))
            data['n_B_H'].append(getattr(r, 'n_B_H', 0.0))
            data['n_C_H'].append(getattr(r, 'n_C_H', 0.0))
            data['n_S_H'].append(getattr(r, 'n_S_H', 0.0))
            data['mu_eL_H'].append(getattr(r, 'mu_eL_H', 0.0))
            data['mu_B_Q'].append(getattr(r, 'mu_B_Q', 0.0))
            data['mu_C_Q'].append(getattr(r, 'mu_C_Q', 0.0))
            data['mu_S_Q'].append(getattr(r, 'mu_S_Q', 0.0))
            data['n_B_Q'].append(getattr(r, 'n_B_Q', 0.0))
            data['n_C_Q'].append(getattr(r, 'n_C_Q', 0.0))
            data['n_S_Q'].append(getattr(r, 'n_S_Q', 0.0))
            data['mu_eL_Q'].append(getattr(r, 'mu_eL_Q', 0.0))
            data['mu_eG'].append(getattr(r, 'mu_eG', 0.0))
    
    return {k: np.array(v) for k, v in data.items()}


def results_to_dict_complete(results: list, eta_override: float = None) -> dict:
    """
    Convert list of results to dict with COMPLETE thermodynamic quantities.
    
    Handles all three result types:
    - MixedPhaseResult: has both H and Q phase quantities
    - ZLEOSResult (pure H): H quantities only, Q = 0
    - VMITEOSResult (pure Q): Q quantities only, H = 0
    
    Complete file contains:
    - Input: n_B, T, eta, converged, error
    - Hadronic phase: mu_p_H, mu_n_H, n_p_H, n_n_H, P_H, e_H, s_H, f_H
    - Quark phase: mu_u_Q, mu_d_Q, mu_s_Q, n_u_Q, n_d_Q, n_s_Q, P_Q, e_Q, s_Q, f_Q
    - Electrons: mu_eL_H, mu_eL_Q, mu_eG, n_eL_H, n_eL_Q, n_eG, P_eL_*, etc.
    - Photons: P_gamma, e_gamma, s_gamma
    - Total quantities: P_total, e_total, s_total, f_total
    - Particle fractions: Y_p_H, Y_n_H, Y_u_Q, Y_d_Q, Y_s_Q, Y_e_tot, etc.
    """
    if not results:
        return {}
    
    # Initialize dict with simplified columns
    data = {
        # Input quantities
        'n_B': [], 'T': [], 'eta': [],
        
        # Convergence info
        'converged': [], 'error': [],
        
        # Hadronic phase - particle chemical potentials and densities
        'mu_p_H': [], 'mu_n_H': [], 'n_p_H': [], 'n_n_H': [],
        
        # Quark phase - particle chemical potentials and densities
        'mu_u_Q': [], 'mu_d_Q': [], 'mu_s_Q': [],
        'n_u_Q': [], 'n_d_Q': [], 'n_s_Q': [],
        
        # Electrons - chemical potentials and densities only
        'mu_eL_H': [], 'mu_eL_Q': [], 'mu_eG': [],
        'n_eL_H': [], 'n_eL_Q': [], 'n_eG': [],
        
        # Total quantities only
        'P_total': [], 'e_total': [], 's_total': [], 'f_total': [],
        'n_e_tot': [],
        
        # Per-phase particle fractions
        'Y_p_H': [], 'Y_n_H': [],
        'Y_u_Q': [], 'Y_d_Q': [], 'Y_s_Q': [],
        'Y_C_H': [], 'Y_C_Q': [], 'Y_S_Q': [],
        
        # Total particle fractions
        'Y_p_tot': [], 'Y_n_tot': [],
        'Y_u_tot': [], 'Y_d_tot': [], 'Y_s_tot': [],
        'Y_e_tot': [],
        'Y_B_tot': [], 'Y_C_tot': [], 'Y_S_tot': [],
    }
    
    for r in results:
        # Determine phase type
        is_mixed = hasattr(r, 'mu_p_H')  # MixedPhaseResult has mu_p_H
        is_pure_Q = hasattr(r, 'mu_u') and not hasattr(r, 'mu_p_H')  # VMITEOSResult
        is_pure_H = hasattr(r, 'mu_p') and not hasattr(r, 'mu_u')  # ZLEOSResult
        
        # Input quantities
        data['n_B'].append(r.n_B)
        data['T'].append(r.T)
        if eta_override is not None:
            data['eta'].append(eta_override)
        else:
            data['eta'].append(getattr(r, 'eta', 0.0))
        
        # Convergence info
        data['converged'].append(1 if getattr(r, 'converged', True) else 0)
        data['error'].append(getattr(r, 'error', 0.0))
        
        if is_mixed:
            # Mixed phase: all quantities from MixedPhaseResult
            # Hadronic phase
            data['mu_p_H'].append(r.mu_p_H)
            data['mu_n_H'].append(r.mu_n_H)
            data['n_p_H'].append(r.n_p_H)
            data['n_n_H'].append(r.n_n_H)
            # Quark phase
            data['mu_u_Q'].append(r.mu_u_Q)
            data['mu_d_Q'].append(r.mu_d_Q)
            data['mu_s_Q'].append(r.mu_s_Q)
            data['n_u_Q'].append(r.n_u_Q)
            data['n_d_Q'].append(r.n_d_Q)
            data['n_s_Q'].append(r.n_s_Q)
            # Electrons - mu and n only
            data['mu_eL_H'].append(r.mu_eL_H)
            data['mu_eL_Q'].append(r.mu_eL_Q)
            data['mu_eG'].append(r.mu_eG)
            data['n_eL_H'].append(r.n_eL_H)
            data['n_eL_Q'].append(r.n_eL_Q)
            data['n_eG'].append(r.n_eG)
            # Totals only
            data['P_total'].append(r.P_total)
            data['e_total'].append(r.e_total)
            data['s_total'].append(r.s_total)
            data['f_total'].append(r.f_total)
            data['n_e_tot'].append(r.n_e_tot)
            # Fractions
            data['Y_p_H'].append(r.Y_p_H)
            data['Y_n_H'].append(r.Y_n_H)
            data['Y_u_Q'].append(r.Y_u_Q)
            data['Y_d_Q'].append(r.Y_d_Q)
            data['Y_s_Q'].append(r.Y_s_Q)
            data['Y_C_H'].append(r.Y_C_H)
            data['Y_C_Q'].append(r.Y_C_Q)
            data['Y_S_Q'].append(r.Y_S_Q)
            data['Y_p_tot'].append(r.Y_p_tot)
            data['Y_n_tot'].append(r.Y_n_tot)
            data['Y_u_tot'].append(r.Y_u_tot)
            data['Y_d_tot'].append(r.Y_d_tot)
            data['Y_s_tot'].append(r.Y_s_tot)
            data['Y_e_tot'].append(r.Y_e_tot)
            data['Y_B_tot'].append(r.Y_B_tot)
            data['Y_C_tot'].append(r.Y_C_tot)
            data['Y_S_tot'].append(r.Y_S_tot)
            
        elif is_pure_H:
            # Pure H phase (ZLEOSResult): H quantities from result, Q = 0
            data['mu_p_H'].append(r.mu_p)
            data['mu_n_H'].append(r.mu_n)
            data['n_p_H'].append(r.n_p)
            data['n_n_H'].append(r.n_n)
            # Quark phase = 0
            data['mu_u_Q'].append(0.0)
            data['mu_d_Q'].append(0.0)
            data['mu_s_Q'].append(0.0)
            data['n_u_Q'].append(0.0)
            data['n_d_Q'].append(0.0)
            data['n_s_Q'].append(0.0)
            # Electrons
            mu_e = getattr(r, 'mu_e', 0.0)
            n_e = getattr(r, 'n_e', 0.0)
            data['mu_eL_H'].append(mu_e)
            data['mu_eL_Q'].append(0.0)
            data['mu_eG'].append(mu_e)
            data['n_eL_H'].append(n_e)
            data['n_eL_Q'].append(0.0)
            data['n_eG'].append(n_e)
            # Totals
            f_H = r.e_total - r.T * r.s_total if r.T > 0 else r.e_total
            data['P_total'].append(r.P_total)
            data['e_total'].append(r.e_total)
            data['s_total'].append(r.s_total)
            data['f_total'].append(f_H)
            data['n_e_tot'].append(n_e)
            # Fractions
            Y_p = r.n_p / r.n_B if r.n_B > 0 else 0.0
            Y_n = r.n_n / r.n_B if r.n_B > 0 else 0.0
            Y_e = n_e / r.n_B if r.n_B > 0 else 0.0
            data['Y_p_H'].append(Y_p)
            data['Y_n_H'].append(Y_n)
            data['Y_u_Q'].append(0.0)
            data['Y_d_Q'].append(0.0)
            data['Y_s_Q'].append(0.0)
            data['Y_C_H'].append(Y_p)
            data['Y_C_Q'].append(0.0)
            data['Y_S_Q'].append(0.0)
            data['Y_p_tot'].append(Y_p)
            data['Y_n_tot'].append(Y_n)
            data['Y_u_tot'].append(0.0)
            data['Y_d_tot'].append(0.0)
            data['Y_s_tot'].append(0.0)
            data['Y_e_tot'].append(Y_e)
            data['Y_B_tot'].append(1.0)
            data['Y_C_tot'].append(Y_p)
            data['Y_S_tot'].append(0.0)
            
        elif is_pure_Q:
            # Pure Q phase (VMITEOSResult): Q quantities from result, H = 0
            data['mu_p_H'].append(0.0)
            data['mu_n_H'].append(0.0)
            data['n_p_H'].append(0.0)
            data['n_n_H'].append(0.0)
            # Quark phase from VMITEOSResult
            data['mu_u_Q'].append(r.mu_u)
            data['mu_d_Q'].append(r.mu_d)
            data['mu_s_Q'].append(r.mu_s)
            data['n_u_Q'].append(r.n_u)
            data['n_d_Q'].append(r.n_d)
            data['n_s_Q'].append(r.n_s)
            # Electrons
            mu_e = getattr(r, 'mu_e', 0.0)
            n_e = getattr(r, 'n_e', 0.0)
            data['mu_eL_H'].append(0.0)
            data['mu_eL_Q'].append(mu_e)
            data['mu_eG'].append(mu_e)
            data['n_eL_H'].append(0.0)
            data['n_eL_Q'].append(n_e)
            data['n_eG'].append(n_e)
            # Totals
            f_Q = r.e_total - r.T * r.s_total if r.T > 0 else r.e_total
            data['P_total'].append(r.P_total)
            data['e_total'].append(r.e_total)
            data['s_total'].append(r.s_total)
            data['f_total'].append(f_Q)
            data['n_e_tot'].append(n_e)
            # Fractions
            Y_u = r.n_u / r.n_B if r.n_B > 0 else 0.0
            Y_d = r.n_d / r.n_B if r.n_B > 0 else 0.0
            Y_s = r.n_s / r.n_B if r.n_B > 0 else 0.0
            Y_e = n_e / r.n_B if r.n_B > 0 else 0.0
            Y_C_Q = (2.0/3.0) * Y_u - (1.0/3.0) * Y_d - (1.0/3.0) * Y_s
            Y_S_Q = -Y_s
            data['Y_p_H'].append(0.0)
            data['Y_n_H'].append(0.0)
            data['Y_u_Q'].append(Y_u)
            data['Y_d_Q'].append(Y_d)
            data['Y_s_Q'].append(Y_s)
            data['Y_C_H'].append(0.0)
            data['Y_C_Q'].append(Y_C_Q)
            data['Y_S_Q'].append(Y_S_Q)
            data['Y_p_tot'].append(0.0)
            data['Y_n_tot'].append(0.0)
            data['Y_u_tot'].append(Y_u)
            data['Y_d_tot'].append(Y_d)
            data['Y_s_tot'].append(Y_s)
            data['Y_e_tot'].append(Y_e)
            data['Y_B_tot'].append(1.0)
            data['Y_C_tot'].append(Y_C_Q)
            data['Y_S_tot'].append(Y_S_Q)
    
    # Convert to numpy arrays
    return {k: np.array(v) for k, v in data.items()}


def save_table_full(data: dict, filename: str, header: str = ""):
    """Save comprehensive table data to .dat file with all thermodynamic quantities."""
    if not data:
        return
    
    keys = list(data.keys())
    n_rows = len(data[keys[0]])
    
    with open(filename, 'w') as f:
        # Write header comments
        if header:
            for line in header.split('\n'):
                f.write(f"# {line}\n")
        f.write(f"# Columns: {len(keys)}\n")
        f.write(f"# Rows: {n_rows}\n")
        
        # Write column header
        header_line = "#"
        for k in keys:
            header_line += f"{k:>15}"
        f.write(header_line + "\n")
        
        # Write data rows
        for i in range(n_rows):
            row_str = ""
            for k in keys:
                val = data[k][i]
                if isinstance(val, (int, np.integer)):
                    row_str += f"{val:>15d}"
                else:
                    row_str += f"{val:>15.6e}"
            f.write(row_str + "\n")
    
    print(f"  Saved: {filename} ({len(keys)} columns, {n_rows} rows)")


# =============================================================================
# PURE PHASE TABLE CACHING
# =============================================================================

def get_pure_table_filename(phase: str, eq_mode: str, vmit_params, output_dir: str) -> str:
    """Get filename for pure phase table cache file."""
    B4 = int(vmit_params.B4)
    a = vmit_params.a
    return os.path.join(output_dir, f"pure_{phase}_table_{eq_mode}_B{B4}_a{a}.pkl")


def save_pure_table(table: dict, phase: str, eq_mode: str, vmit_params, output_dir: str, n_B_values: np.ndarray):
    """Save pure phase table to pickle file."""
    import pickle
    
    filename = get_pure_table_filename(phase, eq_mode, vmit_params, output_dir)
    
    # Store table along with metadata for verification
    data = {
        'table': table,
        'n_B_values': n_B_values,
        'B4': vmit_params.B4,
        'a': vmit_params.a,
        'eq_mode': eq_mode
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"    Saved pure {phase} table: {os.path.basename(filename)}")


def load_pure_table(phase: str, eq_mode: str, vmit_params, output_dir: str, n_B_values: np.ndarray) -> dict:
    """
    Load pure phase table from pickle file if exists and matches parameters.
    
    Returns None if file doesn't exist or parameters don't match.
    """
    import pickle
    
    filename = get_pure_table_filename(phase, eq_mode, vmit_params, output_dir)
    
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Verify parameters match
        if (data['B4'] != vmit_params.B4 or 
            data['a'] != vmit_params.a or 
            data['eq_mode'] != eq_mode or
            len(data['n_B_values']) != len(n_B_values)):
            print(f"    Pure {phase} table parameters don't match, recomputing...")
            return None
        
        print(f"    Loaded pure {phase} table: {os.path.basename(filename)} ({len(data['table'])} points)")
        return data['table']
    
    except Exception as e:
        print(f"    Failed to load pure {phase} table: {e}")
        return None


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_orchestrator():
    """
    Main orchestrator function that coordinates all EOS computation steps.
    
    Workflow:
        1. Initialize parameters (ZL hadronic, vMIT quark, grids)
        2. Compute pure phase tables (H and Q)
        3. Compute phase boundaries for each η
        4. Generate unified EOS tables (H → Mixed → Q) for each η
        5. Save results to files
    
    Supports two equilibrium modes:
        - "beta": Beta equilibrium with electrons (charge neutrality)
        - "fixed_yc": Fixed charge fraction Y_C (loops over Y_C values)
    """
    
    print("=" * 70)
    print("ZL + vMIT Hybrid EOS Orchestrator")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    # =========================================================================
    # STEP 1: INITIALIZE PARAMETERS
    # =========================================================================
    # Load model parameters and create computational grids
    
    print("\n" + "=" * 70)
    print("STEP 1: Initializing Parameters")
    print("=" * 70)
    
    from zl_parameters import get_zl_default
    from vmit_parameters import VMITParams
    
    # Model parameters
    zl_params = get_zl_default()
    vmit_params = VMITParams(name=f"vMIT_B{int(B4)}_a{a}", B4=B4, a=a)
    
    # Computational grids
    n_B_values = np.linspace(n_B_min, n_B_max, n_B_steps)
    T_array = np.array(T_values)
    
    # Determine composition values based on equilibrium mode
    if EQUILIBRIUM_MODE == "fixed_yc":
        composition_values = Y_C_values
        composition_name = "Y_C"
    elif EQUILIBRIUM_MODE == "trapped":
        composition_values = Y_L_values
        composition_name = "Y_L"
    else:  # beta equilibrium
        composition_values = [None]
        composition_name = None
    
    # Print configuration summary
    print(f"  ZL parameters: {zl_params.name}")
    print(f"  vMIT parameters: B^(1/4) = {B4} MeV, a = {a} fm²")
    print(f"  Density grid: [{n_B_min:.4f}, {n_B_max:.4f}] fm⁻³ ({n_B_steps} points)")
    print(f"  Temperature grid: {len(T_array)} values from {T_array.min():.1f} to {T_array.max():.1f} MeV")
    print(f"  η values: {eta_values}")
    print(f"  Equilibrium mode: {EQUILIBRIUM_MODE}")
    if composition_name:
        print(f"  {composition_name} values: {composition_values}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Output directory: {OUTPUT_DIR}")
    
    # =========================================================================
    # MODE-SPECIFIC PROCESSING
    # =========================================================================
    
    if EQUILIBRIUM_MODE == "fixed_yc":
        _run_fixed_yc_mode(
            composition_values, T_array, n_B_values, eta_values,
            zl_params, vmit_params
        )
    else:
        _run_beta_equilibrium_mode(
            T_array, n_B_values, eta_values,
            zl_params, vmit_params
        )
    
    # =========================================================================
    # STEP 6: ISENTROPIC TABLES
    # =========================================================================
    # Compute T(n_B) curves for fixed entropy values s/n_B = S
    
    print("\n" + "=" * 70)
    print("STEP 6: Computing Isentropic Tables")
    print("=" * 70)
    
    from zlvmit_isentropic import (
        create_entropy_interpolator_from_file,
        compute_isentropes,
        save_isentropes_to_file
    )
    from pathlib import Path
    
    # Entropy values to compute (in units of k_B)
    S_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    output_path = Path(OUTPUT_DIR)
    
    for eta in eta_values:
        if EQUILIBRIUM_MODE == "beta":
            # Beta equilibrium mode
            table_file = output_path / f"table_hybrid_eta{eta:.2f}_B{int(B4)}_a{a}_complete.dat"
            output_file = output_path / f"isentropes_eta{eta:.2f}_B{int(B4)}_a{a}.dat"
            header = f"ZL+vMIT isentropes, eta={eta}, B^1/4={B4} MeV, a={a} fm^2"
        else:
            # For fixed_yc mode, use the first Y_C (can be extended)
            Y_C = composition_values[0]
            table_file = output_path / f"table_hybrid_eta{eta:.2f}_YC{Y_C:.2f}_B{int(B4)}_a{a}_complete.dat"
            output_file = output_path / f"isentropes_eta{eta:.2f}_YC{Y_C:.2f}_B{int(B4)}_a{a}.dat"
            header = f"ZL+vMIT isentropes, eta={eta}, Y_C={Y_C}, B^1/4={B4} MeV, a={a} fm^2"
        
        if table_file.exists():
            try:
                print(f"  Computing isentropes for eta={eta}...")
                interpolator, nB_grid, T_grid = create_entropy_interpolator_from_file(table_file)
                
                isentropes = compute_isentropes(
                    nB_grid, 
                    S_values, 
                    interpolator,
                    T_bounds=(T_array.min(), T_array.max())
                )
                
                save_isentropes_to_file(isentropes, nB_grid, output_file, header)
                
                # Report coverage
                for S, T_arr in isentropes.items():
                    valid = ~np.isnan(T_arr)
                    print(f"    S={S:.1f}: {np.sum(valid)}/{len(T_arr)} valid points")
            except Exception as e:
                print(f"  [WARNING] Could not compute isentropes for eta={eta}: {e}")
        else:
            print(f"  [SKIP] Table not found: {table_file}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("ORCHESTRATION COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)


# =============================================================================
# FIXED Y_C MODE
# =============================================================================

def _run_fixed_yc_mode(composition_values, T_array, n_B_values, eta_values,
                       zl_params, vmit_params):
    """
    Run the orchestrator in fixed Y_C mode.
    
    For each Y_C value:
        1. Compute pure H and Q phase tables
        2. Find phase boundaries for each η
        3. Generate unified tables (pure + mixed phases)
        4. Save results
    """
    from zlvmit_mixed_phase_eos import get_or_compute_boundaries, generate_unified_table
    from zl_compute_tables import ZLTableSettings, compute_zl_table
    from vmit_compute_tables import VMITTableSettings, compute_vmit_table
    
    all_results = {}
    
    for i_yc, Y_C in enumerate(composition_values):
        yc_start_time = time.time()
        
        print("\n" + "=" * 70)
        print(f"Processing Y_C = {Y_C} [{i_yc+1}/{len(composition_values)}]")
        print("=" * 70)
        
        # ---------------------------------------------------------------------
        # STEP 2: COMPUTE PURE PHASE TABLES
        # ---------------------------------------------------------------------
        # Solve pure H (hadronic) and pure Q (quark) phases for all (n_B, T).
        # These are used outside the mixed phase region and for initial guesses.
        
        print(f"\n  [Step 2] Computing pure phase tables...")
        
        t_start = time.time()
        eq_mode_str = f"fixed_yc_YC{Y_C}"
        
        # --- Pure H Phase ---
        H_table_yc = None
        if not FORCE_RECOMPUTE_PURE_PHASES:
            H_table_yc = load_pure_table("H", eq_mode_str, vmit_params, OUTPUT_DIR, n_B_values)
            
        if H_table_yc is None:
            # Compute pure H table using existing infrastructure
            zl_settings = ZLTableSettings(
                n_B_values=n_B_values,
                T_values=T_array,
                equilibrium='fixed_yc',
                Y_C_values=[Y_C],
                params=zl_params,
                print_results=VERBOSE,
                print_timing=VERBOSE,
                save_to_file=False
            )
            H_results_by_T = compute_zl_table(zl_settings)
            
            # Convert to (n_B, T) lookup format for boundary finder
            H_table_yc = {}
            for grid_key, results in H_results_by_T.items():
                T = grid_key[0]  # (T, Y_C) tuple
                for i, r in enumerate(results):
                    if r.converged:
                        H_table_yc[(n_B_values[i], T)] = r
            
            # Save to cache
            save_pure_table(H_table_yc, "H", eq_mode_str, vmit_params, OUTPUT_DIR, n_B_values)
        
        # --- Pure Q Phase ---
        Q_table_yc = None
        if not FORCE_RECOMPUTE_PURE_PHASES:
            Q_table_yc = load_pure_table("Q", eq_mode_str, vmit_params, OUTPUT_DIR, n_B_values)
            
        if Q_table_yc is None:
            # Compute pure Q table using existing infrastructure
            vmit_settings = VMITTableSettings(
                n_B_values=n_B_values,
                T_values=T_array,
                equilibrium='fixed_yc',
                Y_C_values=[Y_C],
                params=vmit_params,
                print_results=VERBOSE,
                print_timing=VERBOSE,
                save_to_file=False
            )
            Q_results_by_T = compute_vmit_table(vmit_settings)
            
            # Convert to (n_B, T) lookup format
            Q_table_yc = {}
            for grid_key, results in Q_results_by_T.items():
                T = grid_key[0]  # (T, Y_C) tuple
                for i, r in enumerate(results):
                    if r.converged:
                        Q_table_yc[(n_B_values[i], T)] = r
            
            # Save to cache
            save_pure_table(Q_table_yc, "Q", eq_mode_str, vmit_params, OUTPUT_DIR, n_B_values)

        t_elapsed = time.time() - t_start
        n_total = len(n_B_values) * len(T_array)
        print(f"    Pure H: {len(H_table_yc)}/{n_total} points")
        print(f"    Pure Q: {len(Q_table_yc)}/{n_total} points")
        print(f"    Time: {t_elapsed:.1f}s")
        
        # ---------------------------------------------------------------------
        # STEP 3: COMPUTE PHASE BOUNDARIES
        # ---------------------------------------------------------------------
        # Find n_B_onset (χ=0) and n_B_offset (χ=1) for each (T, η).
        # Uses fixed-chi solvers with bidirectional search.
        
        print(f"\n  [Step 3] Computing phase boundaries...")
        
        all_boundaries_yc = {'full_results': {}}
        
        for eta in eta_values:
            print(f"    η = {eta:.2f}: ", end="", flush=True)
            t_start = time.time()
            
            try:
                boundary_results = get_or_compute_boundaries(
                    eta, T_array,
                    output_dir=BOUNDARY_DIR,
                    force_recompute=FORCE_RECOMPUTE_BOUNDARIES,
                    zl_params=zl_params, vmit_params=vmit_params,
                    verbose=VERBOSE,
                    debug=False,
                    H_table_lookup=H_table_yc,
                    Q_table_lookup=Q_table_yc,
                    eq_mode="fixed_yc", Y_C=Y_C
                )
                
                
                # Store converged boundaries
                boundaries = []
                boundary_results_yc = {}
                
                for b in boundary_results:
                    if b.converged_onset and b.converged_offset:
                        boundaries.append({
                            'T': b.T,
                            'n_B_onset': b.n_B_onset,
                            'n_B_offset': b.n_B_offset,
                        })
                        boundary_results_yc[b.T] = b
                
                all_boundaries_yc[eta] = boundaries
                all_boundaries_yc['full_results'][eta] = boundary_results_yc
                
                t_elapsed = time.time() - t_start
                print(f"{len(boundaries)}/{len(T_array)} converged ({t_elapsed:.1f}s)")
                
            except Exception as e:
                print(f"failed: {e}")
                all_boundaries_yc[eta] = None
        
        # Save phase boundaries to file
        _save_boundaries_file(all_boundaries_yc, Y_C, eta_values, vmit_params)
        
        # ---------------------------------------------------------------------
        # STEP 4: GENERATE UNIFIED TABLES
        # ---------------------------------------------------------------------
        # For each (n_B, T), determine the phase and solve:
        #   n_B < n_onset: Pure H phase
        #   n_onset ≤ n_B ≤ n_offset: Mixed phase
        #   n_B > n_offset: Pure Q phase
        
        print(f"\n  [Step 4] Generating unified tables...")
        
        for eta in eta_values:
            if all_boundaries_yc.get(eta) is None:
                continue
            
            print(f"    η = {eta:.2f}: ", end="", flush=True)
            t_start = time.time()
            
            # Convert boundaries format: list of dicts -> dict by T
            raw_boundaries = all_boundaries_yc.get(eta, [])
            onset_solutions = all_boundaries_yc.get('full_results', {}).get(eta, {})
            boundaries_by_T = {}
            for b in raw_boundaries:
                T = b['T']
                onset_guess = None
                if T in onset_solutions:
                    onset_result = onset_solutions[T]
                    # Use new function to create onset_guess from boundary fields
                    onset_guess = boundary_result_to_onset_guess(onset_result, eta)
                boundaries_by_T[T] = {
                    'n_onset': b['n_B_onset'],
                    'n_offset': b['n_B_offset'],
                    'onset_guess': onset_guess
                }
            
            results = generate_unified_table(
                n_B_values, T_array, eta,
                zl_params, vmit_params,
                H_table_yc, Q_table_yc,
                boundaries=boundaries_by_T,
                verbose=VERBOSE,
                eq_mode="fixed_yc", Y_C=Y_C
            )
            
            t_elapsed = time.time() - t_start
            all_results[(eta, Y_C)] = results
            
            n_mixed = sum(1 for r in results if hasattr(r, 'chi') and 0 < r.chi < 1)
            print(f"{len(results)} points, {n_mixed} mixed ({t_elapsed:.1f}s)")
        
        # ---------------------------------------------------------------------
        # STEP 5: SAVE RESULTS
        # ---------------------------------------------------------------------
        print(f"\n  [Step 5] Saving tables...")
        
        for (eta, yc), results in all_results.items():
            if yc != Y_C or not results:
                continue
            _save_eos_tables(results, eta, OUTPUT_DIR, vmit_params, Y_C=yc)
        
        yc_elapsed = time.time() - yc_start_time
        print(f"\n  Total time for Y_C = {Y_C}: {yc_elapsed:.1f}s")


# =============================================================================
# BETA EQUILIBRIUM MODE
# =============================================================================

def _run_beta_equilibrium_mode(T_array, n_B_values, eta_values,
                                zl_params, vmit_params):
    """
    Run the orchestrator in beta equilibrium mode.
    
    Steps:
        1. Compute pure H and Q phase tables
        2. Load or compute phase boundaries for each η (using pure phases as guesses)
        3. Generate unified tables for each η
        4. Save results
    """
    from zlvmit_mixed_phase_eos import get_or_compute_boundaries, generate_unified_table
    from zl_compute_tables import ZLTableSettings, compute_zl_table
    from vmit_compute_tables import VMITTableSettings, compute_vmit_table
    
    # -------------------------------------------------------------------------
    # STEP 2: COMPUTE PURE PHASE TABLES
    # -------------------------------------------------------------------------
    # Compute first so we can use them as initial guesses for boundary finding
    print("\n" + "=" * 70)
    print("STEP 2: Loading/Computing Pure Phase Tables")
    print("=" * 70)
    
    t_start = time.time()
    eq_mode_cache = "beta"  # Use simplified name for cache files
    
    # Try to load pure H table from cache
    H_table = None
    if not FORCE_RECOMPUTE_PURE_PHASES:
        H_table = load_pure_table("H", eq_mode_cache, vmit_params, OUTPUT_DIR, n_B_values)
    
    if H_table is None:
        # Compute pure H table using existing infrastructure
        print("  Computing pure H table...")
        zl_settings = ZLTableSettings(
            n_B_values=n_B_values,
            T_values=T_array,
            equilibrium='beta_eq',
            params=zl_params,
            print_results=False,
            print_timing=VERBOSE,
            save_to_file=False
        )
        H_results_by_T = compute_zl_table(zl_settings)
        
        # Convert to (n_B, T) lookup format for boundary finder
        H_table = {}
        for (T,), results in H_results_by_T.items():
            for i, r in enumerate(results):
                if r.converged:
                    H_table[(n_B_values[i], T)] = r
        
        # Save to cache
        save_pure_table(H_table, "H", eq_mode_cache, vmit_params, OUTPUT_DIR, n_B_values)
    
    # Try to load pure Q table from cache
    Q_table = None
    if not FORCE_RECOMPUTE_PURE_PHASES:
        Q_table = load_pure_table("Q", eq_mode_cache, vmit_params, OUTPUT_DIR, n_B_values)
    
    if Q_table is None:
        # Compute pure Q table using existing infrastructure
        print("  Computing pure Q table...")
        vmit_settings = VMITTableSettings(
            n_B_values=n_B_values,
            T_values=T_array,
            equilibrium='beta_eq',
            params=vmit_params,
            print_results=False,
            print_timing=VERBOSE,
            save_to_file=False
        )
        Q_results_by_T = compute_vmit_table(vmit_settings)
        
        # Convert to (n_B, T) lookup format
        Q_table = {}
        for (T,), results in Q_results_by_T.items():
            for i, r in enumerate(results):
                if r.converged:
                    Q_table[(n_B_values[i], T)] = r
        
        # Save to cache
        save_pure_table(Q_table, "Q", eq_mode_cache, vmit_params, OUTPUT_DIR, n_B_values)
    
    print(f"  Pure H: {len(H_table)} converged points")
    print(f"  Pure Q: {len(Q_table)} converged points")
    print(f"  Time: {time.time() - t_start:.2f}s")
    
    # -------------------------------------------------------------------------
    # STEP 3: LOAD OR COMPUTE PHASE BOUNDARIES
    # -------------------------------------------------------------------------
    # Use pure H table as initial guesses for better convergence
    print("\n" + "=" * 70)
    print("STEP 3: Loading/Computing Phase Boundaries")
    print("=" * 70)
    
    all_boundaries = {}
    
    for i, eta in enumerate(eta_values):
        print(f"\n  [{i+1}/{len(eta_values)}] η = {eta:.2f}:")
        t_start = time.time()
        
        # get_or_compute_boundaries handles caching internally
        boundary_results = get_or_compute_boundaries(
            eta, T_array,
            output_dir=BOUNDARY_DIR,
            force_recompute=FORCE_RECOMPUTE_BOUNDARIES,
            zl_params=zl_params,
            vmit_params=vmit_params,
            verbose=VERBOSE,
            debug=True,
            H_table_lookup=H_table,
            Q_table_lookup=Q_table,
            eq_mode="beta"
        )
        
        # Convert boundary_results to dict format expected by generate_unified_table
        boundaries_by_T = {}
        for b in boundary_results:
            if b.converged_onset and b.converged_offset:
                # Use new function to create onset_guess from boundary fields
                onset_guess = boundary_result_to_onset_guess(b, eta)
                boundaries_by_T[b.T] = {
                    'n_onset': b.n_B_onset,
                    'n_offset': b.n_B_offset,
                    'onset_guess': onset_guess
                }
        
        all_boundaries[eta] = boundaries_by_T
        
        if boundaries_by_T:
            print(f"    {len(boundaries_by_T)} T points")
        print(f"    Time: {time.time() - t_start:.2f}s")
    
    # -------------------------------------------------------------------------
    # STEP 4: GENERATE UNIFIED TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Generating Unified EOS Tables")
    print("=" * 70)
    
    all_results = {}
    
    for i, eta in enumerate(eta_values):
        print(f"\n  [{i+1}/{len(eta_values)}] η = {eta:.2f}:")
        
        boundaries = all_boundaries[eta]
        if not boundaries:
            print(f"    Skipping: No phase boundaries")
            continue
        
        t_start = time.time()
        results = generate_unified_table(
            n_B_values, T_array, eta,
            zl_params, vmit_params,
            H_table, Q_table,
            boundaries=boundaries,
            verbose=VERBOSE,
            eq_mode="beta"
        )
        
        if results:
            all_results[eta] = results
            n_converged = sum(1 for r in results if r.converged)
            print(f"    {len(results)} points, {n_converged/len(results)*100}% converged")
            print(f"    Time: {time.time() - t_start:.2f}s")
    
    # -------------------------------------------------------------------------
    # STEP 5: SAVE RESULTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Saving Results")
    print("=" * 70)
    
    for eta, results in all_results.items():
        if results:
            _save_eos_tables(results, eta, OUTPUT_DIR, vmit_params)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _save_boundaries_file(boundaries, Y_C, eta_values, vmit_params):
    """Save phase boundaries to a data file."""
    filename = f"boundaries_YC{Y_C:.2f}_B{int(vmit_params.B4)}_a{vmit_params.a}.dat"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"# Phase Boundaries for Y_C = {Y_C}\n")
        f.write(f"# B^(1/4) = {vmit_params.B4} MeV, a = {vmit_params.a} fm²\n")
        f.write("# Columns: eta T n_B_onset n_B_offset conv mu_p_H mu_n_H mu_u_Q mu_d_Q mu_s_Q mu_eL_H mu_eL_Q mu_eG n_p_H n_n_H n_u_Q n_d_Q n_s_Q\n")
        
        full_results = boundaries.get('full_results', {})
        for eta in eta_values:
            if eta not in full_results:
                continue
            for T, b in sorted(full_results[eta].items()):
                conv = 1 if (b.converged_onset and b.converged_offset) else 0
                f.write(f"{eta:.2f} {b.T:.6f} {b.n_B_onset:.6f} {b.n_B_offset:.6f} ")
                f.write(f"{conv} {b.mu_p_H_onset:.6f} {b.mu_n_H_onset:.6f} ")
                f.write(f"{b.mu_u_Q_onset:.6f} {b.mu_d_Q_onset:.6f} {b.mu_s_Q_onset:.6f} ")
                f.write(f"{b.mu_eL_H_onset:.6f} {b.mu_eL_Q_onset:.6f} {b.mu_eG_onset:.6f} ")
                f.write(f"{b.n_p_H_onset:.6f} {b.n_n_H_onset:.6f} ")
                f.write(f"{b.n_u_Q_onset:.6f} {b.n_d_Q_onset:.6f} {b.n_s_Q_onset:.6f}\n")
    
    print(f"    Saved: {filename}")


def _save_eos_tables(results, eta, output_dir, vmit_params, Y_C=None):
    """Save EOS tables (primary and complete) to data files."""
    if Y_C is not None:
        base_name = f"table_hybrid_eta{eta:.2f}_YC{Y_C:.2f}_B{int(vmit_params.B4)}_a{vmit_params.a}"
    else:
        base_name = f"table_hybrid_eta{eta:.2f}_B{int(vmit_params.B4)}_a{vmit_params.a}"
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Primary file
    data_primary = results_to_dict_primary(results, eta_override=eta)
    filepath_primary = os.path.join(output_dir, f"{base_name}_primary.dat")
    header = f"ZL+vMIT Hybrid EOS - PRIMARY OUTPUT\nη={eta:.2f}, B^(1/4)={vmit_params.B4} MeV\nGenerated: {timestamp}"
    save_table_full(data_primary, filepath_primary, header)
    
    # Complete file
    data_complete = results_to_dict_complete(results)
    filepath_complete = os.path.join(output_dir, f"{base_name}_complete.dat")
    header = f"ZL+vMIT Hybrid EOS - COMPLETE OUTPUT\nη={eta:.2f}, B^(1/4)={vmit_params.B4} MeV\nGenerated: {timestamp}"
    save_table_full(data_complete, filepath_complete, header)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_orchestrator()

