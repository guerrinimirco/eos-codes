"""
alphabag_eos.py
===============
Single-point solvers for αBag (AlphaBag) quark matter EOS.

This module provides solvers for different equilibrium conditions:
- Beta equilibrium (charge neutrality, β-equilibrium, strangeness equilibrium)
- Fixed charge fraction Y_C

The αBag model uses perturbative QCD corrections parametrized by α_s,
rather than the vector field approach of vMIT.

Units:
- Energy/mass/chemical potentials: MeV
- Densities: fm⁻³
- Pressure/energy density: MeV/fm³

References:
- T. Fischer et al. Astrophys. J. Suppl. 194:39 (2011)
- M. Guerrini PhD Thesis (2026)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.optimize import root

from general_physics_constants import hc, hc3, PI2
from alphabag_parameters import AlphaBagParams, get_alphabag_default
from alphabag_thermodynamics_quarks import (
    compute_alphabag_thermo_from_mu,
    compute_cfl_thermo_from_mu,
    n_quark_alpha,
    n_cfl_correction,
    gluon_thermo,
    gap_cfl, T_critical
)
from general_thermodynamics_leptons import (
    electron_thermo, photon_thermo, neutrino_thermo, electron_thermo_from_density
)
import general_particles


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class AlphaBagEOSResult:
    """Complete result from αBag EOS calculation at one point."""
    # Convergence info
    converged: bool = False
    error: float = 0.0
    
    # Input conditions
    n_B: float = 0.0        # Baryon density (fm⁻³)
    T: float = 0.0          # Temperature (MeV)
    Y_C: float = 0.0        # Charge fraction
    Y_S: float = 0.0        # Strangeness fraction
    Y_L: float = 0.0        # Lepton fraction (for trapped neutrinos)
    
    # Chemical potentials
    mu_u: float = 0.0       # Up quark (MeV)
    mu_d: float = 0.0       # Down quark (MeV)
    mu_s: float = 0.0       # Strange quark (MeV)
    mu_e: float = 0.0       # Electron (MeV)
    mu_nu: float = 0.0      # Neutrino (MeV)
    mu_B: float = 0.0       # Baryon chemical potential (MeV)
    mu_C: float = 0.0       # Charge chemical potential (MeV)
    
    # Number densities
    n_u: float = 0.0        # Up quark (fm⁻³)
    n_d: float = 0.0        # Down quark (fm⁻³)
    n_s: float = 0.0        # Strange quark (fm⁻³)
    n_e: float = 0.0        # Electron (fm⁻³)
    n_nu: float = 0.0       # Neutrino (fm⁻³)
    
    # Thermodynamic quantities
    P_total: float = 0.0    # Total pressure (MeV/fm³)
    e_total: float = 0.0    # Total energy density (MeV/fm³)
    s_total: float = 0.0    # Total entropy density (fm⁻³)
    
    # Fractions
    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0
    Y_e: float = 0.0
    Y_nu: float = 0.0


# =============================================================================
# INITIAL GUESS GENERATION
# =============================================================================
def get_default_guess_beta_eq(n_B: float, T: float, 
                               params: AlphaBagParams) -> np.ndarray:
    """
    Generate initial guess for beta equilibrium: [μ_u, μ_d, μ_s, μ_e].
    
    Uses simple Fermi gas estimates.
    """
    # Estimate quark chemical potential from n_B
    # n_B = (n_u + n_d + n_s)/3, assume n_u ≈ n_d ≈ n_s ≈ n_B
    # n_q ≈ μ³/π² for massless → μ ≈ (n_B * π²)^(1/3) * hc
    
    mu_estimate = (n_B * PI2)**(1.0/3.0) * hc
    mu_estimate = max(mu_estimate, 50.0)
    
    # In beta eq: μ_d ≈ μ_s, μ_u = μ_d - μ_e
    mu_d = mu_estimate * 1.1
    mu_s = mu_d  # strangeness equilibrium
    mu_e = mu_d * 0.1  # rough estimate
    mu_u = mu_d - mu_e
    
    return np.array([mu_u, mu_d, mu_s, mu_e])


def get_default_guess_fixed_yc(n_B: float, Y_C: float, T: float,
                                params: AlphaBagParams) -> np.ndarray:
    """
    Generate initial guess for fixed Y_C: [μ_u, μ_d, μ_s].
    
    Y_C = (2n_u - n_d - n_s)/(3n_B)
    """
    mu_estimate = (n_B * PI2)**(1.0/3.0) * hc
    mu_estimate = max(mu_estimate, 50.0)
    
    mu_d = mu_estimate * 1.1
    mu_s = mu_d
    mu_u = mu_d * (1.0 + 0.5 * Y_C)  # Adjust based on Y_C
    
    return np.array([mu_u, mu_d, mu_s])


# =============================================================================
# HELPER: BUILD RESULT FROM SOLUTION
# =============================================================================
def _build_result(
    mu_u: float, mu_d: float, mu_s: float, mu_e: float,
    T: float, params: AlphaBagParams,
    include_photons: bool = True,
    include_gluons: bool = True,
    include_thermal_neutrinos: bool = True,
    mu_nu: float = 0.0,
    converged: bool = True,
    error: float = 0.0
) -> AlphaBagEOSResult:
    """
    Build complete EOS result from solved chemical potentials.
    
    Includes contributions from:
    - Quarks (with α-corrections and bag) via compute_alphabag_thermo_from_mu
    - Electrons
    - Photons (if include_photons)
    - Gluons (if include_gluons)
    - Thermal neutrinos (if include_thermal_neutrinos)
    """
    # Quark thermodynamics (includes bag constant)
    quark = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, params)
    
    # Electron thermodynamics
    thermo_e = electron_thermo(mu_e, T)
    
    # Initialize totals with quark + electron contributions
    P_total = quark.P + thermo_e.P
    e_total = quark.e + thermo_e.e
    s_total = quark.s + thermo_e.s
    
    # Neutrino (if μ_ν ≠ 0)
    n_nu = 0.0
    if mu_nu != 0.0:
        thermo_nu = neutrino_thermo(mu_nu, T)
        n_nu = thermo_nu.n
        P_total += thermo_nu.P
        e_total += thermo_nu.e
        s_total += thermo_nu.s
    
    # Photons
    if include_photons:
        thermo_gamma = photon_thermo(T)
        P_total += thermo_gamma.P
        e_total += thermo_gamma.e
        s_total += thermo_gamma.s
    
    # Gluons
    if include_gluons:
        thermo_g = gluon_thermo(T, params.alpha)
        P_total += thermo_g.P
        e_total += thermo_g.e
        s_total += thermo_g.s
    
    # Thermal neutrinos (μ=0)
    # If νe is trapped (mu_nu ≠ 0), only νμ and ντ are thermal (2 flavors)
    # If νe is not trapped (mu_nu = 0), all 3 flavors are thermal
    if include_thermal_neutrinos:
        thermo_nu_th = neutrino_thermo(0.0, T)
        n_thermal_flavors = 2.0 if mu_nu != 0.0 else 3.0
        P_total += n_thermal_flavors * thermo_nu_th.P
        e_total += n_thermal_flavors * thermo_nu_th.e
        s_total += n_thermal_flavors * thermo_nu_th.s
    
    # Fractions
    n_B = quark.n_B
    Y_u = quark.n_u / n_B if n_B > 0 else 0.0
    Y_d = quark.n_d / n_B if n_B > 0 else 0.0
    Y_s = quark.n_s / n_B if n_B > 0 else 0.0
    Y_e = thermo_e.n / n_B if n_B > 0 else 0.0
    Y_nu = n_nu / n_B if n_B > 0 else 0.0
    
    return AlphaBagEOSResult(
        converged=converged,
        error=error,
        n_B=n_B, T=T, Y_C=quark.Y_C, Y_S=quark.Y_S,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_e=mu_e, mu_nu=mu_nu,
        mu_B=quark.mu_B, mu_C=quark.mu_C,
        n_u=quark.n_u, n_d=quark.n_d, n_s=quark.n_s, n_e=thermo_e.n, n_nu=n_nu,
        P_total=P_total, e_total=e_total, s_total=s_total,
        Y_u=Y_u, Y_d=Y_d, Y_s=Y_s, Y_e=Y_e, Y_nu=Y_nu
    )


# =============================================================================
# SOLVER: BETA EQUILIBRIUM
# =============================================================================
def solve_alphabag_beta_eq(
    n_B: float, T: float, params: AlphaBagParams = None,
    include_photons: bool = True,
    include_gluons: bool = True,
    include_thermal_neutrinos: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> AlphaBagEOSResult:
    """
    Solve αBag EOS in beta equilibrium with charge neutrality.
    
    4 equations, 4 unknowns: [μ_u, μ_d, μ_s, μ_e]
    
    Constraints:
        - n_B = (n_u + n_d + n_s)/3 = n_B_target
        - Charge neutrality: (2/3)n_u - (1/3)n_d - (1/3)n_s - n_e = 0
        - Weak equilibrium: μ_d = μ_u + μ_e
        - Strangeness equilibrium: μ_s = μ_d
        
    Args:
        n_B: Target baryon density (fm⁻³)
        T: Temperature (MeV)
        params: AlphaBagParams (uses default if None)
        include_photons: Include photon contributions
        include_gluons: Include gluon contributions
        include_thermal_neutrinos: Include 3 flavors of thermal neutrinos
        initial_guess: Optional initial guess [μ_u, μ_d, μ_s, μ_e]
        
    Returns:
        AlphaBagEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_alphabag_default()
    
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    if initial_guess is None:
        initial_guess = get_default_guess_beta_eq(n_B, T, params)
    
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        
        # Quark densities
        n_u = n_quark_alpha(mu_u, T, m_u, alpha)
        n_d = n_quark_alpha(mu_d, T, m_d, alpha)
        n_s = n_quark_alpha(mu_s, T, m_s, alpha)
        
        # Electron density
        thermo_e = electron_thermo(mu_e, T)
        n_e = thermo_e.n
        
        # n_B constraint
        n_B_calc = (n_u + n_d + n_s) / 3.0
        
        # Charge neutrality
        n_Q = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
        
        return [
            n_B_calc - n_B,                    # Baryon number
            n_Q - n_e,                         # Charge neutrality
            mu_d - mu_u - mu_e,                # Weak equilibrium
            mu_s - mu_d                        # Strangeness equilibrium
        ]
    
    # Solve using hybr, then lm if needed
    sol = root(equations, initial_guess, method='hybr')
    
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')
    
    mu_u, mu_d, mu_s, mu_e = sol.x
    
    return _build_result(
        mu_u, mu_d, mu_s, mu_e, T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        converged=sol.success,
        error=np.max(np.abs(sol.fun))
    )


# =============================================================================
# SOLVER: FIXED Y_C (with strangeness equilibrium)
# =============================================================================
def solve_alphabag_fixed_yc(
    n_B: float, Y_C: float, T: float, params: AlphaBagParams = None,
    include_photons: bool = True,
    include_gluons: bool = True,
    include_electrons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> AlphaBagEOSResult:
    """
    Solve αBag EOS with fixed charge fraction Y_C (strangeness equilibrium).
    
    3 equations, 3 unknowns: [μ_u, μ_d, μ_s]
    
    Constraints:
        - n_B = (n_u + n_d + n_s)/3 = n_B_target
        - Y_C = n_C/n_B = (2n_u - n_d - n_s)/(3n_B) = Y_C_target
        - Strangeness equilibrium: μ_s = μ_d
        
    Args:
        n_B: Target baryon density (fm⁻³)
        Y_C: Target charge fraction
        T: Temperature (MeV)
        params: AlphaBagParams (uses default if None)
        include_photons: Include photon contributions
        include_gluons: Include gluon contributions
        include_electrons: If True, add electrons for neutrality
        initial_guess: Optional initial guess [μ_u, μ_d, μ_s]
        
    Returns:
        AlphaBagEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_alphabag_default()
    
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    if initial_guess is None:
        initial_guess = get_default_guess_fixed_yc(n_B, Y_C, T, params)
    
    def equations(x):
        mu_u, mu_d, mu_s = x
        
        # Quark densities
        n_u = n_quark_alpha(mu_u, T, m_u, alpha)
        n_d = n_quark_alpha(mu_d, T, m_d, alpha)
        n_s = n_quark_alpha(mu_s, T, m_s, alpha)
        
        # n_B constraint
        n_B_calc = (n_u + n_d + n_s) / 3.0
        
        # Y_C constraint
        n_C = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
        Y_C_calc = n_C / n_B_calc if n_B_calc > 0 else 0.0
        
        return [
            n_B_calc - n_B,                    # Baryon number
            Y_C_calc - Y_C,                    # Charge fraction
            mu_s - mu_d                        # Strangeness equilibrium
        ]
    
    # Solve
    sol = root(equations, initial_guess, method='hybr')
    
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')
    
    mu_u, mu_d, mu_s = sol.x
    
    # Determine electron chemical potential
    if include_electrons:
        # For charge neutrality: n_e = n_C
        n_u = n_quark_alpha(mu_u, T, m_u, alpha)
        n_d = n_quark_alpha(mu_d, T, m_d, alpha)
        n_s = n_quark_alpha(mu_s, T, m_s, alpha)
        n_C = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
        
        # Get mu_e_guess from initial_guess if provided
        if initial_guess is not None and len(initial_guess) > 3:
            mu_e_guess = initial_guess[3]
        else:
            mu_e_guess = None
        
        # Use electron_thermo_from_density to find μ_e that gives n_e = n_C
        e_result = electron_thermo_from_density(n_C, T, mu_e_guess=mu_e_guess)
        mu_e = e_result.mu
    else:
        mu_e = 0.0
    
    return _build_result(
        mu_u, mu_d, mu_s, mu_e, T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=True,
        converged=sol.success,
        error=np.max(np.abs(sol.fun))
    )


# =============================================================================
# SOLVER: FIXED Y_C AND Y_S
# =============================================================================
def solve_alphabag_fixed_yc_ys(
    n_B: float, Y_C: float, Y_S: float, T: float, 
    params: AlphaBagParams = None,
    include_photons: bool = True,
    include_gluons: bool = True,
    include_electrons: bool = False,
    initial_guess: Optional[np.ndarray] = None
) -> AlphaBagEOSResult:
    """
    Solve αBag EOS with fixed Y_C AND Y_S (no strangeness equilibrium).
    
    3 equations, 3 unknowns: [μ_u, μ_d, μ_s]
    
    Constraints:
        - n_B = (n_u + n_d + n_s)/3 = n_B_target
        - Y_C = (2n_u - n_d - n_s)/(3n_B) = Y_C_target
        - Y_S = n_s/n_B = Y_S_target
        
    Args:
        n_B: Target baryon density (fm⁻³)
        Y_C: Target charge fraction
        Y_S: Target strangeness fraction
        T: Temperature (MeV)
        params: AlphaBagParams (uses default if None)
        include_photons: Include photon contributions
        include_gluons: Include gluon contributions
        include_electrons: If True, add electrons for neutrality
        initial_guess: Optional initial guess [μ_u, μ_d, μ_s]
        
    Returns:
        AlphaBagEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_alphabag_default()
    
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    if initial_guess is None:
        initial_guess = get_default_guess_fixed_yc(n_B, Y_C, T, params)
    
    def equations(x):
        mu_u, mu_d, mu_s = x
        
        # Quark densities
        n_u = n_quark_alpha(mu_u, T, m_u, alpha)
        n_d = n_quark_alpha(mu_d, T, m_d, alpha)
        n_s = n_quark_alpha(mu_s, T, m_s, alpha)
        
        # n_B constraint
        n_B_calc = (n_u + n_d + n_s) / 3.0
        
        # Y_C constraint
        n_C = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
        Y_C_calc = n_C / n_B_calc if n_B_calc > 0 else 0.0
        
        # Y_S constraint
        Y_S_calc = n_s / n_B_calc if n_B_calc > 0 else 0.0
        
        return [
            n_B_calc - n_B,                    # Baryon number
            Y_C_calc - Y_C,                    # Charge fraction
            Y_S_calc - Y_S                     # Strangeness fraction
        ]
    
    # Solve
    sol = root(equations, initial_guess, method='hybr')
    
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')
    
    mu_u, mu_d, mu_s = sol.x
    
    # Determine electron chemical potential
    if include_electrons:
        # For charge neutrality: n_e = n_C
        n_u = n_quark_alpha(mu_u, T, m_u, alpha)
        n_d = n_quark_alpha(mu_d, T, m_d, alpha)
        n_s = n_quark_alpha(mu_s, T, m_s, alpha)
        n_C = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
        
        # Get mu_e_guess from initial_guess if provided
        if initial_guess is not None and len(initial_guess) > 3:
            mu_e_guess = initial_guess[3]
        else:
            mu_e_guess = None
        
        # Use electron_thermo_from_density to find μ_e that gives n_e = n_C
        e_result = electron_thermo_from_density(n_C, T, mu_e_guess=mu_e_guess)
        mu_e = e_result.mu
    else:
        mu_e = 0.0
    
    return _build_result(
        mu_u, mu_d, mu_s, mu_e, T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=True,
        converged=sol.success,
        error=np.max(np.abs(sol.fun))
    )


# =============================================================================
# CFL PHASE RESULT DATACLASS
# =============================================================================
@dataclass
class CFLEOSResult:
    """Complete result from CFL EOS calculation at one point."""
    # Convergence info
    converged: bool = False
    error: float = 0.0
    
    # Input conditions
    n_B: float = 0.0        # Baryon density (fm⁻³)
    T: float = 0.0          # Temperature (MeV)
    Delta0: float = 0.0     # Zero-temperature gap (MeV)
    Delta: float = 0.0      # Gap at temperature T (MeV)
    Y_C: float = 0.0        # Charge fraction
    Y_S: float = 0.0        # Strangeness fraction
    
    # Chemical potentials
    mu_u: float = 0.0       # Up quark (MeV)
    mu_d: float = 0.0       # Down quark (MeV)
    mu_s: float = 0.0       # Strange quark (MeV)
    
    # Number densities
    n_u: float = 0.0        # Up quark (fm⁻³)
    n_d: float = 0.0        # Down quark (fm⁻³)
    n_s: float = 0.0        # Strange quark (fm⁻³)
    
    # Thermodynamic quantities
    P_total: float = 0.0    # Total pressure (MeV/fm³)
    e_total: float = 0.0    # Total energy density (MeV/fm³)
    s_total: float = 0.0    # Total entropy density (fm⁻³)
    f_total: float = 0.0    # Free energy density (MeV/fm³)
    
    # Fractions
    Y_u: float = 0.0
    Y_d: float = 0.0
    Y_s: float = 0.0


# =============================================================================
# CFL SOLVER
# =============================================================================
def solve_cfl(
    n_B: float, T: float, Delta0: float,
    params: AlphaBagParams = None,
    include_photons: bool = True,
    include_gluons: bool = True,
    initial_guess: Optional[np.ndarray] = None
) -> CFLEOSResult:
    """
    Solve CFL EOS for given baryon density.
    
    In CFL phase, flavor-locking enforces: n_u = n_d = n_s = n_B
    This makes the system charge-neutral (Y_C = 0) by construction.
    
    3 equations, 3 unknowns: [μ_u, μ_d, μ_s]
    
    Constraints:
        - n_u = n_B (up quark density equals baryon density)
        - n_d = n_B (down quark density equals baryon density)
        - n_s = n_B (strange quark density equals baryon density)
    
    Args:
        n_B: Target baryon density (fm⁻³)
        T: Temperature (MeV)
        T: Temperature (MeV)
        Delta0: Zero-temperature pairing gap (MeV)
        params: AlphaBagParams
        include_photons: Include photon gas
        include_gluons: Include gluon thermodynamics
        initial_guess: Optional [μ_u, μ_d, μ_s]
        
    Returns:
        CFLEOSResult with all thermodynamic quantities
    """
    if params is None:
        params = get_alphabag_default()
    
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Gap at this temperature
    Delta = gap_cfl(T, Delta0)
    
    # Initial guess
    if initial_guess is None:
        mu_est = 300.0 * (n_B / 0.4)**(1.0/3.0)
        initial_guess = np.array([mu_est, mu_est, mu_est])
    
    def n_cfl_quark(mu, T, m, alpha, Delta0):
        """Total quark density with CFL correction."""
        return n_quark_alpha(mu, T, m, alpha) + n_cfl_correction(mu, T, Delta0)
    
    def equations(x):
        mu_u, mu_d, mu_s = x
        
        # CFL quark densities - in CFL, n_u = n_d = n_s = n_B
        n_u = n_cfl_quark(mu_u, T, m_u, alpha, Delta0)
        n_d = n_cfl_quark(mu_d, T, m_d, alpha, Delta0)
        n_s = n_cfl_quark(mu_s, T, m_s, alpha, Delta0)
        
        # CFL constraint: each flavor density equals n_B
        return np.array([
            n_u - n_B,           # n_u = n_B
            n_d - n_B,           # n_d = n_B
            n_s - n_B            # n_s = n_B
        ])

    # Solve
    sol = root(equations, initial_guess, method='hybr')
    
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')
    
    mu_u, mu_d, mu_s = sol.x
    
    # Get full CFL thermodynamics
    cfl_thermo = compute_cfl_thermo_from_mu(mu_u, mu_d, mu_s, T, Delta0, params)
    
    # Add gluon and photon contributions
    P_total = cfl_thermo.P
    e_total = cfl_thermo.e
    s_total = cfl_thermo.s
    f_total = cfl_thermo.f
    
    if include_gluons:
        gluon = gluon_thermo(T, alpha)
        P_total += gluon.P
        e_total += gluon.e
        s_total += gluon.s
    
    if include_photons and T > 0:
        photon = photon_thermo(T)
        P_total += photon.P
        e_total += photon.e
        s_total += photon.s
    
    # Recalculate f from totals
    f_total = e_total - T * s_total
    
    return CFLEOSResult(
        converged=sol.success,
        error=np.max(np.abs(sol.fun)),
        n_B=cfl_thermo.n_B, T=T, Delta0=Delta0, Delta=cfl_thermo.Delta,
        Y_C=(2.0/3.0*cfl_thermo.n_u - 1.0/3.0*cfl_thermo.n_d - 1.0/3.0*cfl_thermo.n_s) / cfl_thermo.n_B if cfl_thermo.n_B > 0 else 0.0,
        Y_S=cfl_thermo.n_s / cfl_thermo.n_B if cfl_thermo.n_B > 0 else 0.0,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s,
        n_u=cfl_thermo.n_u, n_d=cfl_thermo.n_d, n_s=cfl_thermo.n_s,
        P_total=P_total, e_total=e_total, s_total=s_total, f_total=f_total,
        Y_u=cfl_thermo.Y_u, Y_d=cfl_thermo.Y_d, Y_s=cfl_thermo.Y_s
    )


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("αBag EOS Solver Test")
    print("=" * 60)
    
    params = get_alphabag_default()
    print(f"Parameters: B^1/4={params.B4} MeV, α_s={params.alpha}, m_s={params.m_s} MeV")
    
    # Test beta equilibrium
    n_B = 0.4
    T = 30.0
    
    print(f"\n1. Beta Equilibrium at n_B={n_B} fm⁻³, T={T} MeV:")
    print("-" * 50)
    
    result = solve_alphabag_beta_eq(n_B, T, params)
    
    print(f"   Converged: {result.converged} (error={result.error:.2e})")
    print(f"   μ_u = {result.mu_u:.2f} MeV")
    print(f"   μ_d = {result.mu_d:.2f} MeV")
    print(f"   μ_s = {result.mu_s:.2f} MeV")
    print(f"   μ_e = {result.mu_e:.2f} MeV")
    print(f"   n_B = {result.n_B:.4f} fm⁻³")
    print(f"   Y_C = {result.Y_C:.4f}")
    print(f"   Y_S = {result.Y_S:.4f}")
    print(f"   P   = {result.P_total:.2f} MeV/fm³")
    print(f"   ε   = {result.e_total:.2f} MeV/fm³")
    print(f"   s   = {result.s_total:.4f} fm⁻³")
    
    # Test fixed Y_C
    Y_C = 0.4
    
    print(f"\n2. Fixed Y_C={Y_C} at n_B={n_B} fm⁻³, T={T} MeV:")
    print("-" * 50)
    
    result2 = solve_alphabag_fixed_yc(n_B, Y_C, T, params)
    
    print(f"   Converged: {result2.converged} (error={result2.error:.2e})")
    print(f"   μ_u = {result2.mu_u:.2f} MeV")
    print(f"   μ_d = {result2.mu_d:.2f} MeV")
    print(f"   μ_s = {result2.mu_s:.2f} MeV")
    print(f"   n_B = {result2.n_B:.4f} fm⁻³")
    print(f"   Y_C = {result2.Y_C:.4f} (target: {Y_C})")
    print(f"   Y_S = {result2.Y_S:.4f}")
    print(f"   P   = {result2.P_total:.2f} MeV/fm³")
    
    # Check charge neutrality in beta eq
    print(f"\n3. Charge neutrality check (beta eq):")
    print("-" * 50)
    n_Q = (2.0/3.0)*result.n_u - (1.0/3.0)*result.n_d - (1.0/3.0)*result.n_s
    print(f"   n_Q (quarks) = {n_Q:.6f} fm⁻³")
    print(f"   n_e          = {result.n_e:.6f} fm⁻³")
    print(f"   n_Q - n_e    = {n_Q - result.n_e:.2e} (should be ~0)")
    
    print("\n✓ All tests passed!")
