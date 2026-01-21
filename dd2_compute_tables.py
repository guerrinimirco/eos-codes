"""
dd2_compute_tables.py
=====================
User-friendly script for generating EOS tables with DD2 model.

Key features:
- Density-dependent couplings with rearrangement term
- OPTIMIZED initial guesses using solutions from previous parameter values
- Self-consistent solver for (σ, ω, ρ, φ, n_B) with R^0 iteration

Usage:
    1. Edit the CONFIGURATION section below
    2. Run: python dd2_compute_tables.py
    
    OR import and use programmatically:
    
    from dd2_compute_tables import compute_table, TableSettings
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
from itertools import product
from scipy.optimize import fsolve, root

# Import DD2 modules
from dd2_parameters import DD2Params, get_dd2_nucleonic, get_dd2y_fortin, get_dd2y_with_deltas
from dd2_thermodynamics_hadrons import (
    compute_hadron_thermo, compute_hadron_thermo_with_R0,
    compute_field_residuals, compute_meson_contribution,
    compute_rearrangement_contribution, compute_total_pressure,
    HadronThermoResult
)
from general_physics_constants import hc3
from general_particles import (
    Proton, Neutron, Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM,
    DeltaPP, DeltaP, Delta0, DeltaM, Electron, Muon
)
from general_thermodynamics_leptons import electron_thermo, muon_thermo


#==============================================================================
# RESULT DATACLASS
#==============================================================================
@dataclass
class EOSResult:
    """Result of a single EOS point calculation."""
    # Input
    n_B: float = 0.0
    T: float = 0.0
    
    # Solution
    sigma: float = 0.0
    omega: float = 0.0
    rho: float = 0.0
    phi: float = 0.0
    R0: float = 0.0
    
    # Chemical potentials
    mu_B: float = 0.0
    mu_Q: float = 0.0
    mu_S: float = 0.0
    mu_L: float = 0.0
    
    # Thermodynamics
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    
    # Composition
    Y_Q: float = 0.0  # Hadronic charge fraction
    Y_S: float = 0.0  # Strangeness fraction
    Y_e: float = 0.0  # Electron fraction
    Y_L: float = 0.0  # Lepton fraction
    n_e: float = 0.0
    n_mu: float = 0.0
    
    # Convergence
    converged: bool = False
    error: float = 1e10
    iterations: int = 0


#==============================================================================
# SETTINGS DATACLASS
#==============================================================================
@dataclass
class TableSettings:
    """
    Configuration for DD2 EOS table generation.
    
    Supports:
    - Multiple parametrizations (DD2, DD2Y, DD2Y_Deltas, custom)
    - Various equilibrium conditions (beta equilibrium, fixed Y_Q, etc.)
    - Multi-dimensional grids (T, Y_C, etc.)
    """
    # Model selection
    parametrization: str = 'dd2'
    particle_content: str = 'nucleons'
    equilibrium: str = 'beta_eq'
    custom_params: Any = None
    
    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.logspace(-2, 0, 50) * 0.149)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    
    # Constraint parameters
    Y_Q_values: Union[float, List[float], None] = None
    Y_S_values: Union[float, List[float], None] = None
    Y_L_values: Union[float, List[float], None] = None
    Y_C_values: Union[float, List[float], None] = None
    
    # Options
    include_muons: bool = True
    max_iterations: int = 100
    tolerance: float = 1e-8
    
    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True
    
    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None
    output_columns: List[str] = field(default_factory=lambda: [
        'n_B', 'T', 
        'sigma', 'omega', 'rho', 'phi', 'R0',
        'mu_B', 'mu_Q', 'mu_S', 'mu_L',
        'P_total', 'e_total', 's_total',
        'Y_Q', 'Y_S', 'Y_e', 'Y_L',
        'converged'
    ])


#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def _to_list(val):
    """Convert value to list if not None."""
    if val is None:
        return [None]
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    return [val]


def _get_particles(content: str) -> List:
    """Get particle list based on content string."""
    nucleons = [Proton, Neutron]
    hyperons = [Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
    deltas = [DeltaPP, DeltaP, Delta0, DeltaM]
    
    content = content.lower()
    if content == 'nucleons':
        return nucleons
    elif content == 'nucleons_hyperons':
        return nucleons + hyperons
    elif content == 'nucleons_hyperons_deltas':
        return nucleons + hyperons + deltas
    else:
        return nucleons


def _get_params(settings: TableSettings) -> DD2Params:
    """Get DD2 parameters based on settings."""
    if settings.custom_params is not None:
        return settings.custom_params
    
    param_name = settings.parametrization.lower()
    if param_name == 'dd2':
        return get_dd2_nucleonic()
    elif param_name in ['dd2y', 'dd2y_fortin']:
        return get_dd2y_fortin()
    elif param_name in ['dd2y_deltas']:
        return get_dd2y_with_deltas()
    else:
        return get_dd2_nucleonic()


#==============================================================================
# SELF-CONSISTENT SOLVER
#==============================================================================

def solve_eos_point_beta_eq(
    n_B_target: float,
    T: float,
    particles: List,
    params: DD2Params,
    include_muons: bool = True,
    initial_guess: Optional[Dict] = None,
    max_iter: int = 100,
    tol: float = 1e-8
) -> EOSResult:
    """
    Solve for EOS point in beta equilibrium.
    
    Conditions:
    - Fixed n_B (baryon density)
    - Fixed T (temperature)
    - Beta equilibrium: μ_e = μ_n - μ_p = -μ_Q
    - Charge neutrality: n_Q + n_e + n_μ = 0
    - μ_S = 0 (no net strangeness imposed chemically)
    
    Unknowns: (σ, ω, ρ, φ, μ_B)
    """
    result = EOSResult(n_B=n_B_target, T=T)
    
    # Initial guess - scale with density for better convergence
    if initial_guess is None:
        # Scale fields with density using approximate relations
        # At saturation: σ ≈ 58 MeV, ω ≈ 74 MeV for DD2
        x = n_B_target / params.n_sat
        sigma0 = 58.0 * x**0.5 if x > 0 else 30.0
        omega0 = 74.0 * x**0.5 if x > 0 else 35.0
        rho0 = 3.0  # Small for near-symmetric matter
        phi0 = 0.0
        mu_B0 = 923.0 if x < 0.5 else 923.0 + 100.0 * (x - 0.5)
    else:
        sigma0 = initial_guess.get('sigma', 50.0)
        omega0 = initial_guess.get('omega', 70.0)
        rho0 = initial_guess.get('rho', 5.0)
        phi0 = initial_guess.get('phi', 0.0)
        mu_B0 = initial_guess.get('mu_B', 920.0)
    
    mu_S = 0.0  # No strangeness chemical potential in beta eq
    mu_L = 0.0  # No trapped neutrinos
    
    def residual(x):
        sigma, omega, rho, phi, mu_B = x
        
        # Ensure positive values for fields
        sigma = abs(sigma)
        omega = abs(omega)
        
        # Compute hadron thermodynamics
        hadron_result = compute_hadron_thermo(
            T, mu_B, -rho, mu_S, sigma, omega, rho, phi, n_B_target,
            particles, params
        )
        
        # mu_Q = -mu_e in beta equilibrium (no neutrinos)
        # For charge neutrality: n_Q_had + n_e = 0
        # So n_e = -n_Q_had
        # And mu_e = -mu_Q
        mu_e = -(-rho * 10)  # Placeholder, will be computed properly
        
        # Actually, we need to solve self-consistently:
        # Use ρ field (times some factor) as proxy for mu_Q initially
        # Then adjust based on charge neutrality
        
        # Field equations (DD2 - linear, no self-interactions)
        res_sigma = params.m_sigma**2 * sigma - hadron_result.src_sigma * hc3
        res_omega = params.m_omega**2 * omega - hadron_result.src_omega * hc3
        res_rho = params.m_rho**2 * rho - hadron_result.src_rho * hc3
        res_phi = params.m_phi**2 * phi - hadron_result.src_phi * hc3
        
        # Baryon density constraint
        res_nB = hadron_result.n_B - n_B_target
        
        return [res_sigma/1e6, res_omega/1e6, res_rho/1e6, res_phi/1e6, res_nB*1e3]
    
    # Full solver with charge neutrality
    def full_residual(x):
        sigma, omega, rho, phi, mu_B, mu_Q = x
        
        sigma = abs(sigma)
        omega = abs(omega)
        
        # Hadron thermodynamics
        hadron_result = compute_hadron_thermo(
            T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_target,
            particles, params
        )
        
        # Lepton thermodynamics
        mu_e = -mu_Q  # Beta equilibrium
        e_result = electron_thermo(mu_e, T, include_antiparticles=True)
        n_e = e_result.n
        P_e = e_result.P
        e_e = e_result.e
        s_e = e_result.s
        
        n_mu = 0.0
        P_mu = 0.0
        e_mu = 0.0
        s_mu = 0.0
        if include_muons and mu_e > 105.66:  # Muon threshold
            mu_result = muon_thermo(mu_e, T, include_antiparticles=True)
            n_mu = mu_result.n
            P_mu = mu_result.P
            e_mu = mu_result.e
            s_mu = mu_result.s
        
        # Field equations
        res_sigma = params.m_sigma**2 * sigma - hadron_result.src_sigma * hc3
        res_omega = params.m_omega**2 * omega - hadron_result.src_omega * hc3
        res_rho = params.m_rho**2 * rho - hadron_result.src_rho * hc3
        res_phi = params.m_phi**2 * phi - hadron_result.src_phi * hc3
        
        # Baryon density constraint
        res_nB = hadron_result.n_B - n_B_target
        
        # Charge neutrality: n_Q_had + n_e + n_mu = 0
        res_charge = hadron_result.n_Q + n_e + n_mu
        
        return [res_sigma/1e6, res_omega/1e6, res_rho/1e6, res_phi/1e6, 
                res_nB*1e3, res_charge*1e3]
    
    # Solve
    x0 = [sigma0, omega0, rho0, phi0, mu_B0, -20.0]  # Initial mu_Q
    
    # Add bounds to prevent unphysical solutions
    bounds = ([1.0, 1.0, -50.0, -50.0, 800.0, -200.0],  # Lower
              [200.0, 300.0, 50.0, 50.0, 1500.0, 50.0])  # Upper
    
    try:
        # Try hybr first (fast)
        sol = root(full_residual, x0, method='hybr', 
                   options={'maxfev': max_iter * 10})
        
        if not sol.success or np.max(np.abs(sol.fun)) > tol * 1e4:
            # Fall back to lm method (more robust)
            sol = root(full_residual, x0, method='lm',
                       options={'maxiter': max_iter * 5})
        
        if sol.success or np.max(np.abs(sol.fun)) < tol * 1e4:
            sigma, omega, rho, phi, mu_B, mu_Q = sol.x
            sigma = abs(sigma)
            omega = abs(omega)
            
            # Final computation
            hadron_result = compute_hadron_thermo(
                T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_target,
                particles, params
            )
            
            mu_e = -mu_Q
            e_result = electron_thermo(mu_e, T, include_antiparticles=True)
            n_e = e_result.n
            P_e = e_result.P
            e_e = e_result.e
            s_e = e_result.s
            
            n_mu = 0.0
            P_mu = e_mu = s_mu = 0.0
            if include_muons and mu_e > 105.66:
                mu_result = muon_thermo(mu_e, T, include_antiparticles=True)
                n_mu = mu_result.n
                P_mu = mu_result.P
                e_mu = mu_result.e
                s_mu = mu_result.s
            
            # Meson contributions
            P_mf, e_mf = compute_meson_contribution(sigma, omega, rho, phi, params)
            
            # Rearrangement
            P_rearr, e_rearr = compute_rearrangement_contribution(
                hadron_result.n_B, hadron_result.R0
            )
            
            # Total
            P_total = hadron_result.P_hadrons + P_mf + P_rearr + P_e + P_mu
            e_total = hadron_result.e_hadrons + e_mf + e_rearr + e_e + e_mu
            s_total = hadron_result.s_hadrons + s_e + s_mu
            
            # Fill result
            result.sigma = sigma
            result.omega = omega
            result.rho = rho
            result.phi = phi
            result.R0 = hadron_result.R0
            result.mu_B = mu_B
            result.mu_Q = mu_Q
            result.mu_S = mu_S
            result.mu_L = mu_L
            result.P_total = P_total
            result.e_total = e_total
            result.s_total = s_total
            result.Y_Q = hadron_result.n_Q / n_B_target if n_B_target > 0 else 0
            result.Y_S = hadron_result.n_S / n_B_target if n_B_target > 0 else 0
            result.Y_e = n_e / n_B_target if n_B_target > 0 else 0
            result.Y_L = (n_e + n_mu) / n_B_target if n_B_target > 0 else 0
            result.n_e = n_e
            result.n_mu = n_mu
            result.converged = True
            result.error = np.max(np.abs(sol.fun))
            result.iterations = sol.nfev
            
    except Exception as e:
        result.converged = False
        result.error = 1e10
    
    return result


def solve_eos_point_fixed_yc(
    n_B_target: float,
    Y_C: float,
    T: float,
    particles: List,
    params: DD2Params,
    include_muons: bool = True,
    initial_guess: Optional[Dict] = None,
    max_iter: int = 100,
    tol: float = 1e-8
) -> EOSResult:
    """
    Solve for EOS point at fixed charge fraction Y_C.
    
    Y_C = Y_Q + Y_e (hadronic charge fraction)
    
    Conditions:
    - Fixed n_B (baryon density)
    - Fixed Y_C (charge fraction)
    - Fixed T (temperature)
    - Charge neutrality: n_Q + n_e + n_μ = 0
    - μ_S = 0
    """
    result = EOSResult(n_B=n_B_target, T=T)
    
    # Initial guess
    if initial_guess is None:
        sigma0 = 50.0
        omega0 = 70.0
        rho0 = 5.0 * (0.5 - Y_C)  # Scale with asymmetry
        phi0 = 0.0
        mu_B0 = 920.0
        mu_Q0 = -50.0 * (0.5 - Y_C)
    else:
        sigma0 = initial_guess.get('sigma', 50.0)
        omega0 = initial_guess.get('omega', 70.0)
        rho0 = initial_guess.get('rho', 5.0)
        phi0 = initial_guess.get('phi', 0.0)
        mu_B0 = initial_guess.get('mu_B', 920.0)
        mu_Q0 = initial_guess.get('mu_Q', -20.0)
    
    mu_S = 0.0
    mu_L = 0.0
    
    def full_residual(x):
        sigma, omega, rho, phi, mu_B, mu_Q = x
        
        sigma = abs(sigma)
        omega = abs(omega)
        
        hadron_result = compute_hadron_thermo(
            T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_target,
            particles, params
        )
        
        # For fixed Y_C (hadrons only), we don't need charge neutrality
        # Instead: Y_Q = Y_C, leptons adjust
        
        # Field equations
        res_sigma = params.m_sigma**2 * sigma - hadron_result.src_sigma * hc3
        res_omega = params.m_omega**2 * omega - hadron_result.src_omega * hc3
        res_rho = params.m_rho**2 * rho - hadron_result.src_rho * hc3
        res_phi = params.m_phi**2 * phi - hadron_result.src_phi * hc3
        
        # Baryon density constraint
        res_nB = hadron_result.n_B - n_B_target
        
        # Charge fraction constraint: Y_Q = Y_C
        Y_Q_current = hadron_result.n_Q / hadron_result.n_B if hadron_result.n_B > 0 else 0
        res_YC = Y_Q_current - Y_C
        
        return [res_sigma/1e6, res_omega/1e6, res_rho/1e6, res_phi/1e6, 
                res_nB*1e3, res_YC*10]
    
    x0 = [sigma0, omega0, rho0, phi0, mu_B0, mu_Q0]
    
    try:
        sol = root(full_residual, x0, method='hybr',
                   options={'maxfev': max_iter * 10})
        
        if sol.success or np.max(np.abs(sol.fun)) < tol * 1e4:
            sigma, omega, rho, phi, mu_B, mu_Q = sol.x
            sigma = abs(sigma)
            omega = abs(omega)
            
            hadron_result = compute_hadron_thermo(
                T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_target,
                particles, params
            )
            
            # Electrons for charge neutrality
            n_Q_had = hadron_result.n_Q
            n_e_needed = -n_Q_had  # For charge neutrality
            
            # Find mu_e that gives this density
            from scipy.optimize import brentq
            
            def electron_density_residual(mu_e):
                e_res = electron_thermo(mu_e, T)
                n_tot = e_res.n
                if include_muons and mu_e > 105.66:
                    mu_res = muon_thermo(mu_e, T)
                    n_tot += mu_res.n
                return n_tot - n_e_needed
            
            try:
                if n_e_needed > 0:
                    mu_e = brentq(electron_density_residual, 0.1, 500.0)
                else:
                    mu_e = 0.1
            except:
                mu_e = -mu_Q if mu_Q < 0 else 10.0
            
            e_result = electron_thermo(mu_e, T)
            n_e = e_result.n
            P_e = e_result.P
            e_e = e_result.e
            s_e = e_result.s
            
            n_mu = P_mu = e_mu = s_mu = 0.0
            if include_muons and mu_e > 105.66:
                mu_result = muon_thermo(mu_e, T)
                n_mu = mu_result.n
                P_mu = mu_result.P
                e_mu = mu_result.e
                s_mu = mu_result.s
            
            P_mf, e_mf = compute_meson_contribution(sigma, omega, rho, phi, params)
            P_rearr, e_rearr = compute_rearrangement_contribution(
                hadron_result.n_B, hadron_result.R0
            )
            
            P_total = hadron_result.P_hadrons + P_mf + P_rearr + P_e + P_mu
            e_total = hadron_result.e_hadrons + e_mf + e_rearr + e_e + e_mu
            s_total = hadron_result.s_hadrons + s_e + s_mu
            
            result.sigma = sigma
            result.omega = omega
            result.rho = rho
            result.phi = phi
            result.R0 = hadron_result.R0
            result.mu_B = mu_B
            result.mu_Q = mu_Q
            result.mu_S = mu_S
            result.mu_L = mu_L
            result.P_total = P_total
            result.e_total = e_total
            result.s_total = s_total
            result.Y_Q = hadron_result.n_Q / n_B_target if n_B_target > 0 else 0
            result.Y_S = hadron_result.n_S / n_B_target if n_B_target > 0 else 0
            result.Y_e = n_e / n_B_target if n_B_target > 0 else 0
            result.Y_L = (n_e + n_mu) / n_B_target if n_B_target > 0 else 0
            result.n_e = n_e
            result.n_mu = n_mu
            result.converged = True
            result.error = np.max(np.abs(sol.fun))
            result.iterations = sol.nfev
            
    except Exception as e:
        result.converged = False
        result.error = 1e10
    
    return result


#==============================================================================
# TABLE GENERATOR
#==============================================================================

def compute_table(settings: TableSettings) -> Dict[Tuple, List[EOSResult]]:
    """
    Compute DD2 EOS table(s) with optimized initial guesses.
    """
    params = _get_params(settings)
    particles = _get_particles(settings.particle_content)
    
    # Build parameter grid
    T_list = list(settings.T_values)
    Y_C_list = _to_list(settings.Y_C_values)
    Y_Q_list = _to_list(settings.Y_Q_values)
    Y_S_list = _to_list(settings.Y_S_values)
    Y_L_list = _to_list(settings.Y_L_values)
    
    n_B_array = np.asarray(settings.n_B_values)
    
    param_grid = list(product(T_list, Y_C_list, Y_Q_list, Y_S_list, Y_L_list))
    
    all_results = {}
    previous_table_results = None
    
    total_start = time.time()
    n_points = len(n_B_array)
    n_tables = len(param_grid)
    
    print(f"\nComputing {n_tables} table(s), {n_points} points each")
    print(f"Model: {params.name}, Particles: {settings.particle_content}")
    print(f"Equilibrium: {settings.equilibrium}")
    print("=" * 70)
    
    for table_idx, (T, Y_C, Y_Q, Y_S, Y_L) in enumerate(param_grid):
        param_str = f"T={T:.1f}"
        if Y_C is not None:
            param_str += f", Y_C={Y_C:.2f}"
        
        print(f"\nTable {table_idx+1}/{n_tables}: {param_str}")
        
        results = []
        previous_solution = None
        start_time = time.time()
        
        for i, n_B in enumerate(n_B_array):
            # Build initial guess
            guess = None
            if previous_solution is not None:
                guess = previous_solution
            elif previous_table_results is not None and i < len(previous_table_results):
                prev = previous_table_results[i]
                if prev.converged:
                    guess = {
                        'sigma': prev.sigma,
                        'omega': prev.omega,
                        'rho': prev.rho,
                        'phi': prev.phi,
                        'mu_B': prev.mu_B,
                        'mu_Q': prev.mu_Q
                    }
            
            # Solve based on equilibrium type
            eq_type = settings.equilibrium.lower()
            
            if eq_type == 'beta_eq':
                result = solve_eos_point_beta_eq(
                    n_B, T, particles, params,
                    include_muons=settings.include_muons,
                    initial_guess=guess,
                    max_iter=settings.max_iterations,
                    tol=settings.tolerance
                )
            elif eq_type in ['fixed_yc', 'fixed_yc_hadrons_only', 'fixed_yc_neutral']:
                result = solve_eos_point_fixed_yc(
                    n_B, Y_C, T, particles, params,
                    include_muons=settings.include_muons,
                    initial_guess=guess,
                    max_iter=settings.max_iterations,
                    tol=settings.tolerance
                )
            else:
                result = solve_eos_point_beta_eq(
                    n_B, T, particles, params,
                    include_muons=settings.include_muons,
                    initial_guess=guess
                )
            
            results.append(result)
            
            # Store for next iteration
            if result.converged:
                previous_solution = {
                    'sigma': result.sigma,
                    'omega': result.omega,
                    'rho': result.rho,
                    'phi': result.phi,
                    'mu_B': result.mu_B,
                    'mu_Q': result.mu_Q
                }
            
            # Print progress
            if settings.print_results:
                should_print = False
                if i < settings.print_first_n:
                    should_print = True
                elif settings.print_errors and not result.converged:
                    should_print = True
                
                if should_print:
                    status = "OK" if result.converged else "FAILED"
                    print(f"[{i:4d}] n_B={n_B:.4e} [{status}] P={result.P_total:.4e} err={result.error:.2e}")
        
        elapsed = time.time() - start_time
        
        # Store results
        params_key = (T, Y_C, Y_Q, Y_S, Y_L)
        all_results[params_key] = results
        previous_table_results = results
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            print(f"\n  Completed in {elapsed:.2f} s ({elapsed*1000/n_points:.1f} ms/point)")
            print(f"  Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    total_elapsed = time.time() - total_start
    
    if settings.print_timing:
        print("\n" + "=" * 70)
        print(f"Total time: {total_elapsed:.2f} s")
        print(f"Average: {total_elapsed*1000/(n_points * n_tables):.1f} ms/point")
    
    if settings.save_to_file:
        save_results(all_results, settings)
    
    return all_results


def save_results(all_results: Dict[Tuple, List[EOSResult]], settings: TableSettings):
    """Save results to file."""
    if settings.output_filename:
        filename = settings.output_filename
    else:
        filename = f"dd2_eos_{settings.parametrization}_{settings.particle_content}_{settings.equilibrium}.dat"
    
    with open(filename, 'w') as f:
        f.write(f"# DD2 EOS Table: {settings.parametrization}, {settings.particle_content}\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")
        f.write("# " + " ".join(f"{col:>14}" for col in settings.output_columns) + "\n")
        
        for params, results in all_results.items():
            T, Y_C, Y_Q, Y_S, Y_L = params
            for r in results:
                if r.converged:
                    row = []
                    for col in settings.output_columns:
                        val = getattr(r, col, 0.0)
                        if val is None:
                            val = 0.0
                        if isinstance(val, bool):
                            val = 1 if val else 0
                        row.append(f"{val:>14.6e}" if isinstance(val, float) else f"{val:>14}")
                    f.write(" ".join(row) + "\n")
    
    print(f"\nSaved to: {filename}")


def results_to_arrays(results: List[EOSResult]) -> Dict[str, np.ndarray]:
    """Convert list of EOSResult to dictionary of numpy arrays."""
    attrs = [
        'n_B', 'T', 'P_total', 'e_total', 's_total',
        'sigma', 'omega', 'rho', 'phi', 'R0',
        'mu_B', 'mu_Q', 'mu_S', 'mu_L',
        'Y_Q', 'Y_S', 'Y_e', 'Y_L', 'n_e', 'n_mu', 'error'
    ]
    arrays = {}
    for attr in attrs:
        arrays[attr] = np.array([getattr(r, attr, np.nan) for r in results if r.converged])
    arrays['converged'] = np.array([r.converged for r in results])
    
    # Derived quantities
    if len(arrays['Y_Q']) > 0:
        arrays['Y_C'] = arrays['Y_Q'] + arrays['Y_e']
    
    return arrays


#==============================================================================
# CONFIGURATION (EDIT THIS SECTION)
#==============================================================================
settings = TableSettings(
    # ===================== MODEL =====================
    parametrization='dd2y',  # dd2, dd2y, dd2y_deltas
    particle_content='nucleons_hyperons',  # nucleons, nucleons_hyperons, nucleons_hyperons_deltas
    include_muons=True,
    
    # ===================== EQUILIBRIUM =====================
    equilibrium='beta_eq',  # beta_eq, fixed_yc
    
    # ===================== GRID =====================
    n_B_values=np.linspace(0.1, 10, 300) * 0.149,
    T_values=[0.1,10.0],
    
    # ===================== CONSTRAINTS =====================
    # Y_C_values=0.5,  # For fixed_yc mode
    
    # ===================== OUTPUT =====================
    print_results=True,
    print_first_n=3,
    print_errors=True,
    print_timing=True,
    save_to_file=True,
    output_filename=None,
)


#==============================================================================
# MAIN
#==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DD2 EOS TABLE GENERATOR")
    print("=" * 70 + "\n")
    
    all_results = compute_table(settings)
    
    if len(all_results) == 1:
        key = list(all_results.keys())[0]
        data = results_to_arrays(all_results[key])
        print("\n" + "=" * 70)
        print("DONE!")
        if len(data['n_B']) > 0:
            print(f"  n_B: [{data['n_B'].min():.4e}, {data['n_B'].max():.4e}] fm^-3")
            print(f"  P:   [{data['P_total'].min():.4e}, {data['P_total'].max():.4e}] MeV/fm^3")
    else:
        print(f"\nGenerated {len(all_results)} tables")
    
    print("=" * 70 + "\n")
