#!/usr/bin/env python3
"""
zlvmit_thermodynamic_derivatives.py
===================================
Thermodynamic derivative quantities following CompOSE conventions.

This module computes:
    - cv: Heat capacity at constant volume
    - cp: Heat capacity at constant pressure
    - βV: Tension coefficient at constant volume
    - κT: Isothermal compressibility
    - κS: Adiabatic compressibility
    - αp: Expansion coefficient at constant pressure
    - Γ: Adiabatic index (cp/cv)
    - Γ̃: Polytropic index
    - cs²: Speed of sound squared

All quantities are computed via numerical differentiation of the free energy
per baryon F(T, n_B, Y_q) following the CompOSE manual (Sec. 3.6).

Units:
    - cv, cp: dimensionless (per baryon)
    - βV: fm⁻³
    - κT, κS: MeV⁻¹ fm³
    - αp: MeV⁻¹
    - Γ, Γ̃: dimensionless
    - cs²: in units of c² (speed of light squared)
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# THERMODYNAMIC COEFFICIENTS DATACLASS
# =============================================================================

@dataclass
class ThermodynamicCoefficients:
    """
    Collection of thermodynamic derivative quantities.
    
    All quantities follow CompOSE conventions (manual Sec. 3.6).
    """
    # Heat capacities
    cv: float = 0.0       # Heat capacity at constant V [dimensionless, per baryon]
    cp: float = 0.0       # Heat capacity at constant P [dimensionless, per baryon]
    
    # Compressibilities
    kappa_T: float = 0.0  # Isothermal compressibility [MeV⁻¹ fm³]
    kappa_S: float = 0.0  # Adiabatic compressibility [MeV⁻¹ fm³]
    
    # Other coefficients
    beta_V: float = 0.0   # Tension coefficient [fm⁻³]
    alpha_p: float = 0.0  # Expansion coefficient [MeV⁻¹]
    
    # Adiabatic indices
    Gamma: float = 0.0    # cp/cv [dimensionless]
    Gamma_tilde: float = 0.0  # Polytropic index ∂ln(p)/∂ln(nB)|S [dimensionless]
    gamma: float = 0.0    # ∂ln(p)/∂ln(e)|S [dimensionless]
    
    # Speed of sound
    cs2: float = 0.0      # Speed of sound squared [c²]
    
    # Validity flag
    valid: bool = True
    error_message: str = ""


# =============================================================================
# NUMERICAL DIFFERENTIATION HELPERS
# =============================================================================

def _second_derivative_T(func: Callable, T: float, n_B: float, delta_T: float = 0.5) -> float:
    """
    Compute second derivative with respect to T using 3-point central difference.
    
    d²f/dT² ≈ (f(T+δ) - 2*f(T) + f(T-δ)) / δ²
    """
    f_plus = func(n_B, T + delta_T)
    f_center = func(n_B, T)
    f_minus = func(n_B, T - delta_T)
    
    return (f_plus - 2*f_center + f_minus) / (delta_T**2)


def _second_derivative_nB(func: Callable, T: float, n_B: float, delta_nB: float = 0.001) -> float:
    """
    Compute second derivative with respect to n_B using 3-point central difference.
    
    d²f/dn_B² ≈ (f(n+δ) - 2*f(n) + f(n-δ)) / δ²
    """
    f_plus = func(n_B + delta_nB, T)
    f_center = func(n_B, T)
    f_minus = func(n_B - delta_nB, T)
    
    return (f_plus - 2*f_center + f_minus) / (delta_nB**2)


def _mixed_derivative_T_nB(func: Callable, T: float, n_B: float, 
                           delta_T: float = 0.5, delta_nB: float = 0.001) -> float:
    """
    Compute mixed second derivative ∂²f/(∂T ∂n_B) using 4-point formula.
    
    ∂²f/∂T∂n_B ≈ (f(n+δ,T+δ) - f(n+δ,T-δ) - f(n-δ,T+δ) + f(n-δ,T-δ)) / (4*δT*δn)
    """
    f_pp = func(n_B + delta_nB, T + delta_T)
    f_pm = func(n_B + delta_nB, T - delta_T)
    f_mp = func(n_B - delta_nB, T + delta_T)
    f_mm = func(n_B - delta_nB, T - delta_T)
    
    return (f_pp - f_pm - f_mp + f_mm) / (4 * delta_T * delta_nB)


def _first_derivative_T(func: Callable, T: float, n_B: float, delta_T: float = 0.5) -> float:
    """
    Compute first derivative with respect to T using central difference.
    """
    return (func(n_B, T + delta_T) - func(n_B, T - delta_T)) / (2 * delta_T)


def _first_derivative_nB(func: Callable, T: float, n_B: float, delta_nB: float = 0.001) -> float:
    """
    Compute first derivative with respect to n_B using central difference.
    """
    return (func(n_B + delta_nB, T) - func(n_B - delta_nB, T)) / (2 * delta_nB)


# =============================================================================
# COMPUTE COEFFICIENTS FROM RAW SOLVER
# =============================================================================

def compute_thermodynamic_derivatives_from_solver(
    n_B: float, 
    T: float, 
    eta: float,
    solver_func: Callable,
    eq_mode: str = "beta",
    Y_C: float = None,
    Y_L: float = None,
    delta_T: float = 0.5,
    delta_nB: float = 0.002,
    zl_params = None,
    vmit_params = None
) -> ThermodynamicCoefficients:
    """
    Compute all thermodynamic derivatives using numerical differentiation.
    
    Uses the solver function to compute thermodynamic quantities at nearby
    points and applies numerical differentiation formulas from CompOSE.
    
    Args:
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        eta: Surface tension parameter
        solver_func: Function(n_B, T, eta, ...) -> result with P, e, s, f
        eq_mode: "beta", "fixed_yc", or "trapped"
        Y_C: Charge fraction (for fixed_yc mode)
        Y_L: Lepton fraction (for trapped mode)
        delta_T: Step size for T derivatives (MeV)
        delta_nB: Step size for n_B derivatives (fm⁻³)
        zl_params, vmit_params: Model parameters
        
    Returns:
        ThermodynamicCoefficients with all computed quantities
    """
    result = ThermodynamicCoefficients()
    
    # Handle low temperature case
    if T < 2 * delta_T:
        delta_T = T / 3.0 if T > 0.1 else 0.05
    
    # Handle low density case
    if n_B < 2 * delta_nB:
        delta_nB = n_B / 3.0 if n_B > 0.01 else 0.002
    
    # Define wrapper functions that return specific quantities
    def solve_at(nB_val: float, T_val: float):
        """Solve EOS at given point and return result."""
        if eq_mode == "fixed_yc":
            return solver_func(nB_val, T_val, eta, eq_mode=eq_mode, Y_C=Y_C,
                              zl_params=zl_params, vmit_params=vmit_params)
        elif eq_mode == "trapped":
            return solver_func(nB_val, T_val, eta, eq_mode=eq_mode, Y_L=Y_L,
                              zl_params=zl_params, vmit_params=vmit_params)
        else:
            return solver_func(nB_val, T_val, eta, eq_mode=eq_mode,
                              zl_params=zl_params, vmit_params=vmit_params)
    
    # Get results at stencil points
    try:
        r_center = solve_at(n_B, T)
        r_T_plus = solve_at(n_B, T + delta_T)
        r_T_minus = solve_at(n_B, T - delta_T)
        r_nB_plus = solve_at(n_B + delta_nB, T)
        r_nB_minus = solve_at(n_B - delta_nB, T)
        r_pp = solve_at(n_B + delta_nB, T + delta_T)
        r_pm = solve_at(n_B + delta_nB, T - delta_T)
        r_mp = solve_at(n_B - delta_nB, T + delta_T)
        r_mm = solve_at(n_B - delta_nB, T - delta_T)
    except Exception as e:
        result.valid = False
        result.error_message = f"Solver failed: {str(e)}"
        return result
    
    # Check convergence
    all_converged = all([
        getattr(r, 'converged', True) for r in 
        [r_center, r_T_plus, r_T_minus, r_nB_plus, r_nB_minus, r_pp, r_pm, r_mp, r_mm]
    ])
    
    if not all_converged:
        result.valid = False
        result.error_message = "Not all stencil points converged"
        # Continue anyway with available data
    
    # Extract thermodynamic quantities
    def get_F(r):
        """Free energy per baryon: F = (e - T*s) / n_B = f/n_B"""
        return (r.e_total - r.T * r.s_total) / r.n_B if r.n_B > 0 else 0.0
    
    def get_P(r):
        return r.P_total
    
    def get_e(r):
        return r.e_total
    
    def get_s(r):
        return r.s_total
    
    # Center point values
    P = get_P(r_center)
    e = get_e(r_center)
    s = get_s(r_center)
    h = e + P  # Enthalpy density
    
    # =======================================================================
    # Compute derivatives of F = f/n_B (free energy per baryon)
    # =======================================================================
    
    # F values on stencil
    F_center = get_F(r_center)
    F_T_plus = get_F(r_T_plus)
    F_T_minus = get_F(r_T_minus)
    F_nB_plus = get_F(r_nB_plus)
    F_nB_minus = get_F(r_nB_minus)
    F_pp = get_F(r_pp)
    F_pm = get_F(r_pm)
    F_mp = get_F(r_mp)
    F_mm = get_F(r_mm)
    
    # Second derivatives
    d2F_dT2 = (F_T_plus - 2*F_center + F_T_minus) / (delta_T**2)
    d2F_dnB2 = (F_nB_plus - 2*F_center + F_nB_minus) / (delta_nB**2)
    d2F_dTdnB = (F_pp - F_pm - F_mp + F_mm) / (4 * delta_T * delta_nB)
    
    # For κT we need d²(F*n_B)/dn_B² = d²f/dn_B²
    # where f = F * n_B is free energy density
    f_center = F_center * n_B
    f_nB_plus = F_nB_plus * (n_B + delta_nB)
    f_nB_minus = F_nB_minus * (n_B - delta_nB)
    d2f_dnB2 = (f_nB_plus - 2*f_center + f_nB_minus) / (delta_nB**2)
    
    # =======================================================================
    # Compute pressure derivatives directly
    # =======================================================================
    P_T_plus = get_P(r_T_plus)
    P_T_minus = get_P(r_T_minus)
    P_nB_plus = get_P(r_nB_plus)
    P_nB_minus = get_P(r_nB_minus)
    
    dP_dT = (P_T_plus - P_T_minus) / (2 * delta_T)
    dP_dnB = (P_nB_plus - P_nB_minus) / (2 * delta_nB)
    
    # =======================================================================
    # Compute thermodynamic coefficients using CompOSE formulas
    # =======================================================================
    
    # cv = -T * ∂²F/∂T² (Eq. 3.42)
    result.cv = -T * d2F_dT2 if T > 0 else 0.0
    
    # βV = n_B² * ∂²F/(∂T∂n_B) (Eq. 3.43)
    result.beta_V = n_B**2 * d2F_dTdnB
    
    # κT = (n_B² * ∂²(F*n_B)/∂n_B²)⁻¹ (Eq. 3.44)
    # Alternatively: κT = (n_B * ∂P/∂n_B)⁻¹
    denom_kT = n_B * dP_dnB
    if abs(denom_kT) > 1e-10:
        result.kappa_T = 1.0 / denom_kT
    else:
        result.kappa_T = 0.0
    
    # αp = κT * βV (Eq. 3.45)
    result.alpha_p = result.kappa_T * result.beta_V
    
    # cp = cv + (T/n_B) * αp * βV (Eq. 3.46)
    if n_B > 0:
        result.cp = result.cv + (T / n_B) * result.alpha_p * result.beta_V
    else:
        result.cp = result.cv
    
    # Γ = cp / cv (Eq. 3.47)
    if abs(result.cv) > 1e-10:
        result.Gamma = result.cp / result.cv
    else:
        result.Gamma = 1.0
    
    # κS = κT / Γ (Eq. 3.48)
    if abs(result.Gamma) > 1e-10:
        result.kappa_S = result.kappa_T / result.Gamma
    else:
        result.kappa_S = result.kappa_T
    
    # Γ̃ = 1 / (P * κS) (Eq. 3.49)
    if abs(P * result.kappa_S) > 1e-10:
        result.Gamma_tilde = 1.0 / (P * result.kappa_S)
    else:
        result.Gamma_tilde = 0.0
    
    # γ = (P/e) * Γ̃ = (P/e) / (P * κS) = 1 / (e * κS) (Eq. 3.50-3.51)
    if abs(e * result.kappa_S) > 1e-10:
        result.gamma = 1.0 / (e * result.kappa_S)
    else:
        result.gamma = 0.0
    
    # cs² = 1 / (h * κS) = (P/h) * Γ̃ (Eq. 3.51)
    if abs(h * result.kappa_S) > 1e-10:
        result.cs2 = 1.0 / (h * result.kappa_S)
    else:
        result.cs2 = 0.0
    
    # Check physical bounds
    if result.cs2 < 0 or result.cs2 > 1:
        result.valid = False
        result.error_message += f" cs2={result.cs2:.4f} out of bounds."
    
    if result.cv < 0:
        result.valid = False
        result.error_message += f" cv={result.cv:.4f} < 0."
    
    return result


# =============================================================================
# COMPUTE COEFFICIENTS FROM TABULATED DATA
# =============================================================================

def compute_thermodynamic_derivatives_from_table(
    n_B: float,
    T: float,
    P_interp: RegularGridInterpolator,
    e_interp: RegularGridInterpolator,
    s_interp: RegularGridInterpolator,
    delta_T: float = 0.5,
    delta_nB: float = 0.002
) -> ThermodynamicCoefficients:
    """
    Compute thermodynamic derivatives from interpolated table data.
    
    This is faster than recomputing the EOS at each stencil point, but requires
    pre-computed tables with sufficient resolution.
    
    Args:
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        P_interp: Interpolator for P(n_B, T)
        e_interp: Interpolator for e(n_B, T)
        s_interp: Interpolator for s(n_B, T)
        delta_T: Step size for T derivatives (MeV)
        delta_nB: Step size for n_B derivatives (fm⁻³)
        
    Returns:
        ThermodynamicCoefficients
    """
    result = ThermodynamicCoefficients()
    
    # Handle boundary cases
    if T < 2 * delta_T:
        delta_T = max(T / 3.0, 0.05)
    if n_B < 2 * delta_nB:
        delta_nB = max(n_B / 3.0, 0.001)
    
    try:
        # Get values at stencil points
        P = P_interp((n_B, T))
        e = e_interp((n_B, T))
        s = s_interp((n_B, T))
        h = e + P
        
        # Free energy per baryon: F = (e - T*s) / n_B
        def get_F(nB_val, T_val):
            e_val = e_interp((nB_val, T_val))
            s_val = s_interp((nB_val, T_val))
            return (e_val - T_val * s_val) / nB_val if nB_val > 0 else 0.0
        
        # Second derivatives of F
        F_center = get_F(n_B, T)
        F_T_plus = get_F(n_B, T + delta_T)
        F_T_minus = get_F(n_B, T - delta_T)
        F_nB_plus = get_F(n_B + delta_nB, T)
        F_nB_minus = get_F(n_B - delta_nB, T)
        F_pp = get_F(n_B + delta_nB, T + delta_T)
        F_pm = get_F(n_B + delta_nB, T - delta_T)
        F_mp = get_F(n_B - delta_nB, T + delta_T)
        F_mm = get_F(n_B - delta_nB, T - delta_T)
        
        d2F_dT2 = (F_T_plus - 2*F_center + F_T_minus) / (delta_T**2)
        d2F_dTdnB = (F_pp - F_pm - F_mp + F_mm) / (4 * delta_T * delta_nB)
        
        # Pressure derivatives
        P_T_plus = P_interp((n_B, T + delta_T))
        P_T_minus = P_interp((n_B, T - delta_T))
        P_nB_plus = P_interp((n_B + delta_nB, T))
        P_nB_minus = P_interp((n_B - delta_nB, T))
        
        dP_dT = (P_T_plus - P_T_minus) / (2 * delta_T)
        dP_dnB = (P_nB_plus - P_nB_minus) / (2 * delta_nB)
        
        # Compute coefficients
        result.cv = -T * d2F_dT2 if T > 0 else 0.0
        result.beta_V = n_B**2 * d2F_dTdnB
        
        denom_kT = n_B * dP_dnB
        result.kappa_T = 1.0 / denom_kT if abs(denom_kT) > 1e-10 else 0.0
        result.alpha_p = result.kappa_T * result.beta_V
        result.cp = result.cv + (T / n_B) * result.alpha_p * result.beta_V if n_B > 0 else result.cv
        result.Gamma = result.cp / result.cv if abs(result.cv) > 1e-10 else 1.0
        result.kappa_S = result.kappa_T / result.Gamma if abs(result.Gamma) > 1e-10 else result.kappa_T
        result.Gamma_tilde = 1.0 / (P * result.kappa_S) if abs(P * result.kappa_S) > 1e-10 else 0.0
        result.gamma = 1.0 / (e * result.kappa_S) if abs(e * result.kappa_S) > 1e-10 else 0.0
        result.cs2 = 1.0 / (h * result.kappa_S) if abs(h * result.kappa_S) > 1e-10 else 0.0
        
    except Exception as e:
        result.valid = False
        result.error_message = str(e)
    
    return result


def add_derivatives_to_table(
    data: Dict[str, np.ndarray],
    n_B_values: np.ndarray = None,
    T_values: np.ndarray = None,
    delta_T: float = 1.0,
    delta_nB: float = 0.005
) -> Dict[str, np.ndarray]:
    """
    Add thermodynamic derivative columns to existing table data.
    
    Args:
        data: Dictionary with 'n_B', 'T', 'P_total', 'e_total', 's_total' columns
        n_B_values: Unique n_B grid (extracted from data if None)
        T_values: Unique T grid (extracted from data if None)
        delta_T, delta_nB: Step sizes for numerical derivatives
        
    Returns:
        Updated data dictionary with additional columns:
        'cv', 'cp', 'kappa_T', 'kappa_S', 'cs2', 'Gamma', 'Gamma_tilde'
    """
    if n_B_values is None:
        n_B_values = np.unique(data['n_B'])
    if T_values is None:
        T_values = np.unique(data['T'])
    
    n_points = len(data['n_B'])
    
    # Reshape data for interpolation (n_B, T) grid
    n_nB = len(n_B_values)
    n_T = len(T_values)
    
    def reshape_to_grid(arr):
        """Reshape flat array to (n_B, T) grid."""
        if len(data['T']) >= 2 and data['T'][0] == data['T'][1]:
            # n_B varies fastest
            return arr.reshape(n_T, n_nB).T
        else:
            # T varies fastest
            return arr.reshape(n_nB, n_T)
    
    P_grid = reshape_to_grid(data['P_total'])
    e_grid = reshape_to_grid(data['e_total'])
    s_grid = reshape_to_grid(data['s_total'])
    
    # Create interpolators
    P_interp = RegularGridInterpolator((n_B_values, T_values), P_grid, 
                                        bounds_error=False, fill_value=None)
    e_interp = RegularGridInterpolator((n_B_values, T_values), e_grid,
                                        bounds_error=False, fill_value=None)
    s_interp = RegularGridInterpolator((n_B_values, T_values), s_grid,
                                        bounds_error=False, fill_value=None)
    
    # Initialize new columns
    cv_arr = np.zeros(n_points)
    cp_arr = np.zeros(n_points)
    kappa_T_arr = np.zeros(n_points)
    kappa_S_arr = np.zeros(n_points)
    cs2_arr = np.zeros(n_points)
    Gamma_arr = np.zeros(n_points)
    Gamma_tilde_arr = np.zeros(n_points)
    
    # Compute derivatives for each point
    for i in range(n_points):
        n_B = data['n_B'][i]
        T = data['T'][i]
        
        # Adjust step sizes near boundaries
        dT = min(delta_T, T / 3.0) if T > 0.1 else 0.05
        dnB = min(delta_nB, n_B / 3.0) if n_B > 0.01 else 0.002
        
        coeffs = compute_thermodynamic_derivatives_from_table(
            n_B, T, P_interp, e_interp, s_interp, dT, dnB
        )
        
        cv_arr[i] = coeffs.cv
        cp_arr[i] = coeffs.cp
        kappa_T_arr[i] = coeffs.kappa_T
        kappa_S_arr[i] = coeffs.kappa_S
        cs2_arr[i] = coeffs.cs2
        Gamma_arr[i] = coeffs.Gamma
        Gamma_tilde_arr[i] = coeffs.Gamma_tilde
    
    # Add to data dict
    data['cv'] = cv_arr
    data['cp'] = cp_arr
    data['kappa_T'] = kappa_T_arr
    data['kappa_S'] = kappa_S_arr
    data['cs2'] = cs2_arr
    data['Gamma'] = Gamma_arr
    data['Gamma_tilde'] = Gamma_tilde_arr
    
    return data


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Thermodynamic Derivatives Test")
    print("=" * 50)
    
    # Create synthetic data: ideal gas P = n_B * T, e = (3/2) * n_B * T
    n_B_grid = np.linspace(0.1, 1.0, 30)
    T_grid = np.linspace(10.0, 100.0, 30)
    
    n_B_mesh, T_mesh = np.meshgrid(n_B_grid, T_grid, indexing='ij')
    
    # Ideal gas: P = n*T, e = (3/2)*n*T, s = (5/2)*n (in appropriate units)
    # For ideal relativistic gas, cs² = 1/3
    P_mesh = n_B_mesh * T_mesh * 0.1  # Scale factor
    e_mesh = 1.5 * n_B_mesh * T_mesh * 0.1
    s_mesh = 2.5 * n_B_mesh * 0.1  # Entropy ~ n at fixed S per particle
    
    data = {
        'n_B': n_B_mesh.flatten(),
        'T': T_mesh.flatten(),
        'P_total': P_mesh.flatten(),
        'e_total': e_mesh.flatten(),
        's_total': s_mesh.flatten()
    }
    
    # Add derivatives
    print("\nComputing derivatives for synthetic ideal gas data...")
    data = add_derivatives_to_table(data, n_B_grid, T_grid)
    
    # Check a sample point
    idx = len(data['n_B']) // 2
    print(f"\nSample point: n_B={data['n_B'][idx]:.3f} fm⁻³, T={data['T'][idx]:.1f} MeV")
    print(f"  cv = {data['cv'][idx]:.4f}")
    print(f"  cp = {data['cp'][idx]:.4f}")
    print(f"  Γ = cp/cv = {data['Gamma'][idx]:.4f}")
    print(f"  κT = {data['kappa_T'][idx]:.4e} MeV⁻¹fm³")
    print(f"  cs² = {data['cs2'][idx]:.4f} c²")
    
    # For ideal gas: Γ = 5/3, cs² ~ 1/γ for non-relativistic
    print(f"\nNote: For ideal monatomic gas, expect Γ ≈ 5/3 = {5/3:.4f}")
    
    print("\nOK!")
