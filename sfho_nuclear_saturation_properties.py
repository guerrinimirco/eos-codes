"""
sfho_nuclear_saturation_properties.py
=====================================
Compute nuclear saturation properties and hyperon potential depths
following Typel 2022 definitions.

Available functions:
- compute_saturation_fields(): Get σ, ω fields at given density
- compute_hyperon_potentials(): Compute U_Λ, U_Σ, U_Ξ from parametrization
- find_saturation_density(): Find n_sat where P = 0
- compute_all_nuclear_properties(): Compute n_sat, B_sat, K, Q, J, L, K_sym
- compute_symmetry_energy(): Compute E_sym(n_B) via Typel eq 6.5
- print_parametrization_summary(): Print summary table

Note: For creating custom parametrizations, use create_custom_parametrization()
from sfho_parameters.py instead.

Units:
- Energies/masses/potentials: MeV
- Densities: fm⁻³
"""
import numpy as np
from typing import Optional, Tuple, Dict
from sfho_parameters import (
    SFHoParams, get_sfho_nucleonic, get_sfhoy_fortin, 
    get_sfhoy_star_fortin, get_sfho_2fam_phi, get_sfho_2fam,
    SU6_RATIOS, SQRT2, _get_base_sfho
)


# =============================================================================
# CONSTANTS
# =============================================================================
N_SAT = 0.158  # fm^-3, saturation density


# =============================================================================
# COMPUTE SATURATION FIELDS
# =============================================================================
def compute_saturation_fields(params: Optional[SFHoParams] = None, 
                               n_B: float = N_SAT, 
                               Y_C: float = 0.5,
                               T: float = 0.01) -> Tuple[float, float, float, float]:
    """
    Compute meson fields (σ, ω, ρ, φ) at given density in nuclear matter.
    
    Args:
        params: SFHo parameters (defaults to nucleonic SFHo)
        n_B: Baryon density (fm⁻³)
        Y_C: Charge fraction (0.5 = symmetric nuclear matter)
        T: Temperature (MeV), use small T for T→0 limit
        
    Returns:
        (sigma, omega, rho, phi) fields in MeV
    """
    from sfho_eos import solve_sfho_fixed_yc, BARYONS_N
    
    if params is None:
        params = get_sfho_nucleonic()
    
    result = solve_sfho_fixed_yc(
        n_B=n_B, Y_C=Y_C, T=T, params=params, particles=BARYONS_N,
        include_electrons=False, include_photons=False
    )
    
    if not result.converged:
        raise RuntimeError(f"Failed to converge at n_B={n_B}, Y_C={Y_C}, T={T}")
    
    return result.sigma, result.omega, result.rho, result.phi


# =============================================================================
# COMPUTE HYPERON POTENTIAL DEPTHS
# =============================================================================
def compute_hyperon_potentials(params: SFHoParams, 
                                sigma: float = None, 
                                omega: float = None) -> Dict[str, float]:
    """
    Compute hyperon potential depths U_H^(N) at saturation in SNM.
    
    U_H = -g_σH × σ + g_ωH × ω
    
    Args:
        params: SFHo parameters with hyperon couplings
        sigma: σ field in MeV (if None, computed at n_sat)
        omega: ω field in MeV (if None, computed at n_sat)
        
    Returns:
        Dictionary with U_Λ, U_Σ, U_Ξ in MeV
    """
    if sigma is None or omega is None:
        sigma, omega, _, _ = compute_saturation_fields()
    
    potentials = {}
    
    for hyperon, label in [('lambda', 'U_Lambda'), 
                           ('sigma+', 'U_Sigma'), 
                           ('xi0', 'U_Xi')]:
        if hyperon in params.couplings_map:
            g_sigma_H = params.couplings_map[hyperon]['sigma']
            g_omega_H = params.couplings_map[hyperon]['omega']
            U_H = -g_sigma_H * sigma + g_omega_H * omega
            potentials[label] = U_H
        else:
            potentials[label] = None
            
    return potentials


# =============================================================================
# COMPUTE NUCLEAR MATTER PROPERTIES (TYPEL 2022 DEFINITIONS)
# =============================================================================
def find_saturation_density(params: Optional[SFHoParams] = None,
                             n_min: float = 0.14, 
                             n_max: float = 0.18) -> float:
    """
    Find saturation density n_sat where P = 0 in symmetric nuclear matter.
    
    Uses bisection to find where pressure vanishes (Typel 2022, eq 6.6).
    
    Returns:
        n_sat in fm⁻³
    """
    from scipy.optimize import brentq
    from sfho_eos import solve_sfho_fixed_yc, BARYONS_N
    
    if params is None:
        params = get_sfho_nucleonic()
    
    def pressure_at_density(n_B: float) -> float:
        result = solve_sfho_fixed_yc(
            n_B=n_B, Y_C=0.5, T=0.01, params=params, particles=BARYONS_N,
            include_electrons=False, include_photons=False
        )
        return result.P_total  # For hadrons-only, P_total = P_hadrons
    
    # Find where P = 0
    n_sat = brentq(pressure_at_density, n_min, n_max)
    return n_sat


def compute_energy_per_baryon(params: Optional[SFHoParams], n_B: float, Y_C: float = 0.5) -> float:
    """Compute energy per baryon ε = e/n_B - M_N at given density and charge fraction."""
    from sfho_eos import solve_sfho_fixed_yc, BARYONS_N
    
    if params is None:
        params = get_sfho_nucleonic()
    
    M_N = (params.m_n + params.m_p) / 2.0
    
    result = solve_sfho_fixed_yc(
        n_B=n_B, Y_C=Y_C, T=0.01, params=params, particles=BARYONS_N,
        include_electrons=False, include_photons=False
    )
    
    if not result.converged:
        raise RuntimeError(f"Failed to converge at n_B={n_B}, Y_C={Y_C}")
    
    # For hadrons-only calculation (no electrons/photons), e_total = e_hadrons
    epsilon = result.e_total / result.n_B - M_N
    return epsilon


def compute_pressure(params: Optional[SFHoParams], n_B: float, Y_C: float = 0.5) -> float:
    """Compute pressure at given density and charge fraction."""
    from sfho_eos import solve_sfho_fixed_yc, BARYONS_N
    
    if params is None:
        params = get_sfho_nucleonic()
    
    result = solve_sfho_fixed_yc(
        n_B=n_B, Y_C=Y_C, T=0.01, params=params, particles=BARYONS_N,
        include_electrons=False, include_photons=False
    )
    
    return result.P_total  # For hadrons-only, P_total = P_hadrons


def compute_symmetry_energy(params: Optional[SFHoParams], n_B: float) -> float:
    """
    Compute symmetry energy E_sym(n_B) following CompOSE/Typel 2022 eq 6.4.
    
    E_sym(n_B) = (1/2) × (∂²ε/∂α²)|_{α=0}
    
    Where α = 1 - 2Y_q (isospin asymmetry):
    - α = 0  → Y_C = 0.5 (symmetric nuclear matter, SNM)
    - α = 1  → Y_C = 0 (pure neutron matter, PNM)
    - α = -1 → Y_C = 1 (pure proton matter, PPM)
    
    Uses cubic spline interpolation over multiple α values for robust second derivative.
    """
    from scipy.interpolate import CubicSpline
    
    # Sample ε(n_B, α) at multiple α values for spline interpolation
    # Range: α from -0.8 to 0.8 (avoid extremes where numerics may be less stable)
    alpha_values = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    
    eps_values = []
    for alpha in alpha_values:
        Y_C = (1 - alpha) / 2.0  # α = 1 - 2Y_C → Y_C = (1-α)/2
        eps = compute_energy_per_baryon(params, n_B, Y_C=Y_C)
        eps_values.append(eps)
    
    eps_values = np.array(eps_values)
    
    # Create cubic spline for ε(α)
    spline = CubicSpline(alpha_values, eps_values, bc_type='natural')
    
    # Second derivative at α = 0
    d2eps_dalpha2 = spline.derivative(nu=2)
    
    # E_sym = (1/2) × (∂²ε/∂α²)|_{α=0}
    E_sym = 0.5 * float(d2eps_dalpha2(0.0))
    
    return E_sym


def compute_all_nuclear_properties(params: Optional[SFHoParams] = None) -> Dict[str, float]:
    """
    Compute all nuclear saturation properties following standard definitions.
    
    Properties computed:
        - n_sat: Saturation density where P = 0 (eq 6.6)
        - B_sat: Binding energy per nucleon at saturation, -ε(n_sat, 0) 
        - m_eff_ratio: Effective mass m*/M at saturation
        - K: Incompressibility K = 9n² (∂²ε/∂n²)|n_sat (eq 6.7)
        - Q: Skewness Q = 27n³ (∂³ε/∂n³)|n_sat (eq 6.9)
        - J: Symmetry energy at saturation J = E_PNM(n_sat) - E_SNM(n_sat)
        - L: Slope parameter L = 3n (dE_sym/dn)|n_sat (eq 6.11)
        - K_sym: Symmetry incompressibility K_sym = 9n² (d²E_sym/dn²)|n_sat
    
    Uses cubic spline interpolation for robust derivative calculation.
    
    Returns:
        Dictionary with all properties
    """
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    
    if params is None:
        params = get_sfho_nucleonic()
    
    M_N = (params.m_n + params.m_p) / 2.0
    
    # Create density grid for spline interpolation
    rho_arr = np.linspace(0.02, 0.30, 50)  # fm^-3
    
    print("  Computing E(n) for SNM and PNM...")
    E_snm_arr = np.array([compute_energy_per_baryon(params, n, Y_C=0.5) for n in rho_arr])
    E_pnm_arr = np.array([compute_energy_per_baryon(params, n, Y_C=0.0) for n in rho_arr])
    
    # Create cubic splines
    spline_snm = CubicSpline(rho_arr, E_snm_arr, bc_type='natural')
    spline_pnm = CubicSpline(rho_arr, E_pnm_arr, bc_type='natural')
    
    # Derivative of SNM energy
    dE_snm_dn = spline_snm.derivative(nu=1)
    d2E_snm_dn2 = spline_snm.derivative(nu=2)
    d3E_snm_dn3 = spline_snm.derivative(nu=3)
    
    # Find saturation density where dE/dn = 0 (equivalent to P = 0)
    a, b = 0.10, 0.25
    if dE_snm_dn(a) * dE_snm_dn(b) > 0:
        # Fall back to pressure-based method
        n_sat = find_saturation_density(params)
    else:
        n_sat = brentq(dE_snm_dn, a, b)
    
    # Energy and binding at saturation
    eps_sat = float(spline_snm(n_sat))
    B_sat = -eps_sat
    
    # Incompressibility K = 9n² (∂²ε/∂n²)
    K = 9.0 * n_sat**2 * float(d2E_snm_dn2(n_sat))
    
    # Skewness Q = 27n³ (∂³ε/∂n³)
    Q = 27.0 * n_sat**3 * float(d3E_snm_dn3(n_sat))
    
    # Meson fields at saturation
    sigma, omega, _, _ = compute_saturation_fields(params, n_B=n_sat, Y_C=0.5)
    m_eff = M_N - params.g_sigma_N * sigma
    m_eff_ratio = m_eff / M_N
    
    # =========================================================================
    # SYMMETRY ENERGY (standard definition: E_sym = E_PNM - E_SNM)
    # =========================================================================
    # This is the commonly used definition that matches CompOSE
    E_sym_arr = E_pnm_arr - E_snm_arr
    spline_sym = CubicSpline(rho_arr, E_sym_arr, bc_type='natural')
    
    dE_sym_dn = spline_sym.derivative(nu=1)
    d2E_sym_dn2 = spline_sym.derivative(nu=2)
    
    # J = E_sym(n_sat)
    J = float(spline_sym(n_sat))
    
    # L = 3n (dE_sym/dn)|n_sat
    L = 3.0 * n_sat * float(dE_sym_dn(n_sat))
    
    # K_sym = 9n² (d²E_sym/dn²)|n_sat
    K_sym = 9.0 * n_sat**2 * float(d2E_sym_dn2(n_sat))
    
    return {
        'n_sat': n_sat,
        'B_sat': B_sat,
        'sigma': sigma,
        'omega': omega,
        'm_eff_ratio': m_eff_ratio,
        'K': K,
        'Q': Q,
        'J': J,
        'L': L,
        'K_sym': K_sym,
        'M_N': M_N
    }


def compute_nuclear_properties(params: Optional[SFHoParams] = None) -> Dict[str, float]:
    """
    Compute nuclear matter properties at saturation.
    
    Returns dictionary with:
        - sigma, omega: meson fields at n_sat
        - m_eff_ratio: m*/M effective mass ratio
        - E_over_A: binding energy per nucleon
    """
    return compute_all_nuclear_properties(params)


# =============================================================================
# CREATE CUSTOM PARAMETRIZATION FROM POTENTIAL DEPTHS
# =============================================================================
def create_custom_parametrization(
    # Hyperon potential depths (MeV)
    U_Lambda_N: float = -30.0,
    U_Sigma_N: float = +30.0,
    U_Xi_N: float = -14.0,
    # Vector coupling enhancement factors (per hyperon family)
    # g_ωH = g_ωN × SU(6)_ratio × y_H, g_φH = g_ωN × SU(6)_ratio × y_H
    y_Lambda: float = 1.0,
    y_Sigma: float = 1.0,
    y_Xi: float = 1.0,
    # Delta couplings
    x_sigma_delta: float = 1.15,
    x_omega_delta: float = 1.0,
    x_rho_delta: float = 1.0,
    # Name
    name: str = "Custom"
) -> SFHoParams:
    """
    Create custom parametrization from target hyperon potential depths.
    
    The scalar coupling R_σH is determined from the target potential depth:
        U_H = -g_σH × σ + g_ωH × ω
        R_σH = (R_ωH × y_H × g_ωN × ω - U_H) / (g_σN × σ)
    
    Vector couplings follow SU(6) symmetry × y_H enhancement factor per family:
        g_ωΛ = g_ωN × (2/3) × y_Lambda
        g_ωΣ = g_ωN × (2/3) × y_Sigma  
        g_ωΞ = g_ωN × (1/3) × y_Xi
    
    Example: y_Lambda=1.5, y_Sigma=1.5, y_Xi=1.875 gives:
        g_ωΛ = 1.0 × g_ωN, g_ωΣ = 1.0 × g_ωN, g_ωΞ = 0.625 × g_ωN
    
    Args:
        U_Lambda_N: Λ potential depth at n_sat in SNM (MeV), ~ -30 MeV
        U_Sigma_N: Σ potential depth at n_sat in SNM (MeV), ~ +30 MeV  
        U_Xi_N: Ξ potential depth at n_sat in SNM (MeV), ~ +10 to -20 MeV
        y_Lambda: Enhancement factor for Λ (1.0 = SU(6))
        y_Sigma: Enhancement factor for Σ (1.0 = SU(6))
        y_Xi: Enhancement factor for Ξ (1.0 = SU(6))
        x_sigma_delta: R_σΔ = g_σΔ/g_σN
        x_omega_delta: R_ωΔ = g_ωΔ/g_ωN
        x_rho_delta: R_ρΔ = g_ρΔ/g_ρN
        name: Name for the parametrization
        
    Returns:
        SFHoParams with computed couplings
    """
    # Get base SFHo parameters
    p = _get_base_sfho()
    p.name = name
    
    # Compute saturation fields
    sigma, omega, _, _ = compute_saturation_fields()
    
    # SU(6) vector ratios (before enhancement)
    R_omega_Lambda_SU6 = 2.0/3.0
    R_omega_Sigma_SU6 = 2.0/3.0
    R_omega_Xi_SU6 = 1.0/3.0
    
    R_phi_Lambda_SU6 = -SQRT2/3.0
    R_phi_Sigma_SU6 = -SQRT2/3.0
    R_phi_Xi_SU6 = -2.0*SQRT2/3.0
    
    # Apply enhancement factors
    R_omega_Lambda = R_omega_Lambda_SU6 * y_Lambda
    R_omega_Sigma = R_omega_Sigma_SU6 * y_Sigma
    R_omega_Xi = R_omega_Xi_SU6 * y_Xi
    
    R_phi_Lambda = R_phi_Lambda_SU6 * y_Lambda
    R_phi_Sigma = R_phi_Sigma_SU6 * y_Sigma
    R_phi_Xi = R_phi_Xi_SU6 * y_Xi
    
    # Compute scalar coupling ratios from potential depths
    # U_H = -R_σH × g_σN × σ + R_ωH × g_ωN × ω
    # R_σH = (R_ωH × g_ωN × ω - U_H) / (g_σN × σ)
    
    def compute_R_sigma(U_H: float, R_omega: float) -> float:
        return (R_omega * p.g_omega_N * omega - U_H) / (p.g_sigma_N * sigma)
    
    R_sigma_Lambda = compute_R_sigma(U_Lambda_N, R_omega_Lambda)
    R_sigma_Sigma = compute_R_sigma(U_Sigma_N, R_omega_Sigma)
    R_sigma_Xi = compute_R_sigma(U_Xi_N, R_omega_Xi)
    
    # Lambda couplings
    p.couplings_map['lambda'] = {
        'sigma': R_sigma_Lambda * p.g_sigma_N,
        'omega': R_omega_Lambda * p.g_omega_N,
        'phi': R_phi_Lambda * p.g_omega_N,
        'rho': 0.0,
    }
    
    # Sigma couplings (all Σ+, Σ0, Σ-)
    sigma_couplings = {
        'sigma': R_sigma_Sigma * p.g_sigma_N,
        'omega': R_omega_Sigma * p.g_omega_N,
        'phi': R_phi_Sigma * p.g_omega_N,
        'rho': 2.0 * p.g_rho_N,
    }
    for s_name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[s_name] = sigma_couplings.copy()
    
    # Xi couplings
    xi_couplings = {
        'sigma': R_sigma_Xi * p.g_sigma_N,
        'omega': R_omega_Xi * p.g_omega_N,
        'phi': R_phi_Xi * p.g_omega_N,
        'rho': 1.0 * p.g_rho_N,
    }
    for x_name in ['xi0', 'xi-']:
        p.couplings_map[x_name] = xi_couplings.copy()
    
    # Delta couplings
    delta_couplings = {
        'sigma': x_sigma_delta * p.g_sigma_N,
        'omega': x_omega_delta * p.g_omega_N,
        'phi': 0.0,  # Deltas don't couple to φ
        'rho': x_rho_delta * p.g_rho_N,
    }
    for d_name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[d_name] = delta_couplings.copy()
    
    return p


# =============================================================================
# PRINT SUMMARY OF ALL PARAMETRIZATIONS
# =============================================================================
def print_parametrization_summary():
    """Print summary of all parametrizations including potentials."""
    
    print("="*80)
    print("NUCLEAR SATURATION PROPERTIES (Typel 2022 definitions)")
    print("="*80)
    
    # Compute all nuclear properties
    print("\nFinding saturation density (P = 0)...")
    props = compute_all_nuclear_properties()
    
    # Reference values from CompOSE / literature for SFHo
    # Note: CompOSE values are from tabulated EOS, may differ from uniform matter RMF
    refs = {
        'n_sat': 0.1583,   # fm^-3
        'B_sat': 16.19,    # MeV (E/A = -16.19 MeV)
        'm_eff_ratio': 0.76,
        'K': 245.0,        # MeV
        'Q': -467.0,       # MeV (approximate)
        'J': 31.57,        # MeV (CompOSE tabulated)
        'L': 47.10,        # MeV (CompOSE tabulated)
        'K_sym': -146.0,   # MeV (approximate)
    }
    
    print(f"\n{'Property':<20} {'This Model':>15} {'CompOSE Ref':>15} {'Diff':>12}")
    print("-"*65)
    print(f"{'n_sat (fm⁻³)':<20} {props['n_sat']:>15.4f} {refs['n_sat']:>15.4f} "
          f"{(props['n_sat']-refs['n_sat'])*1000:>+11.2f}×10⁻³")
    print(f"{'B_sat (MeV)':<20} {props['B_sat']:>15.2f} {refs['B_sat']:>15.2f} "
          f"{props['B_sat']-refs['B_sat']:>+12.2f}")
    print(f"{'m*/M':<20} {props['m_eff_ratio']:>15.4f} {refs['m_eff_ratio']:>15.2f} "
          f"{props['m_eff_ratio']-refs['m_eff_ratio']:>+12.4f}")
    print(f"{'K (MeV)':<20} {props['K']:>15.1f} {refs['K']:>15.1f} "
          f"{props['K']-refs['K']:>+12.1f}")
    print(f"{'Q (MeV)':<20} {props['Q']:>15.1f} {refs['Q']:>15.1f} "
          f"{props['Q']-refs['Q']:>+12.1f}")
    print("-"*65)
    print("Isospin properties (Typel eq 6.5: E_sym = ½[ε(PNM)-2ε(SNM)+ε(PPM)]):")
    print(f"{'J (MeV)':<20} {props['J']:>15.2f} {refs['J']:>15.2f} "
          f"{props['J']-refs['J']:>+12.2f}")
    print(f"{'L (MeV)':<20} {props['L']:>15.1f} {refs['L']:>15.1f} "
          f"{props['L']-refs['L']:>+12.1f}")
    print(f"{'K_sym (MeV)':<20} {props['K_sym']:>15.1f} {refs['K_sym']:>15.1f} "
          f"{props['K_sym']-refs['K_sym']:>+12.1f}")
    
    print(f"\nMeson fields at saturation:")
    print(f"  σ = {props['sigma']:.3f} MeV")
    print(f"  ω = {props['omega']:.3f} MeV")
    sigma = props['sigma']
    omega = props['omega']
    n_sat = props['n_sat']
    
    # All parametrizations
    parametrizations = {
        'SFHoY (Fortin)': get_sfhoy_fortin(),
        'SFHoY* (SU6)': get_sfhoy_star_fortin(),
        '2fam_phi': get_sfho_2fam_phi(),
        '2fam': get_sfho_2fam(),
    }
    
    print("\n" + "="*80)
    print("HYPERON COUPLING RATIOS AND POTENTIAL DEPTHS")
    print("="*80)
    print(f"\n{'Parametrization':<18} | {'R_σΛ':>6} {'R_σΣ':>6} {'R_σΞ':>6} | "
          f"{'U_Λ':>8} {'U_Σ':>8} {'U_Ξ':>8}")
    print("-"*80)
    
    for name, p in parametrizations.items():
        potentials = compute_hyperon_potentials(p, sigma, omega)
        
        R_sL = p.couplings_map.get('lambda', {}).get('sigma', 0) / p.g_sigma_N
        R_sS = p.couplings_map.get('sigma+', {}).get('sigma', 0) / p.g_sigma_N
        R_sX = p.couplings_map.get('xi0', {}).get('sigma', 0) / p.g_sigma_N
        
        U_L = potentials.get('U_Lambda', 0)
        U_S = potentials.get('U_Sigma', 0)
        U_X = potentials.get('U_Xi', 0)
        
        print(f"{name:<18} | {R_sL:>6.3f} {R_sS:>6.3f} {R_sX:>6.3f} | "
              f"{U_L:>+8.2f} {U_S:>+8.2f} {U_X:>+8.2f} MeV")
    
    print("\n" + "="*80)
    print("CUSTOM PARAMETRIZATION EXAMPLE")
    print("="*80)
    
    # Example: create custom param with target potentials and enhanced vectors
    custom = create_custom_parametrization(
        U_Lambda_N=-30.0,
        U_Sigma_N=+30.0,
        U_Xi_N=-14.0,
        y_Lambda=1.5,     # g_ωΛ = 1.5 × (2/3) × g_ωN = 1.0 × g_ωN
        y_Sigma=1.5,      # g_ωΣ = 1.5 × (2/3) × g_ωN = 1.0 × g_ωN
        y_Xi=1.875,       # g_ωΞ = 1.875 × (1/3) × g_ωN = 0.625 × g_ωN
        x_sigma_delta=0,
        name="Custom_Test"
    )
    
    potentials = compute_hyperon_potentials(custom, sigma, omega)
    print(f"\nCreated custom parametrization with target potentials:")
    print(f"  Resulting vector ratios:")
    
    R_wL = custom.couplings_map['lambda']['omega'] / custom.g_omega_N
    R_wS = custom.couplings_map['sigma+']['omega'] / custom.g_omega_N
    R_wX = custom.couplings_map['xi0']['omega'] / custom.g_omega_N
    
    print(f"    R_ωΛ = {R_wL:.4f}")
    print(f"    R_ωΣ = {R_wS:.4f}")
    print(f"    R_ωΞ = {R_wX:.4f}")
    
    print(f"  Computed R_σ values:")
    
    R_sL = custom.couplings_map['lambda']['sigma'] / custom.g_sigma_N
    R_sS = custom.couplings_map['sigma+']['sigma'] / custom.g_sigma_N
    R_sX = custom.couplings_map['xi0']['sigma'] / custom.g_sigma_N
    
    print(f"    R_σΛ = {R_sL:.4f}")
    print(f"    R_σΣ = {R_sS:.4f}")
    print(f"    R_σΞ = {R_sX:.4f}")
    
    print(f"  Verification - computed potentials:")
    print(f"    U_Λ = {potentials['U_Lambda']:+.2f} MeV (target: -30)")
    print(f"    U_Σ = {potentials['U_Sigma']:+.2f} MeV (target: +30)")
    print(f"    U_Ξ = {potentials['U_Xi']:+.2f} MeV (target: -18)")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print_parametrization_summary()

