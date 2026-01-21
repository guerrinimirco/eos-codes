"""
dd2_parameters.py
==================
DD2 Relativistic Mean Field model parameters with density-dependent couplings.

Contains parametrizations for:
- Nucleonic matter DD2 (Typel et al. 2010)
- Hyperonic matter DD2Y (Fortin et al. 2017, Marques et al. 2017)
- Alternative hyperonic parametrizations

Key difference from SFHo:
- DD2 uses DENSITY-DEPENDENT couplings: g_M(n_B) = g_M(n_0) × h_M(x), x = n_B/n_0
- No nonlinear self-interactions (g2 = g3 = c3 = c4 = 0)
- Requires rearrangement term R^0 for thermodynamic consistency

Units:
- Masses: MeV
- Couplings: dimensionless (at saturation)
- Length: fm
- Density: fm^-3

References:
- Typel, Röpke, Klähn, Blaschke, Wolter, Phys. Rev. C 81 (2010) 015803
- Marques, Oertel, Hempel, Novak, Phys. Rev. C 96 (2017) 045806
- Fortin, Oertel, Providência, PASA 35 (2018) e044
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Callable
from general_physics_constants import hc

# =============================================================================
# SU(6) SYMMETRY RATIOS FOR VECTOR MESONS
# =============================================================================
SQRT2 = np.sqrt(2.0)

SU6_RATIOS = {
    'lambda': {
        'omega': 2.0/3.0,
        'rho': 0.0,
        'phi': -SQRT2/3.0,
    },
    'sigma': {
        'omega': 2.0/3.0,
        'rho': 2.0,
        'phi': -SQRT2/3.0,
    },
    'xi': {
        'omega': 1.0/3.0,
        'rho': 1.0,
        'phi': -2.0*SQRT2/3.0,
    },
    'delta': {
        'omega': 1.0,
        'rho': 1.0,
        'phi': 0.0,
    },
}


# =============================================================================
# DENSITY-DEPENDENT COUPLING FUNCTIONS (Typel et al. 2010)
# =============================================================================

def make_isoscalar_h_function(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
    """
    Create isoscalar density-dependent function h_M(x).
    
    From Eq. (4) of Fortin et al. 2017 / Typel et al. 2010:
        h_M(x) = a_M × (1 + b_M(x + d_M)²) / (1 + c_M(x + d_M)²)
    
    where x = n_B / n_0 (ratio to saturation density).
    
    At x = 1 (saturation): h_M(1) = a × (1 + b(1+d)²) / (1 + c(1+d)²)
    
    Args:
        a, b, c, d: Typel function parameters
        
    Returns:
        Function h(x) giving the density dependence
    """
    def h_func(x: float) -> float:
        xd = x + d
        xd2 = xd * xd
        return a * (1.0 + b * xd2) / (1.0 + c * xd2)
    return h_func


def make_isovector_h_function(a: float) -> Callable[[float], float]:
    """
    Create isovector density-dependent function h_ρ(x).
    
    From Eq. (5) of Fortin et al. 2017 / Typel et al. 2010:
        h_ρ(x) = exp[-a_ρ(x - 1)]
    
    At x = 1 (saturation): h_ρ(1) = 1
    
    Args:
        a: Exponential decay parameter a_ρ
        
    Returns:
        Function h(x) giving the density dependence
    """
    def h_func(x: float) -> float:
        return np.exp(-a * (x - 1.0))
    return h_func


def make_derivative_isoscalar_h(a: float, b: float, c: float, d: float) -> Callable[[float], float]:
    """
    Derivative of isoscalar h_M(x) with respect to x.
    
    dh/dx = a × 2(x+d) × (b - c) / (1 + c(x+d)²)²
    
    Needed for rearrangement term calculation.
    """
    def dh_func(x: float) -> float:
        xd = x + d
        xd2 = xd * xd
        denom = 1.0 + c * xd2
        return a * 2.0 * xd * (b - c) / (denom * denom)
    return dh_func


def make_derivative_isovector_h(a: float) -> Callable[[float], float]:
    """
    Derivative of isovector h_ρ(x) with respect to x.
    
    dh/dx = -a × exp[-a(x-1)]
    """
    def dh_func(x: float) -> float:
        return -a * np.exp(-a * (x - 1.0))
    return dh_func


# =============================================================================
# PARAMETER DATACLASS
# =============================================================================
@dataclass
class DD2Params:
    """
    Holds the parameters for the DD2 Relativistic Mean Field model
    with density-dependent couplings.
    
    The model features:
    - Density-dependent nucleon-meson couplings g_M(n_B)
    - No nonlinear self-interactions
    - Rearrangement term R^0 for thermodynamic consistency
    - Hyperon/Delta couplings via ratio maps
    
    Field equations (mean-field approximation):
        m_σ² σ = Σⱼ g_σⱼ(n_B) n^s_j × (ℏc)³
        m_ω² ω = Σⱼ g_ωⱼ(n_B) nⱼ × (ℏc)³
        m_ρ² ρ = Σⱼ g_ρⱼ(n_B) I₃ⱼ nⱼ × (ℏc)³
        m_φ² φ = Σⱼ g_φⱼ(n_B) nⱼ × (ℏc)³
    
    Effective chemical potential includes rearrangement:
        μ*_j = μ_j - g_ωⱼ ω - g_ρⱼ I₃ⱼ ρ - g_φⱼ φ - R^0
    
    Rearrangement term (Eq. 15 of Fortin 2017):
        R^0 = Σⱼ [∂g_ωⱼ/∂n_B × ω × nⱼ + ∂g_ρⱼ/∂n_B × ρ × I₃ⱼ × nⱼ 
                + ∂g_φⱼ/∂n_B × φ × nⱼ - ∂g_σⱼ/∂n_B × σ × n^s_j]
    """
    # Name identifier
    name: str = "DD2"
    
    # ---------------------------------------------------------
    # 1. Baryon Masses (MeV) - used in RMF calculations
    # ---------------------------------------------------------
    # Nucleons (DD2Y values from Fortin 2017)
    m_n: float = 939.565346    # Neutron mass
    m_p: float = 938.272013    # Proton mass
    
    # Hyperons (DD2Y values from Fortin 2017 - slightly different from SFHo)
    m_lambda: float = 1115.683    # Λ
    m_sigma_p: float = 1190.0     # Σ⁺
    m_sigma_0: float = 1190.0     # Σ⁰ (using degenerate mass)
    m_sigma_m: float = 1190.0     # Σ⁻
    m_xi_0: float = 1314.83       # Ξ⁰
    m_xi_m: float = 1321.68       # Ξ⁻
    
    # Delta resonances
    m_delta: float = 1232.0
    
    # ---------------------------------------------------------
    # 2. Meson Masses (MeV)
    # ---------------------------------------------------------
    m_sigma: float = 546.212453   # σ (scalar-isoscalar)
    m_omega: float = 783.0        # ω (vector-isoscalar)
    m_rho: float = 763.0          # ρ (vector-isovector)
    m_phi: float = 1020.0         # φ (vector-isoscalar, hidden strangeness)
    
    # ---------------------------------------------------------
    # 3. Pseudoscalar Meson Masses (MeV) - for thermal contributions
    # ---------------------------------------------------------
    m_pi_pm: float = 139.570
    m_pi_0: float = 134.977
    m_kaon_pm: float = 493.677
    m_kaon_0: float = 497.611
    m_eta: float = 547.862
    m_eta_prime: float = 957.78
    
    # ---------------------------------------------------------
    # 4. Nucleon Couplings at Saturation (dimensionless)
    # ---------------------------------------------------------
    # These are g_MN(n_0), the coupling values at saturation density
    g_sigma_N: float = 0.0
    g_omega_N: float = 0.0
    g_rho_N: float = 0.0
    g_phi_N: float = 0.0   # Usually 0 for nucleons
    
    # ---------------------------------------------------------
    # 5. Saturation Density (fm⁻³)
    # ---------------------------------------------------------
    n_sat: float = 0.149065     # DD2 saturation density
    
    # ---------------------------------------------------------
    # 6. Density-Dependent Function Parameters (Typel et al. 2010)
    # ---------------------------------------------------------
    # Isoscalar σ parameters: h_σ(x) = a_σ(1 + b_σ(x+d_σ)²)/(1 + c_σ(x+d_σ)²)
    a_sigma: float = 1.357630
    b_sigma: float = 0.634442
    c_sigma: float = 1.005358
    d_sigma: float = 0.575810
    
    # Isoscalar ω parameters
    a_omega: float = 1.369718
    b_omega: float = 0.496475
    c_omega: float = 0.817753
    d_omega: float = 0.638452
    
    # Isovector ρ parameter: h_ρ(x) = exp[-a_ρ(x-1)]
    a_rho: float = 0.518903
    
    # φ uses same form as ω (or constant = 1 if no density dependence)
    # For simplicity, assume φ has no density dependence for nucleons
    
    # ---------------------------------------------------------
    # 7. Coupling Ratios for Hyperons/Deltas
    # ---------------------------------------------------------
    # couplings_map[particle_name][meson] = ratio relative to nucleon coupling
    couplings_map: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize density-dependent functions after dataclass creation."""
        # Create the h functions
        self._h_sigma = make_isoscalar_h_function(
            self.a_sigma, self.b_sigma, self.c_sigma, self.d_sigma)
        self._h_omega = make_isoscalar_h_function(
            self.a_omega, self.b_omega, self.c_omega, self.d_omega)
        self._h_rho = make_isovector_h_function(self.a_rho)
        
        # Create derivative functions
        self._dh_sigma = make_derivative_isoscalar_h(
            self.a_sigma, self.b_sigma, self.c_sigma, self.d_sigma)
        self._dh_omega = make_derivative_isoscalar_h(
            self.a_omega, self.b_omega, self.c_omega, self.d_omega)
        self._dh_rho = make_derivative_isovector_h(self.a_rho)
    
    def h_sigma(self, n_B: float) -> float:
        """Density dependence function for σ coupling."""
        x = n_B / self.n_sat
        return self._h_sigma(x)
    
    def h_omega(self, n_B: float) -> float:
        """Density dependence function for ω coupling."""
        x = n_B / self.n_sat
        return self._h_omega(x)
    
    def h_rho(self, n_B: float) -> float:
        """Density dependence function for ρ coupling."""
        x = n_B / self.n_sat
        return self._h_rho(x)
    
    def h_phi(self, n_B: float) -> float:
        """Density dependence function for φ coupling (constant = 1)."""
        return 1.0
    
    def dh_sigma_dn(self, n_B: float) -> float:
        """Derivative dh_σ/dn_B for rearrangement term."""
        x = n_B / self.n_sat
        return self._dh_sigma(x) / self.n_sat
    
    def dh_omega_dn(self, n_B: float) -> float:
        """Derivative dh_ω/dn_B for rearrangement term."""
        x = n_B / self.n_sat
        return self._dh_omega(x) / self.n_sat
    
    def dh_rho_dn(self, n_B: float) -> float:
        """Derivative dh_ρ/dn_B for rearrangement term."""
        x = n_B / self.n_sat
        return self._dh_rho(x) / self.n_sat
    
    def dh_phi_dn(self, n_B: float) -> float:
        """Derivative dh_φ/dn_B (zero since constant)."""
        return 0.0
    
    def get_coupling(self, particle_name: str, meson: str, n_B: float = None) -> float:
        """
        Returns the meson-baryon coupling constant g_{MB}(n_B).
        
        For density-dependent couplings:
            g_M(n_B) = g_M(n_0) × h_M(x) × R_M
        
        where R_M is the ratio for hyperons/deltas (1.0 for nucleons).
        
        Args:
            particle_name: Particle name (case-insensitive)
            meson: 'sigma', 'omega', 'rho', or 'phi'
            n_B: Baryon density (fm⁻³). If None, returns coupling at saturation.
            
        Returns:
            Coupling constant (dimensionless)
        """
        p_name = particle_name.lower()
        
        # Get base coupling at saturation
        if meson == 'sigma':
            g_sat = self.g_sigma_N
        elif meson == 'omega':
            g_sat = self.g_omega_N
        elif meson == 'rho':
            g_sat = self.g_rho_N
        elif meson == 'phi':
            g_sat = self.g_phi_N
        else:
            return 0.0
        
        # Get ratio for non-nucleons
        ratio = 1.0
        if p_name in ['n', 'p', 'neutron', 'proton']:
            ratio = 1.0
        elif p_name in self.couplings_map:
            if meson in self.couplings_map[p_name]:
                # couplings_map stores the ratio R_MH
                ratio = self.couplings_map[p_name][meson]
        
        # Apply density dependence if n_B provided
        if n_B is not None and n_B > 0:
            if meson == 'sigma':
                h = self.h_sigma(n_B)
            elif meson == 'omega':
                h = self.h_omega(n_B)
            elif meson == 'rho':
                h = self.h_rho(n_B)
            elif meson == 'phi':
                h = self.h_phi(n_B)
            else:
                h = 1.0
        else:
            h = 1.0
        
        return g_sat * h * ratio
    
    def get_coupling_derivative(self, particle_name: str, meson: str, n_B: float) -> float:
        """
        Returns ∂g_{MB}/∂n_B for rearrangement term calculation.
        
        ∂g_M/∂n_B = g_M(n_0) × (dh_M/dn_B) × R_M
        
        Args:
            particle_name: Particle name
            meson: 'sigma', 'omega', 'rho', or 'phi'
            n_B: Baryon density (fm⁻³)
            
        Returns:
            Derivative of coupling w.r.t. n_B (fm³)
        """
        p_name = particle_name.lower()
        
        # Get base coupling at saturation
        if meson == 'sigma':
            g_sat = self.g_sigma_N
            dh_dn = self.dh_sigma_dn(n_B)
        elif meson == 'omega':
            g_sat = self.g_omega_N
            dh_dn = self.dh_omega_dn(n_B)
        elif meson == 'rho':
            g_sat = self.g_rho_N
            dh_dn = self.dh_rho_dn(n_B)
        elif meson == 'phi':
            g_sat = self.g_phi_N
            dh_dn = self.dh_phi_dn(n_B)
        else:
            return 0.0
        
        # Get ratio for non-nucleons
        ratio = 1.0
        if p_name in ['n', 'p', 'neutron', 'proton']:
            ratio = 1.0
        elif p_name in self.couplings_map:
            if meson in self.couplings_map[p_name]:
                ratio = self.couplings_map[p_name][meson]
        
        return g_sat * dh_dn * ratio
    
    def get_baryon_mass(self, particle_name: str) -> float:
        """Get the baryon mass for RMF calculations."""
        p_name = particle_name.lower()
        
        if p_name in ['p', 'proton']:
            return self.m_p
        elif p_name in ['n', 'neutron']:
            return self.m_n
        elif p_name == 'lambda':
            return self.m_lambda
        elif p_name in ['sigma+', 'sigmap']:
            return self.m_sigma_p
        elif p_name in ['sigma0']:
            return self.m_sigma_0
        elif p_name in ['sigma-', 'sigmam']:
            return self.m_sigma_m
        elif p_name in ['xi0']:
            return self.m_xi_0
        elif p_name in ['xi-', 'xim']:
            return self.m_xi_m
        elif p_name.startswith('delta'):
            return self.m_delta
        else:
            return 0.0
    
    def get_meson_mass(self, meson_name: str) -> float:
        """Get pseudoscalar meson mass for thermal calculations."""
        m_name = meson_name.lower()
        
        if m_name in ['pi+', 'pi-', 'pip', 'pim']:
            return self.m_pi_pm
        elif m_name in ['pi0']:
            return self.m_pi_0
        elif m_name in ['k+', 'k-', 'kp', 'km', 'kaon+', 'kaon-']:
            return self.m_kaon_pm
        elif m_name in ['k0', 'k0_bar', 'kaon0', 'kaon0bar']:
            return self.m_kaon_0
        elif m_name == 'eta':
            return self.m_eta
        elif m_name in ['eta_prime', 'etaprime', "eta'"]:
            return self.m_eta_prime
        else:
            return 0.0


# =============================================================================
# BASE DD2 PARAMETERS (Typel et al. 2010)
# =============================================================================
def _get_base_dd2() -> DD2Params:
    """
    Returns base DD2 nuclear parameters from Typel et al. 2010.
    
    Nuclear matter properties at saturation:
    - n_sat = 0.149065 fm⁻³
    - E_0 = 16.02 MeV
    - K = 242.7 MeV
    - J = 31.67 MeV
    - L = 55.03 MeV
    """
    p = DD2Params(name="DD2")
    
    # Saturation density
    p.n_sat = 0.149065  # fm⁻³
    
    # Meson masses (MeV)
    p.m_sigma = 546.212453
    p.m_omega = 783.0
    p.m_rho = 763.0
    p.m_phi = 1020.0
    
    # Couplings at saturation (from Typel et al. 2010)
    # These are dimensionless g values
    p.g_sigma_N = 10.686681
    p.g_omega_N = 13.342362
    p.g_rho_N = 3.626940
    p.g_phi_N = 0.0
    
    # Density-dependent function parameters (Typel et al. 2010, Table I)
    # σ meson
    p.a_sigma = 1.357630
    p.b_sigma = 0.634442
    p.c_sigma = 1.005358
    p.d_sigma = 0.575810
    
    # ω meson
    p.a_omega = 1.369718
    p.b_omega = 0.496475
    p.c_omega = 0.817753
    p.d_omega = 0.638452
    
    # ρ meson (exponential form)
    p.a_rho = 0.518903
    
    # Reinitialize h functions with new parameters
    p.__post_init__()
    
    return p


# =============================================================================
# PARAMETRIZATION FACTORY FUNCTIONS
# =============================================================================

def get_dd2_nucleonic() -> DD2Params:
    """
    DD2 with nucleons only (Typel et al. 2010).
    
    Use for pure nuclear matter calculations.
    """
    p = _get_base_dd2()
    p.name = "DD2_Nucleonic"
    return p


def get_dd2y_fortin() -> DD2Params:
    """
    DD2Y parametrization from Marques et al. 2017 / Fortin et al. 2017.
    
    Features:
    - SU(6) vector couplings for hyperons
    - Scalar couplings from hyperon potential depths
    - Supports M_max ≈ 2.04 M_sun for cold NS
    
    Hyperon potential depths at saturation (SNM):
    - U_Λ^(N) = -30 MeV
    - U_Σ^(N) = +30 MeV  (repulsive)
    - U_Ξ^(N) = -18 MeV
    
    Coupling ratios from Table 1 of Fortin 2017:
    - R_σΛ = 0.62, R_ωΛ = 2/3, R_φΛ = -√2/3 × g_ωN
    - R_σΣ = 0.48, R_ωΣ = 2/3, R_φΣ = -√2/3 × g_ωN, R_ρΣ = 2
    - R_σΞ = 0.32, R_ωΞ = 1/3, R_φΞ = -2√2/3 × g_ωN, R_ρΞ = 1
    """
    p = _get_base_dd2()
    p.name = "DD2Y_Fortin"
    
    # From Table 1 of Fortin 2017 (DD2Y row)
    # R_σ values are computed from potential depths
    
    # Lambda (uds)
    p.couplings_map['lambda'] = {
        'sigma': 0.62,                    # R_σΛ
        'omega': 2.0/3.0,                 # R_ωΛ (SU(6))
        'phi': -SQRT2/3.0,                # R_φΛ (relative to g_ωN)
        'rho': 0.0,                       # R_ρΛ (isospin singlet)
    }
    
    # Sigma (uus, uds, dds)
    sigma_couplings = {
        'sigma': 0.48,                    # R_σΣ
        'omega': 2.0/3.0,                 # R_ωΣ (SU(6))
        'phi': -SQRT2/3.0,                # R_φΣ
        'rho': 2.0,                       # R_ρΣ
    }
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = sigma_couplings.copy()
    
    # Xi (uss, dss)
    xi_couplings = {
        'sigma': 0.32,                    # R_σΞ
        'omega': 1.0/3.0,                 # R_ωΞ (SU(6))
        'phi': -2.0*SQRT2/3.0,            # R_φΞ
        'rho': 1.0,                       # R_ρΞ
    }
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = xi_couplings.copy()
    
    return p


def get_dd2y_with_deltas(
    x_sigma_delta: float = 1.15,
    x_omega_delta: float = 1.0,
    x_rho_delta: float = 1.0
) -> DD2Params:
    """
    DD2Y with Delta resonances.
    
    Args:
        x_sigma_delta: Ratio R_σΔ (default 1.15 from quark counting)
        x_omega_delta: Ratio R_ωΔ (default 1.0)
        x_rho_delta: Ratio R_ρΔ (default 1.0)
    """
    p = get_dd2y_fortin()
    p.name = "DD2Y_Deltas"
    
    # Delta couplings
    delta_couplings = {
        'sigma': x_sigma_delta,
        'omega': x_omega_delta,
        'phi': 0.0,  # No strangeness
        'rho': x_rho_delta,
    }
    for name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[name] = delta_couplings.copy()
    
    return p


def create_custom_dd2_parametrization(
    # Hyperon potential depths (MeV) at saturation
    U_Lambda_N: float = -30.0,
    U_Sigma_N: float = +30.0,
    U_Xi_N: float = -18.0,
    # Vector coupling enhancement (1.0 = SU(6))
    y_Lambda: float = 1.0,
    y_Sigma: float = 1.0,
    y_Xi: float = 1.0,
    # Delta couplings
    x_sigma_delta: float = 1.15,
    x_omega_delta: float = 1.0,
    x_rho_delta: float = 1.0,
    # Name
    name: str = "Custom_DD2"
) -> DD2Params:
    """
    Create custom DD2 parametrization from target hyperon potential depths.
    
    The scalar coupling R_σH is computed from the target potential depth:
        U_H = -g_σH × σ + g_ωH × ω  at n_sat, Y_C = 0.5
        R_σH = (R_ωH × g_ωN × ω - U_H) / (g_σN × σ)
    
    Vector couplings follow SU(6) × y_H enhancement.
    
    Args:
        U_Lambda_N, U_Sigma_N, U_Xi_N: Potential depths (MeV)
        y_Lambda, y_Sigma, y_Xi: Vector coupling enhancement factors
        x_sigma_delta, x_omega_delta, x_rho_delta: Delta coupling ratios
        name: Parametrization name
    """
    p = _get_base_dd2()
    p.name = name
    
    # Saturation fields for DD2 at n_sat = 0.149 fm⁻³ (approximate values)
    # These would need to be solved self-consistently for exact values
    SIGMA_SAT = 58.0  # MeV (approximate)
    OMEGA_SAT = 74.0  # MeV (approximate)
    
    # SU(6) vector ratios with enhancement
    R_omega_Lambda = (2.0/3.0) * y_Lambda
    R_omega_Sigma = (2.0/3.0) * y_Sigma
    R_omega_Xi = (1.0/3.0) * y_Xi
    
    R_phi_Lambda = (-SQRT2/3.0) * y_Lambda
    R_phi_Sigma = (-SQRT2/3.0) * y_Sigma
    R_phi_Xi = (-2.0*SQRT2/3.0) * y_Xi
    
    # Compute R_σ from potential depth
    # U_H(n_sat) = -g_σH × σ + g_ωH × ω
    # U_H = -R_σH × g_σN × σ + R_ωH × g_ωN × ω
    # R_σH = (R_ωH × g_ωN × ω - U_H) / (g_σN × σ)
    def compute_R_sigma(U_H: float, R_omega: float) -> float:
        return (R_omega * p.g_omega_N * OMEGA_SAT - U_H) / (p.g_sigma_N * SIGMA_SAT)
    
    R_sigma_Lambda = compute_R_sigma(U_Lambda_N, R_omega_Lambda)
    R_sigma_Sigma = compute_R_sigma(U_Sigma_N, R_omega_Sigma)
    R_sigma_Xi = compute_R_sigma(U_Xi_N, R_omega_Xi)
    
    # Lambda
    p.couplings_map['lambda'] = {
        'sigma': R_sigma_Lambda,
        'omega': R_omega_Lambda,
        'phi': R_phi_Lambda,
        'rho': 0.0,
    }
    
    # Sigma
    sigma_couplings = {
        'sigma': R_sigma_Sigma,
        'omega': R_omega_Sigma,
        'phi': R_phi_Sigma,
        'rho': 2.0,
    }
    for s_name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[s_name] = sigma_couplings.copy()
    
    # Xi
    xi_couplings = {
        'sigma': R_sigma_Xi,
        'omega': R_omega_Xi,
        'phi': R_phi_Xi,
        'rho': 1.0,
    }
    for x_name in ['xi0', 'xi-']:
        p.couplings_map[x_name] = xi_couplings.copy()
    
    # Delta
    delta_couplings = {
        'sigma': x_sigma_delta,
        'omega': x_omega_delta,
        'phi': 0.0,
        'rho': x_rho_delta,
    }
    for d_name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[d_name] = delta_couplings.copy()
    
    return p


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_params_summary(params: DD2Params) -> None:
    """Print a summary of the DD2 parametrization."""
    print(f"Parametrization: {params.name}")
    print("=" * 60)
    print(f"\nSaturation density: n_sat = {params.n_sat:.6f} fm⁻³")
    
    print("\nNucleon couplings (at saturation):")
    print(f"  g_σN = {params.g_sigma_N:.6f}")
    print(f"  g_ωN = {params.g_omega_N:.6f}")
    print(f"  g_ρN = {params.g_rho_N:.6f}")
    print(f"  g_φN = {params.g_phi_N:.6f}")
    
    print("\nDensity-dependent function parameters:")
    print(f"  σ: a={params.a_sigma:.6f}, b={params.b_sigma:.6f}, "
          f"c={params.c_sigma:.6f}, d={params.d_sigma:.6f}")
    print(f"  ω: a={params.a_omega:.6f}, b={params.b_omega:.6f}, "
          f"c={params.c_omega:.6f}, d={params.d_omega:.6f}")
    print(f"  ρ: a={params.a_rho:.6f}")
    
    # Test h functions at saturation
    print(f"\nh functions at n_sat:")
    print(f"  h_σ(n_sat) = {params.h_sigma(params.n_sat):.6f}")
    print(f"  h_ω(n_sat) = {params.h_omega(params.n_sat):.6f}")
    print(f"  h_ρ(n_sat) = {params.h_rho(params.n_sat):.6f}")
    
    if params.couplings_map:
        print("\nHyperon/Delta coupling ratios:")
        for particle, couplings in params.couplings_map.items():
            Rs = couplings.get('sigma', 0)
            Rw = couplings.get('omega', 0)
            Rr = couplings.get('rho', 0)
            Rp = couplings.get('phi', 0)
            print(f"  {particle:10s}: R_σ={Rs:.3f}, R_ω={Rw:.3f}, "
                  f"R_ρ={Rr:.3f}, R_φ={Rp:.3f}")


def get_all_parametrizations() -> Dict[str, DD2Params]:
    """Return dictionary of all available DD2 parametrizations."""
    return {
        'DD2_Nucleonic': get_dd2_nucleonic(),
        'DD2Y_Fortin': get_dd2y_fortin(),
        'DD2Y_Deltas': get_dd2y_with_deltas(),
    }


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("DD2 Parameters Module (Density-Dependent Couplings)")
    print("=" * 70)
    
    # Test all parametrizations
    params_dict = get_all_parametrizations()
    
    for name, params in params_dict.items():
        print(f"\n{'='*70}")
        print_params_summary(params)
    
    # Test density dependence
    print("\n" + "=" * 70)
    print("Testing density dependence of couplings")
    print("-" * 50)
    
    p = get_dd2_nucleonic()
    densities = [0.5, 1.0, 2.0, 3.0]  # multiples of n_sat
    
    print(f"{'n_B/n_sat':<12} {'h_σ':<12} {'h_ω':<12} {'h_ρ':<12}")
    print("-" * 50)
    for x in densities:
        n_B = x * p.n_sat
        print(f"{x:<12.1f} {p.h_sigma(n_B):<12.6f} "
              f"{p.h_omega(n_B):<12.6f} {p.h_rho(n_B):<12.6f}")
    
    # Test coupling retrieval
    print("\n" + "=" * 70)
    print("Testing coupling retrieval for DD2Y_Fortin:")
    print("-" * 50)
    p = get_dd2y_fortin()
    test_particles = ['proton', 'neutron', 'lambda', 'sigma+', 'xi-']
    n_test = p.n_sat
    
    print(f"At n_B = n_sat = {n_test:.4f} fm⁻³:")
    print(f"{'Particle':<12} {'g_σ':<10} {'g_ω':<10} {'g_ρ':<10} {'g_φ':<10}")
    print("-" * 54)
    for part in test_particles:
        gs = p.get_coupling(part, 'sigma', n_test)
        gw = p.get_coupling(part, 'omega', n_test)
        gr = p.get_coupling(part, 'rho', n_test)
        gp = p.get_coupling(part, 'phi', n_test)
        print(f"{part:<12} {gs:>10.4f} {gw:>10.4f} {gr:>10.4f} {gp:>10.4f}")
