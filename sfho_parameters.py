"""
sfho_parameters.py
==================
SFHo Relativistic Mean Field model parameters.

Contains parametrizations for:
- Nucleonic matter (Steiner et al. 2013)
- Hyperonic matter SFHoY (Fortin et al. 2017)
- Hyperonic matter SFHoY* with SU(6) vector couplings (Fortin et al. 2017)
- Hyperons + Deltas (SFHo_HD) as in Mathematica notebook
- General parametrization with customizable scalar couplings

Units:
- Masses: MeV
- Couplings: dimensionless (g)
- Length: fm

References:
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
- Fortin, Oertel, Providência, PASA 35 (2018) e044
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict
from general_physics_constants import hc


# =============================================================================
# SU(6) SYMMETRY RATIOS FOR VECTOR MESONS
# =============================================================================
# These are the standard SU(6) ratios relative to nucleon couplings
# Based on quark counting and ideal ω-φ mixing

SQRT2 = np.sqrt(2.0)

# SU(6) ratios: R_M = g_MH / g_MN (or g_MH / g_ωN for φ)
SU6_RATIOS = {
    # Lambda (uds): I=0, S=1 (in our convention)
    'lambda': {
        'omega': 2.0/3.0,
        'rho': 0.0,
        'phi': -SQRT2/3.0,  # ≈ -0.4714
    },
    # Sigma (uus, uds, dds): I=1, S=1
    'sigma': {
        'omega': 2.0/3.0,
        'rho': 2.0,  # Isospin factor
        'phi': -SQRT2/3.0,
    },
    # Xi (uss, dss): I=1/2, S=2
    'xi': {
        'omega': 1.0/3.0,
        'rho': 1.0,
        'phi': -2.0*SQRT2/3.0,  # ≈ -0.9428
    },
    # Delta (uuu, uud, udd, ddd): I=3/2, S=0
    'delta': {
        'omega': 1.0,
        'rho': 1.0,
        'phi': 0.0,  # No strangeness
    },
}


# =============================================================================
# PARAMETER DATACLASS
# =============================================================================
@dataclass
class SFHoParams:
    """
    Holds the parameters for the SFHo Relativistic Mean Field model.
    
    The model includes:
    - Nucleon-meson couplings
    - Non-linear σ-meson self-interactions
    - Vector meson self-interactions (ω, ρ)
    - Symmetry energy A-function (σ-ω-ρ mixing)
    - Hyperon/Delta couplings via ratio maps
    
    Note on masses:
    - Baryon masses here are the ones used in RMF calculations
    - They may differ slightly from PDG values in particles.py
    - particles.py contains reference PDG masses
    - These parametrization masses should be used for thermodynamics
    
    Field equations (mean-field approximation):
        m_σ² σ + g₂σ² + g₃σ³ = Σⱼ g_σⱼ n^s_j × (ℏc)³ + ∂A/∂σ ρ²
        m_ω² ω + c₃ω³ = Σⱼ g_ωⱼ nⱼ × (ℏc)³ - ∂A/∂ω ρ²
        m_ρ² ρ + c₄ρ³ + 2Aρ = Σⱼ g_ρⱼ I₃ⱼ nⱼ × (ℏc)³
        m_φ² φ = Σⱼ g_φⱼ nⱼ × (ℏc)³
    """
    # Name identifier for the parametrization
    name: str = "SFHo"
    
    # ---------------------------------------------------------
    # 1. Baryon Masses (MeV) - used in RMF calculations
    # ---------------------------------------------------------
    # Nucleons
    m_n: float = 939.565346   # Neutron mass
    m_p: float = 938.272013   # Proton mass
    
    # Hyperons (Fortin 2017 values for SFHoY)
    m_lambda: float = 1116.0      # Λ
    m_sigma_p: float = 1189.0     # Σ⁺
    m_sigma_0: float = 1193.0     # Σ⁰
    m_sigma_m: float = 1197.0     # Σ⁻
    m_xi_0: float = 1315.0        # Ξ⁰
    m_xi_m: float = 1321.0        # Ξ⁻
    
    # Delta resonances
    m_delta: float = 1232.0       # All Δ states (same mass approximation)
    
    # ---------------------------------------------------------
    # 2. Meson Masses (MeV) - mean-field mesons
    # ---------------------------------------------------------
    # Values from CompOSE table (fm^-1) converted to MeV using hc
    # m [MeV] = m [fm^-1] × hc
    m_sigma: float = 2.3689528914 * hc   # σ (scalar-isoscalar)  467.458 MeV
    m_omega: float = 3.9655047020 * hc   # ω (vector-isoscalar)  782.501 MeV
    m_rho: float = 3.8666788766 * hc     # ρ (vector-isovector)  763.000 MeV
    m_phi: float = 1020.0                # φ (vector-isoscalar, hidden strangeness)
    
    # ---------------------------------------------------------
    # 3. Pseudoscalar Meson Masses (MeV) - for thermal contributions
    # ---------------------------------------------------------
    m_pi_pm: float = 139.570   # π±
    m_pi_0: float = 134.977    # π⁰
    m_kaon_pm: float = 493.677 # K±
    m_kaon_0: float = 497.611  # K⁰, K̄⁰
    m_eta: float = 547.862     # η
    m_eta_prime: float = 957.78 # η'

    # ---------------------------------------------------------
    # 3. Nucleon Couplings (dimensionless g)
    # ---------------------------------------------------------
    g_sigma_N: float = 0.0
    g_omega_N: float = 0.0
    g_rho_N: float = 0.0
    g_phi_N: float = 0.0   # Usually 0 for nucleons

    # ---------------------------------------------------------
    # 4. Non-linear Scalar Potential Parameters
    # ---------------------------------------------------------
    # U(σ) = (g₂/3)σ³ + (g₃/4)σ⁴
    # dU/dσ = g₂σ² + g₃σ³
    g2: float = 0.0  # MeV (dimension of mass)
    g3: float = 0.0  # dimensionless
    
    # ---------------------------------------------------------
    # 5. Vector Self-Interaction Parameters
    # ---------------------------------------------------------
    # L = ... + (c₃/4)ω⁴ + (c₄/4)ρ⁴
    # Field eq: m²ω + c₃ω³ = ...
    c3: float = 0.0  # coefficient for ω⁴ term
    c4: float = 0.0  # coefficient for ρ⁴ term

    # ---------------------------------------------------------
    # 6. Symmetry Energy A-function Coefficients
    # ---------------------------------------------------------
    # A(σ,ω) = Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j)
    # Affects ρ-meson field equation and symmetry energy
    a_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(7))
    b_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ---------------------------------------------------------
    # 7. Coupling Ratios for Hyperons/Deltas
    # ---------------------------------------------------------
    # couplings_map[particle_name][meson] = absolute coupling value
    # Populated by factory functions
    couplings_map: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get_coupling(self, particle_name: str, meson: str) -> float:
        """
        Returns the meson-baryon coupling constant g_{MB}.
        
        Args:
            particle_name: Particle name (case-insensitive)
            meson: 'sigma', 'omega', 'rho', or 'phi'
            
        Returns:
            Coupling constant (dimensionless)
        """
        p_name = particle_name.lower()
        
        # Check explicit coupling map first
        if p_name in self.couplings_map:
            if meson in self.couplings_map[p_name]:
                return self.couplings_map[p_name][meson]
        
        # Handle nucleons
        if p_name in ['n', 'p', 'neutron', 'proton']:
            if meson == 'sigma':
                return self.g_sigma_N
            elif meson == 'omega':
                return self.g_omega_N
            elif meson == 'rho':
                return self.g_rho_N
            elif meson == 'phi':
                return self.g_phi_N
        
        return 0.0
    
    def compute_A(self, sigma: float, omega: float) -> float:
        """
        Compute the A-function for symmetry energy.
        
        A(σ,ω) = g_ρN² × f(σ,ω) = g_ρN² × [Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j)]
        
        Note: CompOSE coefficients define f (Steiner 2005 Eq. 15), 
        and A = g_ρ² × f per Fortin 2017.
        """
        f = 0.0
        for i in range(1, len(self.a_coeffs)):
            f += self.a_coeffs[i] * sigma**i
        for j in range(1, len(self.b_coeffs)):
            f += self.b_coeffs[j] * omega**(2*j)
        return self.g_rho_N**2 * f
    
    def compute_dA_dsigma(self, sigma: float) -> float:
        """Compute ∂A/∂σ = g_ρN² × ∂f/∂σ"""
        df = 0.0
        for i in range(1, len(self.a_coeffs)):
            df += i * self.a_coeffs[i] * sigma**(i-1)
        return self.g_rho_N**2 * df
    
    def compute_dA_domega(self, omega: float) -> float:
        """Compute ∂A/∂ω = g_ρN² × ∂f/∂ω"""
        df = 0.0
        for j in range(1, len(self.b_coeffs)):
            df += 2*j * self.b_coeffs[j] * omega**(2*j - 1)
        return self.g_rho_N**2 * df
    
    def get_baryon_mass(self, particle_name: str) -> float:
        """
        Get the baryon mass for RMF calculations.
        
        These masses are specific to the parametrization and may differ
        from PDG values. Use these for thermodynamic calculations.
        
        Args:
            particle_name: Particle name (case-insensitive)
            
        Returns:
            Mass in MeV
        """
        p_name = particle_name.lower()
        
        # Nucleons
        if p_name in ['p', 'proton']:
            return self.m_p
        elif p_name in ['n', 'neutron']:
            return self.m_n
        # Hyperons
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
        # Deltas
        elif p_name.startswith('delta'):
            return self.m_delta
        else:
            return 0.0
    
    def get_meson_mass(self, meson_name: str) -> float:
        """
        Get pseudoscalar meson mass for thermal calculations.
        
        Args:
            meson_name: Meson name (case-insensitive)
            
        Returns:
            Mass in MeV
        """
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
# BASE SFHo PARAMETERS (CompOSE table values)
# =============================================================================
def _get_base_sfho() -> SFHoParams:
    """
    Returns base SFHo nuclear parameters from CompOSE table.
    
    Reference: Steiner, Hempel, Fischer, ApJ 774 (2013) 17
    CompOSE table parameters for exact reproducibility.
    
    Nuclear matter properties at saturation:
    - n_sat = 0.1583 fm⁻³
    - E_0 = 16.19 MeV
    - K = 245.4 MeV
    - J = 31.57 MeV
    - L = 47.10 MeV
    - K_sym = -205.4 MeV
    """
    p = SFHoParams(name="SFHo")
    
    # Couplings from CompOSE table (c = g/m in fm)
    # g = c × m / hc
    c_sigma = 3.1791606374  # fm
    c_omega = 2.2752188529  # fm
    c_rho = 2.4062374629    # fm
    
    p.g_sigma_N = c_sigma / hc * p.m_sigma
    p.g_omega_N = c_omega / hc * p.m_omega
    p.g_rho_N = c_rho / hc * p.m_rho
    p.g_phi_N = 0.0  # Nucleons don't couple to φ

    # Non-linear σ potential parameters from CompOSE table
    # U = (b·M·g³_σ/3)σ³ + (c·g⁴_σ/4)σ⁴
    b_val = 7.3536466626e-3
    c_val = -3.8202821956e-3
    p.g2 = b_val * p.m_n * (p.g_sigma_N**3)
    p.g3 = c_val * (p.g_sigma_N**4)

    # Vector self-interaction parameters from CompOSE table
    # c3 = (ζ/6) × g_ωN⁴, c4 = (ξ/6) × g_ρN⁴
    zeta = -1.6155896062e-3
    xi = 4.1286242877e-3
    p.c3 = (zeta / 6.0) * (p.g_omega_N**4)
    p.c4 = (xi / 6.0) * (p.g_rho_N**4)


    # Symmetry energy A-function coefficients from CompOSE table
    # Per Steiner 2005 Eq. 13: Lagrangian has g_ρ² f(σ,ω) ρ²
    # Per Fortin 2017: A = g_ρ² × f, where f = Σᵢ aᵢσⁱ + Σⱼ bⱼω^(2j)
    # CompOSE gives the f coefficients (Steiner form), NOT A coefficients
    # The g_ρ² multiplication is done in compute_A()
    
    # a coefficients: a[i] has units fm^(i-1) in CompOSE table
    # Stored as: a_coeffs[i] = a_i × hc^(2-i) so that f has units MeV² with σ in MeV
    p.a_coeffs = np.zeros(7)
    p.a_coeffs[1] = -1.9308602647e-1 * hc            # a₁ [fm⁻¹] → MeV
    p.a_coeffs[2] = 5.6150318121e-1                   # a₂ [1] → dimensionless
    p.a_coeffs[3] = 2.8617603774e-1 / hc              # a₃ [fm] → MeV⁻¹
    p.a_coeffs[4] = 2.7717729776 / (hc**2)            # a₄ [fm²] → MeV⁻²
    p.a_coeffs[5] = 1.2307286924 / (hc**3)            # a₅ [fm³] → MeV⁻³
    p.a_coeffs[6] = 6.1480060734e-1 / (hc**4)         # a₆ [fm⁴] → MeV⁻⁴
    
    # b coefficients: b[j] has units fm^(2j-2) in CompOSE table
    # Stored as: b_coeffs[j] = b_j × hc^(2-2j) so that f has units MeV² with ω in MeV
    p.b_coeffs = np.zeros(4)
    p.b_coeffs[1] = 5.5118461115                      # b₁ [1] → dimensionless
    p.b_coeffs[2] = -1.8007283681 / (hc**2)           # b₂ [fm²] → MeV⁻²
    p.b_coeffs[3] = 4.2610479708e2 / (hc**4)          # b₃ [fm⁴] → MeV⁻⁴

    return p


# =============================================================================
# PARAMETRIZATION FACTORY FUNCTIONS
# =============================================================================

def get_sfho_nucleonic() -> SFHoParams:
    """
    SFHo with nucleons only (Steiner et al. 2013).
    
    Use for pure nuclear matter calculations.
    """
    p = _get_base_sfho()
    p.name = "SFHo_Nucleonic"
    return p


def get_sfhoy_fortin() -> SFHoParams:
    """
    SFHoY parametrization from Fortin et al. 2017 (PASA 35, e044).
    
    Features:
    - Scaled vector couplings with y=1.5 enhancement for hyperons
    - Scalar couplings computed from hyperon potential depths
    - Supports M_max ≈ 2.0 M_sun for cold NS
    
    Hyperon potential depths at saturation (SNM, n_sat = 0.158 fm⁻³):
    - U_Λ^(N) = -30 MeV
    - U_Σ^(N) = +30 MeV
    - U_Ξ^(N) = -14 MeV
    
    Vector coupling enhancement (y = 1.5):
    - R_ωΛ = 1.5 × (2/3) = 1.0
    - R_ωΣ = 1.5 × (2/3) = 1.0
    - R_ωΞ = 1.875 × (1/3) = 0.625
    """
    p = _get_base_sfho()
    p.name = "SFHoY_Fortin"
    
    # Lambda (uds)
    # U_Λ = -30 MeV → R_σ = 0.8542, R_ω = 1.0 (y=1.5)
    p.couplings_map['lambda'] = {
        'sigma': 0.854315 * p.g_sigma_N,
        'omega': 1.0 * p.g_omega_N,
        'phi': (-SQRT2/3.0) * 1.5 * p.g_omega_N,  # y=1.5
        'rho': 0.0,
    }
    
    # Sigma (uus, uds, dds)
    # U_Σ = +30 MeV → R_σ = 0.5861, R_ω = 1.0 (y=1.5)
    sigma_couplings = {
        'sigma': 0.586611 * p.g_sigma_N,
        'omega': 1.0 * p.g_omega_N,
        'phi': (-SQRT2/3.0) * 1.5 * p.g_omega_N,  # y=1.5
        'rho': 2.0 * p.g_rho_N,
    }
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = sigma_couplings.copy()
    
    # Xi (uss, dss)
    # U_Ξ = -14 MeV → R_σ = 0.5127, R_ω = 0.625 (y=1.875)
    xi_couplings = {
        'sigma': 0.512754 * p.g_sigma_N,
        'omega': 0.625 * p.g_omega_N,
        'phi': (-2.0*SQRT2/3.0) * 1.875 * p.g_omega_N,  # y=1.875
        'rho': 1.0 * p.g_rho_N,
    }
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = xi_couplings.copy()
    
    return p


def get_sfhoy_star_fortin() -> SFHoParams:
    """
    SFHoY* parametrization from Fortin et al. 2017.
    
    Hyperon potential depths at saturation (SNM, n_sat = 0.158 fm⁻³):
    - U_Λ^(N) = -30 MeV
    - U_Σ^(N) = +30 MeV
    - U_Ξ^(N) = -14 MeV
    
    Features:
    - SU(6) vector couplings: R_ωΛ = R_ωΣ = 2/3, R_ωΞ = 1/3
    - Scalar couplings from potential depths:
      - R_σΛ = 0.6142, R_σΣ = 0.3461, R_σΞ = 0.3026
    - Does NOT satisfy 2 M_sun constraint (M_max ≈ 1.75 M_sun)
    """
    p = _get_base_sfho()
    p.name = "SFHoY*_Fortin"
    
    # Lambda (uds) - SU(6) vectors
    # U_Λ = -30 MeV → R_σ = 0.6142
    p.couplings_map['lambda'] = {
        'sigma': 0.614161 * p.g_sigma_N,
        'omega': (2.0/3.0) * p.g_omega_N,
        'phi': (-SQRT2/3.0) * p.g_omega_N,
        'rho': 0.0,
    }
    
    # Sigma - SU(6) vectors
    # U_Σ = +30 MeV → R_σ = 0.3461
    sigma_couplings = {
        'sigma': 0.346456 * p.g_sigma_N,
        'omega': (2.0/3.0) * p.g_omega_N,
        'phi': (-SQRT2/3.0) * p.g_omega_N,
        'rho': 2.0 * p.g_rho_N,
    }
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = sigma_couplings.copy()
    
    # Xi - SU(6) vectors
    # U_Ξ = -14 MeV → R_σ = 0.3026
    xi_couplings = {
        'sigma': 0.302619 * p.g_sigma_N,
        'omega': (1.0/3.0) * p.g_omega_N,
        'phi': (-2.0*SQRT2/3.0) * p.g_omega_N,
        'rho': 1.0 * p.g_rho_N,
    }
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = xi_couplings.copy()
    
    return p


def get_sfho_2fam_phi() -> SFHoParams:
    """
    SFHoYD: SFHo with Hyperons and Deltas - includes phi meson coupling.
    
    Hyperon potential depths at saturation (SNM, n_sat = 0.158 fm⁻³):
    - U_Λ^(N) = -28 MeV (different from SFHoY* which uses -30)
    - U_Σ^(N) = +30 MeV
    - U_Ξ^(N) = -18 MeV (different from SFHoY* which uses -14)
    
    Features:
    - SU(6) vector couplings: R_ωΛ = R_ωΣ = 2/3, R_ωΞ = 1/3
    - Scalar couplings from potential depths:
      - R_σΛ = 0.6052, R_σΣ = 0.3461, R_σΞ = 0.3205
    - Delta couplings: R_σ = 1.15, R_ω = 1, R_ρ = 1, R_φ = 0
    - Hyperons couple to phi meson (hidden strangeness)
    
    This is the SFHoYD parametrization used in Mathematica.
    """
    p = _get_base_sfho()
    p.name = "SFHo_2fam_phi"
    
    # Lambda (uds) - SU(6) vectors
    # U_Λ = -28 MeV → R_σ = 0.6052
    p.couplings_map['lambda'] = {
        'sigma': 0.605237 * p.g_sigma_N,
        'omega': (2.0/3.0) * p.g_omega_N,
        'phi': (-SQRT2/3.0) * p.g_omega_N,
        'rho': 0.0,
    }
    
    # Sigma - SU(6) vectors
    # U_Σ = +30 MeV → R_σ = 0.3461
    sigma_couplings = {
        'sigma': 0.346456 * p.g_sigma_N,
        'omega': (2.0/3.0) * p.g_omega_N,
        'phi': (-SQRT2/3.0) * p.g_omega_N,
        'rho': 2.0 * p.g_rho_N,
    }
    for name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name] = sigma_couplings.copy()
    
    # Xi - SU(6) vectors
    # U_Ξ = -18 MeV → R_σ = 0.3205
    xi_couplings = {
        'sigma': 0.320466 * p.g_sigma_N,
        'omega': (1.0/3.0) * p.g_omega_N,
        'phi': (-2.0*SQRT2/3.0) * p.g_omega_N,
        'rho': 1.0 * p.g_rho_N,
    }
    for name in ['xi0', 'xi-']:
        p.couplings_map[name] = xi_couplings.copy()
    
    # Add Delta couplings
    # R_σ = 1.15, R_ω = 1.0, R_φ = 0, R_ρ = 1.0
    delta_couplings = {
        'sigma': 1.15 * p.g_sigma_N,
        'omega': 1.0 * p.g_omega_N,
        'phi': 0.0,
        'rho': 1.0 * p.g_rho_N,
    }
    for name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[name] = delta_couplings.copy()
    
    return p


def get_sfho_2fam() -> SFHoParams:
    """
    SFHo with Hyperons and Deltas - NO phi meson coupling (2-family without phi).
    
    Features:
    - SFHoY scalar couplings for hyperons
    - SFHoY vector couplings (omega, rho) for hyperons
    - g_phi = 0 for ALL hyperons (no hidden strangeness coupling)
    - Delta couplings: R_σ = 1.15, R_ω = 1, R_ρ = 1, R_φ = 0
    
    This is the same as 2fam_phi but with phi meson couplings set to zero
    for all strange baryons.
    """
    p = get_sfho_2fam_phi()
    p.name = "SFHo_2fam"
    
    # Set phi coupling to zero for all hyperons
    for particle in ['lambda', 'sigma+', 'sigma0', 'sigma-', 'xi0', 'xi-']:
        if particle in p.couplings_map:
            p.couplings_map[particle]['phi'] = 0.0
    
    return p




def get_sfho_general(
    x_sigma_lambda: float = 0.85,
    x_sigma_sigma: float = 0.58,
    x_sigma_xi: float = 0.51,
    x_sigma_delta: float = 1.15,
    x_omega_delta: float = 1.0,
    x_rho_delta: float = 1.0,
    use_scaled_vectors: bool = True,
    name: str = "SFHo_General"
) -> SFHoParams:
    """
    General SFHo parametrization with customizable couplings.
    
    Scalar couplings (x_sigma_*) are free parameters.
    Vector couplings for hyperons use either:
    - Scaled values (use_scaled_vectors=True, like SFHoY)
    - SU(6) values (use_scaled_vectors=False, like SFHoY*)
    
    Delta vector couplings are also customizable.
    
    Args:
        x_sigma_lambda: R_σΛ = g_σΛ/g_σN
        x_sigma_sigma: R_σΣ = g_σΣ/g_σN  
        x_sigma_xi: R_σΞ = g_σΞ/g_σN
        x_sigma_delta: R_σΔ = g_σΔ/g_σN
        x_omega_delta: R_ωΔ = g_ωΔ/g_ωN
        x_rho_delta: R_ρΔ = g_ρΔ/g_ρN
        use_scaled_vectors: If True, use SFHoY vectors; else SU(6)
        name: Identifier for the parametrization
        
    Returns:
        SFHoParams with specified couplings
    """
    p = _get_base_sfho()
    p.name = name
    
    if use_scaled_vectors:
        # Scaled vector ratios (SFHoY-like)
        vec_lambda = {'omega': 1.0, 'phi': -0.71, 'rho': 0.0}
        vec_sigma = {'omega': 1.0, 'phi': -0.71, 'rho': 2.0}
        vec_xi = {'omega': 0.62, 'phi': -1.77, 'rho': 1.0}
    else:
        # SU(6) vector ratios
        vec_lambda = SU6_RATIOS['lambda'].copy()
        vec_sigma = SU6_RATIOS['sigma'].copy()
        vec_xi = SU6_RATIOS['xi'].copy()
    
    # Lambda
    p.couplings_map['lambda'] = {
        'sigma': x_sigma_lambda * p.g_sigma_N,
        'omega': vec_lambda['omega'] * p.g_omega_N,
        'phi': vec_lambda['phi'] * p.g_omega_N,
        'rho': vec_lambda['rho'] * p.g_rho_N if vec_lambda['rho'] != 0 else 0.0,
    }
    
    # Sigma
    sigma_couplings = {
        'sigma': x_sigma_sigma * p.g_sigma_N,
        'omega': vec_sigma['omega'] * p.g_omega_N,
        'phi': vec_sigma['phi'] * p.g_omega_N,
        'rho': vec_sigma['rho'] * p.g_rho_N,
    }
    for name_s in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[name_s] = sigma_couplings.copy()
    
    # Xi
    xi_couplings = {
        'sigma': x_sigma_xi * p.g_sigma_N,
        'omega': vec_xi['omega'] * p.g_omega_N,
        'phi': vec_xi['phi'] * p.g_omega_N,
        'rho': vec_xi['rho'] * p.g_rho_N,
    }
    for name_x in ['xi0', 'xi-']:
        p.couplings_map[name_x] = xi_couplings.copy()
    
    # Delta
    delta_couplings = {
        'sigma': x_sigma_delta * p.g_sigma_N,
        'omega': x_omega_delta * p.g_omega_N,
        'phi': 0.0,  # No strangeness
        'rho': x_rho_delta * p.g_rho_N,
    }
    for name_d in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[name_d] = delta_couplings.copy()
    
    return p


def create_custom_parametrization(
    # Hyperon potential depths (MeV) at saturation
    U_Lambda_N: float = -30.0,
    U_Sigma_N: float = +30.0,
    U_Xi_N: float = -18.0,
    # Vector coupling enhancement factors (per hyperon family)
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
    
    The scalar coupling R_σH is computed from the target potential depth:
        U_H = -g_σH × σ + g_ωH × ω  at n_sat, Y_C = 0.5
        R_σH = (R_ωH × g_ωN × ω - U_H) / (g_σN × σ)
    
    Vector couplings follow SU(6) symmetry × y_H enhancement per family:
        g_ωΛ = g_ωN × (2/3) × y_Lambda
        g_ωΣ = g_ωN × (2/3) × y_Sigma  
        g_ωΞ = g_ωN × (1/3) × y_Xi
    
    Args:
        U_Lambda_N: Λ potential at n_sat in SNM (MeV), default -30
        U_Sigma_N: Σ potential at n_sat in SNM (MeV), default +30
        U_Xi_N: Ξ potential at n_sat in SNM (MeV), default -18
        y_Lambda, y_Sigma, y_Xi: Enhancement factors (1.0 = SU(6))
        x_sigma_delta, x_omega_delta, x_rho_delta: Delta coupling ratios
        name: Parametrization name
        
    Example:
        params = create_custom_parametrization(
            U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
            name="My_Custom"
        )
    """
    p = _get_base_sfho()
    p.name = name
    
    # Saturation fields for SFHo at n_sat = 0.158 fm^-3
    SIGMA_SAT = 29.697  # MeV
    OMEGA_SAT = 18.354  # MeV
    
    # SU(6) vector ratios with enhancement
    R_omega_Lambda = (2.0/3.0) * y_Lambda
    R_omega_Sigma = (2.0/3.0) * y_Sigma
    R_omega_Xi = (1.0/3.0) * y_Xi
    
    R_phi_Lambda = (-SQRT2/3.0) * y_Lambda
    R_phi_Sigma = (-SQRT2/3.0) * y_Sigma
    R_phi_Xi = (-2.0*SQRT2/3.0) * y_Xi
    
    # Compute R_σ from potential depth
    def compute_R_sigma(U_H: float, R_omega: float) -> float:
        return (R_omega * p.g_omega_N * OMEGA_SAT - U_H) / (p.g_sigma_N * SIGMA_SAT)
    
    R_sigma_Lambda = compute_R_sigma(U_Lambda_N, R_omega_Lambda)
    R_sigma_Sigma = compute_R_sigma(U_Sigma_N, R_omega_Sigma)
    R_sigma_Xi = compute_R_sigma(U_Xi_N, R_omega_Xi)
    
    # Lambda
    p.couplings_map['lambda'] = {
        'sigma': R_sigma_Lambda * p.g_sigma_N,
        'omega': R_omega_Lambda * p.g_omega_N,
        'phi': R_phi_Lambda * p.g_omega_N,
        'rho': 0.0,
    }
    
    # Sigma
    sigma_couplings = {
        'sigma': R_sigma_Sigma * p.g_sigma_N,
        'omega': R_omega_Sigma * p.g_omega_N,
        'phi': R_phi_Sigma * p.g_omega_N,
        'rho': 2.0 * p.g_rho_N,
    }
    for s_name in ['sigma+', 'sigma0', 'sigma-']:
        p.couplings_map[s_name] = sigma_couplings.copy()
    
    # Xi
    xi_couplings = {
        'sigma': R_sigma_Xi * p.g_sigma_N,
        'omega': R_omega_Xi * p.g_omega_N,
        'phi': R_phi_Xi * p.g_omega_N,
        'rho': 1.0 * p.g_rho_N,
    }
    for x_name in ['xi0', 'xi-']:
        p.couplings_map[x_name] = xi_couplings.copy()
    
    # Delta
    delta_couplings = {
        'sigma': x_sigma_delta * p.g_sigma_N,
        'omega': x_omega_delta * p.g_omega_N,
        'phi': 0.0,
        'rho': x_rho_delta * p.g_rho_N,
    }
    for d_name in ['delta++', 'delta+', 'delta0', 'delta-']:
        p.couplings_map[d_name] = delta_couplings.copy()
    
    return p


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_params_summary(params: SFHoParams) -> None:
    """Print a summary of the parametrization."""
    print(f"Parametrization: {params.name}")
    print("=" * 60)
    print("\nNucleon couplings:")
    print(f"  g_σN = {params.g_sigma_N:.5f}")
    print(f"  g_ωN = {params.g_omega_N:.5f}")
    print(f"  g_ρN = {params.g_rho_N:.5f}")
    print(f"  g_φN = {params.g_phi_N:.5f}")
    
    print("\nNon-linear parameters:")
    print(f"  g2 = {params.g2:.4e} MeV")
    print(f"  g3 = {params.g3:.4e}")
    print(f"  c3 = {params.c3:.4e}")
    print(f"  c4 = {params.c4:.4e}")
    
    if params.couplings_map:
        print("\nHyperon/Delta coupling ratios (R = g_MH / g_MN):")
        for particle, couplings in params.couplings_map.items():
            Rs = couplings['sigma'] / params.g_sigma_N
            Rw = couplings['omega'] / params.g_omega_N
            Rr = couplings['rho'] / params.g_rho_N if couplings['rho'] != 0 else 0
            Rp = couplings['phi'] / params.g_omega_N if couplings['phi'] != 0 else 0
            print(f"  {particle:10s}: R_σ={Rs:.3f}, R_ω={Rw:.3f}, R_ρ={Rr:.3f}, R_φ={Rp:.3f}")


def get_all_parametrizations() -> Dict[str, SFHoParams]:
    """Return dictionary of all available parametrizations."""
    return {
        'SFHo_Nucleonic': get_sfho_nucleonic(),
        'SFHoY_Fortin': get_sfhoy_fortin(),
        'SFHoY*_Fortin': get_sfhoy_star_fortin(),
        'SFHo_2fam_phi': get_sfho_2fam_phi(),
        'SFHo_2fam': get_sfho_2fam(),
    }


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("SFHo Parameters Module")
    print("=" * 70)
    
    # Test all parametrizations
    params_dict = get_all_parametrizations()
    
    for name, params in params_dict.items():
        print(f"\n{'='*70}")
        print_params_summary(params)
    
    # Test general parametrization
    print(f"\n{'='*70}")
    p_general = get_sfho_general(
        x_sigma_lambda=0.7,
        x_sigma_sigma=0.5,
        x_sigma_xi=0.4,
        x_sigma_delta=1.2,
        use_scaled_vectors=True,
        name="Custom_Test"
    )
    print_params_summary(p_general)
    
    # Test custom parametrization
    print("\n" + "=" * 70)
    print("Testing create_custom_parametrization:")
    print("-" * 50)
    p_custom = create_custom_parametrization(
        U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
        name="Custom_From_Potentials"
    )
    print_params_summary(p_custom)
    
    # Verify coupling retrieval
    print("\n" + "=" * 70)
    print("Testing coupling retrieval for SFHo_2fam_phi:")
    print("-" * 50)
    p = get_sfho_2fam_phi()
    test_particles = ['proton', 'neutron', 'lambda', 'sigma+', 'xi-', 'delta++']
    
    print(f"{'Particle':<12} {'σ':>10} {'ω':>10} {'ρ':>10} {'φ':>10}")
    print("-" * 54)
    for part in test_particles:
        gs = p.get_coupling(part, 'sigma')
        gw = p.get_coupling(part, 'omega')
        gr = p.get_coupling(part, 'rho')
        gp = p.get_coupling(part, 'phi')
        print(f"{part:<12} {gs:>10.4f} {gw:>10.4f} {gr:>10.4f} {gp:>10.4f}")