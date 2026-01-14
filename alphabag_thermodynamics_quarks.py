"""
alphabag_thermodynamics_quarks.py
=================================
Low-level thermodynamic functions for αBag quark matter.

Supports two phases:
- UNPAIRED: Normal quark matter with perturbative QCD α-corrections
- CFL: Color-Flavor Locked phase with pairing gap Δ

== UNPAIRED PHASE ==

The key equations for massless quarks with α-corrections (Fischer et al. 2011):

    P = [7/60 π² T⁴ (1 - 50α/(21π)) + (T²μ²/2 + μ⁴/(4π²))(1 - 2α/π)] / (ℏc)³
    ε = 3P  (ultra-relativistic limit)
    n = [μT² + μ³/π²] (1 - 2α/π) / (ℏc)³
    s = [7π²T³/15 (1 - 50α/(21π)) + Tμ² (1 - 2α/π)] / (ℏc)³

For massive quarks, we use:
    Thermo(m,α) = Thermo_Fermi(m,α=0) + [Thermo_massless(m=0,α) - Thermo_massless(m=0,α=0)]

== CFL PHASE ==

In CFL phase, quarks form Cooper pairs:
- Single chemical potential for all quarks (flavor-locking: μ_u = μ_d = μ_s = μ)
- Temperature-dependent gap: Δ(T) = Δ₀ √(1 - T²/T_c²)
- ABPR equations: P_ABPR = (3/π²) × [μ⁴/4 - m_s²μ²/4 + Δ²μ²/2 - 3Δ⁴/4] / (ℏc)³

Units:
- Energy/mass/chemical potentials: MeV
- Densities: fm⁻³
- Pressure/energy density: MeV/fm³

References:
- T. Fischer et al. Astrophys. J. Suppl. 194:39 (2011)
- M. Alford et al. Phys. Rev. D 71, 054009 (2005) [ABPR]
- M. Guerrini PhD Thesis (2026)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import brentq

from general_physics_constants import hc, hc3, PI, PI2
from alphabag_parameters import AlphaBagParams, get_alphabag_default
from general_fermi_integrals import solve_fermi_jel
import general_particles


# =============================================================================
# CONSTANTS
# =============================================================================
G_QUARK = general_particles.get_particle("quark").g_degen  # Degeneracy: spin(2) × color(3) = 6




# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class QuarkThermo:
    """Thermodynamic result for a single quark flavor."""
    n: float = 0.0      # Number density (fm⁻³)
    P: float = 0.0      # Pressure (MeV/fm³)
    e: float = 0.0      # Energy density (MeV/fm³)
    s: float = 0.0      # Entropy density (fm⁻³)
    f: float = 0.0      # Free energy density (MeV/fm³)


@dataclass
class AlphaBagThermo:
    """Full thermodynamic result for αBag quark matter (without leptons)."""
    # Inputs
    n_u: float = 0.0       # Up quark density (fm⁻³)
    n_d: float = 0.0       # Down quark density (fm⁻³)
    n_s: float = 0.0       # Strange quark density (fm⁻³)
    n_B: float = 0.0       # Baryon density (fm⁻³)
    n_C: float = 0.0       # Charge density (fm⁻³)
    n_S: float = 0.0       # Strangeness density (fm⁻³)
    T: float = 0.0         # Temperature (MeV)
    mu_u: float = 0.0      # Up quark chemical potential (MeV)
    mu_d: float = 0.0      # Down quark chemical potential (MeV)
    mu_s: float = 0.0      # Strange quark chemical potential (MeV)
    # Outputs
    P: float = 0.0         # Total pressure (MeV/fm³)
    e: float = 0.0         # Total energy density (MeV/fm³)
    s: float = 0.0         # Total entropy density (fm⁻³)
    f: float = 0.0         # Free energy density f = e - s*T (MeV/fm³)
    Y_C: float = 0.0       # Charge fraction
    Y_S: float = 0.0       # Strangeness fraction
    mu_B: float = 0.0      # Baryon chemical potential (MeV)
    mu_C: float = 0.0      # Charge chemical potential (MeV)
    mu_S: float = 0.0      # Strangeness chemical potential (MeV)


@dataclass
class CFLThermo:
    """CFL phase thermodynamic result."""
    # Inputs
    n_B: float = 0.0        # Baryon density (fm⁻³)
    T: float = 0.0          # Temperature (MeV)
    mu: float = 0.0         # Common quark chemical potential (MeV)
    Delta: float = 0.0      # Gap at this temperature (MeV)
    Delta0: float = 0.0     # Zero-temperature gap (MeV)
    # Outputs
    P: float = 0.0          # Pressure (MeV/fm³)
    e: float = 0.0          # Energy density (MeV/fm³)
    s: float = 0.0          # Entropy density (fm⁻³)
    f: float = 0.0          # Free energy density (MeV/fm³)
    # Quark fractions (all equal in CFL)
    n_u: float = 0.0        # Up quark density (fm⁻³)
    n_d: float = 0.0        # Down quark density (fm⁻³)
    n_s: float = 0.0        # Strange quark density (fm⁻³)
    Y_u: float = 1.0/3.0    # Up quark fraction per baryon
    Y_d: float = 1.0/3.0    # Down quark fraction per baryon
    Y_s: float = 1.0/3.0    # Strange quark fraction per baryon


# =============================================================================
# MASSLESS QUARK THERMODYNAMICS WITH α-CORRECTIONS
# =============================================================================
def P_massless_alpha(mu: float, T: float, alpha: float) -> float:
    """
    Pressure for massless quark at finite T with perturbative α correction.
    
    P = [7/60 π² T⁴ (1 - 50α/(21π)) + (T²μ²/2 + μ⁴/(4π²))(1 - 2α/π)] / (ℏc)³
    
    This is Eq. (A1) from Fischer et al. 2011 for single flavor with degeneracy g=6.
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        Pressure (MeV/fm³)
    """
    # Thermal correction factor
    alpha_T_factor = 1.0 - 50.0 * alpha / (21.0 * PI)
    # Density correction factor  
    alpha_n_factor = 1.0 - 2.0 * alpha / PI
    
    T4 = T**4
    T2_mu2 = T**2 * mu**2
    mu4 = mu**4
    
    # Pressure thermal term (with α correction)
    P_thermal = (7.0 / 60.0) * PI2 * T4 * alpha_T_factor
    
    # Fermi term (with α correction)
    P_fermi = (0.5 * T2_mu2 + mu4 / (4.0 * PI2)) * alpha_n_factor
    
    return (P_thermal + P_fermi) / hc3


def e_massless_alpha(mu: float, T: float, alpha: float) -> float:
    """
    Energy density for massless quark (ε = 3P in UR limit).
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        Energy density (MeV/fm³)
    """
    return 3.0 * P_massless_alpha(mu, T, alpha)


def n_massless_alpha(mu: float, T: float, alpha: float) -> float:
    """
    Number density for massless quark with α correction.
    
    n = [μT² + μ³/π²] (1 - 2α/π) / (ℏc)³
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        Number density (fm⁻³)
    """
    alpha_factor = 1.0 - 2.0 * alpha / PI
    
    n_value = (mu * T**2 + mu**3 / PI2) * alpha_factor
    
    return n_value / hc3


def s_massless_alpha(mu: float, T: float, alpha: float) -> float:
    """
    Entropy density for massless quark with α correction.
    
    s = [7π²T³/15 (1 - 50α/(21π)) + Tμ² (1 - 2α/π)] / (ℏc)³
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        Entropy density (fm⁻³)
    """
    alpha_T_factor = 1.0 - 50.0 * alpha / (21.0 * PI)
    alpha_n_factor = 1.0 - 2.0 * alpha / PI
    
    T3 = T**3
    T_mu2 = T * mu**2
    
    # Thermal entropy
    s_thermal = (7.0 / 15.0) * PI2 * T3 * alpha_T_factor
    
    # Fermi contribution to entropy
    s_fermi = T_mu2 * alpha_n_factor
    
    return (s_thermal + s_fermi) / hc3


# =============================================================================
# MASSIVE QUARK THERMODYNAMICS WITH α-CORRECTIONS
# =============================================================================
def _fermi_thermo_massive(mu: float, T: float, m: float) -> Tuple[float, float, float, float]:
    """
    Compute Fermi gas thermodynamics for massive quarks (no α correction).
    
    Uses the JEL approximation from general_fermi_integrals, which already
    handles T=0 (via _compute_exact_T0) and m=0 (via _compute_ur_limit) limits.
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Quark mass (MeV)
        
    Returns:
        (n, P, e, s) tuple
    """
    # solve_fermi_jel handles all limits internally:
    # - T < 1e-4: uses _compute_exact_T0
    # - m < 1e-5: uses _compute_ur_limit
    result = solve_fermi_jel(mu, max(T, 0.0), m, G_QUARK, include_antiparticles=True)
    return (result[0], result[1], result[2], result[3])

def n_massive_alpha(mu: float, T: float, m: float, alpha: float) -> float:
    """
    Number density for massive quark with α correction.
    
    Strategy: n(m,α) = n_Fermi(m,0) + [n_massless(α) - n_massless(0)]
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Quark mass (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        Number density (fm⁻³)
    """
    n_fermi, _, _, _ = _fermi_thermo_massive(mu, T, m)
    n_correction = n_massless_alpha(mu, T, alpha) - n_massless_alpha(mu, T, 0.0)
    return n_fermi + n_correction


def n_quark_alpha(mu: float, T: float, m: float, alpha: float) -> float:
    """
    Number density for quark with α correction, choosing massless/massive based on mass.
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Quark mass (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        Number density (fm⁻³)
    """
    if m < 1e-5:
        return n_massless_alpha(mu, T, alpha)
    return n_massive_alpha(mu, T, m, alpha)


# =============================================================================
# INDIVIDUAL QUARK THERMODYNAMICS
# =============================================================================
def compute_massless_quark_thermo(mu: float, T: float, alpha: float) -> QuarkThermo:
    """
    Thermodynamics for a massless quark (u or d) with α correction.
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        QuarkThermo with n, P, e, s, f
    """
    n = n_massless_alpha(mu, T, alpha)
    P = P_massless_alpha(mu, T, alpha)
    e = e_massless_alpha(mu, T, alpha)
    s = s_massless_alpha(mu, T, alpha)
    f = e - T * s
    return QuarkThermo(n=n, P=P, e=e, s=s, f=f)


def compute_massive_quark_thermo(mu: float, T: float, m: float, alpha: float) -> QuarkThermo:
    """
    Thermodynamics for a massive quark (s) with α correction.
    
    Computes all quantities in a single call for efficiency.
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Quark mass (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        QuarkThermo with n, P, e, s, f
    """
    # Get base Fermi gas values (single call)
    n_fermi, P_fermi, e_fermi, s_fermi = _fermi_thermo_massive(mu, T, m)
    
    # α-corrections (difference between massless with α and without)
    P_corr = P_massless_alpha(mu, T, alpha) - P_massless_alpha(mu, T, 0.0)
    e_corr = e_massless_alpha(mu, T, alpha) - e_massless_alpha(mu, T, 0.0)
    n_corr = n_massless_alpha(mu, T, alpha) - n_massless_alpha(mu, T, 0.0)
    s_corr = s_massless_alpha(mu, T, alpha) - s_massless_alpha(mu, T, 0.0)
    
    n = n_fermi + n_corr
    P = P_fermi + P_corr
    e = e_fermi + e_corr
    s = s_fermi + s_corr
    f = e - T * s
    
    return QuarkThermo(n=n, P=P, e=e, s=s, f=f)


def compute_quark_thermo(mu: float, T: float, m: float, alpha: float) -> QuarkThermo:
    """
    Unified thermodynamics for any quark flavor.
    
    Chooses massless or massive treatment based on mass value (m < 1e-5 → massless).
    
    Args:
        mu: Chemical potential (MeV)
        T: Temperature (MeV)
        m: Quark mass (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        QuarkThermo with n, P, e, s, f
    """
    if m < 1e-5:
        return compute_massless_quark_thermo(mu, T, alpha)
    return compute_massive_quark_thermo(mu, T, m, alpha)


# Convenience aliases for backward compatibility
def compute_u_thermo(mu_u: float, T: float, m_u: float, alpha: float) -> QuarkThermo:
    """Thermodynamics for up quark."""
    return compute_quark_thermo(mu_u, T, m_u, alpha)


def compute_d_thermo(mu_d: float, T: float, m_d: float, alpha: float) -> QuarkThermo:
    """Thermodynamics for down quark."""
    return compute_quark_thermo(mu_d, T, m_d, alpha)


def compute_s_thermo(mu_s: float, T: float, m_s: float, alpha: float) -> QuarkThermo:
    """Thermodynamics for strange quark."""
    return compute_quark_thermo(mu_s, T, m_s, alpha)



# =============================================================================
# BAG CONSTANT CONTRIBUTIONS
# =============================================================================
def compute_bag_pressure(params: AlphaBagParams) -> float:
    """
    Compute pressure contribution from bag constant.
    
    P_B = -B / (ℏc)³
    
    Returns:
        P_B: Bag pressure contribution (MeV/fm³), negative
    """
    return -params.B / hc3


def compute_bag_energy(params: AlphaBagParams) -> float:
    """
    Compute energy density contribution from bag constant.
    
    e_B = +B / (ℏc)³
    
    Returns:
        e_B: Bag energy density contribution (MeV/fm³), positive
    """
    return params.B / hc3


# =============================================================================
# GLUON THERMODYNAMICS (with α-corrections)
# =============================================================================
def gluon_thermo(T: float, alpha: float) -> QuarkThermo:
    """
    Thermodynamic quantities for thermal gluons with α correction.
    
    P_g = 8 π²/45 T⁴ (1 - 15α/(4π)) / (ℏc)³
    e_g = 3 P_g
    s_g = 32 π²/45 T³ (1 - 15α/(4π)) / (ℏc)³
    
    Args:
        T: Temperature (MeV)
        alpha: QCD coupling constant α_s
        
    Returns:
        QuarkThermo with n=0, P, e, s, f
    """
    if T <= 0:
        return QuarkThermo(n=0.0, P=0.0, e=0.0, s=0.0, f=0.0)
    
    alpha_factor = 1.0 - 15.0 * alpha / (4.0 * PI)
    
    T3 = T**3
    T4 = T * T3
    
    P = 8.0 * PI2 / 45.0 * T4 * alpha_factor / hc3
    e = 3.0 * P
    s = 32.0 * PI2 / 45.0 * T3 * alpha_factor / hc3
    f = e - T * s
    
    return QuarkThermo(n=0.0, P=P, e=e, s=s, f=f)


# =============================================================================
# FULL QUARK MATTER THERMODYNAMICS (without leptons)
# =============================================================================
def compute_alphabag_thermo_from_mu(
    mu_u: float, mu_d: float, mu_s: float, T: float,
    params: AlphaBagParams = None
) -> AlphaBagThermo:
    """
    Compute full αBag thermodynamics from chemical potentials.
    
    Total pressure includes quark contributions and bag subtraction.
    Gluon contributions are NOT included here (add separately if needed).
    
    Args:
        mu_u, mu_d, mu_s: Quark chemical potentials (MeV)
        T: Temperature (MeV)
        params: AlphaBagParams (uses default if None)
        
    Returns:
        AlphaBagThermo with all thermodynamic quantities
    """
    if params is None:
        params = get_alphabag_default()
    
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Compute individual quark contributions
    thermo_u = compute_u_thermo(mu_u, T, m_u, alpha)
    thermo_d = compute_d_thermo(mu_d, T, m_d, alpha)
    thermo_s = compute_s_thermo(mu_s, T, m_s, alpha)
    
    # Number densities
    n_u = thermo_u.n
    n_d = thermo_d.n
    n_s = thermo_s.n
    
    # Total kinetic contributions
    P_kin = thermo_u.P + thermo_d.P + thermo_s.P
    e_kin = thermo_u.e + thermo_d.e + thermo_s.e
    s_kin = thermo_u.s + thermo_d.s + thermo_s.s
    
    # Bag contributions
    P_B = compute_bag_pressure(params)
    e_B = compute_bag_energy(params)
    
    # Total thermodynamics
    P_total = P_kin + P_B
    e_total = e_kin + e_B
    s_total = s_kin
    f_total = e_total - s_total * T
    
    # Conserved quantities
    n_B = (n_u + n_d + n_s) / 3.0
    n_C = (2.0/3.0)*n_u - (1.0/3.0)*n_d - (1.0/3.0)*n_s
    n_S = n_s
    
    Y_C = n_C / n_B 
    Y_S = n_S / n_B 
    
    # Conserved charge chemical potentials
    mu_B = mu_u + 2*mu_d
    mu_C = mu_u - mu_d
    mu_S = mu_s - mu_d
    
    return AlphaBagThermo(
        n_u=n_u, n_d=n_d, n_s=n_s, n_B=n_B, n_C=n_C, n_S=n_S,
        Y_C=Y_C, Y_S=Y_S,
        T=T,
        mu_u=mu_u, mu_d=mu_d, mu_s=mu_s, mu_B=mu_B, mu_C=mu_C, mu_S=mu_S,
        P=P_total, e=e_total, s=s_total, f=f_total
    )


# =============================================================================
# CFL PHASE: TEMPERATURE-DEPENDENT GAP
# =============================================================================
def T_critical(Delta0: float) -> float:
    """
    Critical temperature for CFL phase transition.
    
    T_c ≈ 0.57 × 2^(1/3) × Δ₀
    """
    # CFL critical temperature coefficient: T_c = TC_COEFF * Delta0
    TC_COEFF = 0.57 * 2**(1.0/3.0)  # ≈ 0.718
    return TC_COEFF * Delta0


def gap_cfl(T: float, Delta0: float) -> float:
    """
    Temperature-dependent CFL pairing gap (BCS-like).
    
    Δ(T) = Δ₀ √(1 - T²/T_c²)  for T < T_c
    Δ(T) = 0                   for T ≥ T_c
    """
    if Delta0 <= 0 or T < 0:
        return 0.0
    T_c = T_critical(Delta0)
    if T >= T_c:
        return 0.0
    return Delta0 * np.sqrt(1.0 - (T/T_c)**2)


def dgap_dT_cfl(T: float, Delta0: float) -> float:
    """
    Derivative dΔ/dT of the temperature-dependent gap.
    
    For T < T_c:
        dΔ/dT = -T/(T_c² √(1 - T²/T_c²)) × Δ₀
              = -1.9389... × T / (Δ₀ √(1 - 1.9389...×T²/Δ₀²))
    """
    if Delta0 <= 0 or T <= 0:
        return 0.0
    T_c = T_critical(Delta0)
    if T >= T_c:
        return 0.0
    sqrt_term = np.sqrt(1.0 - (T/T_c)**2)
    if sqrt_term < 1e-10:
        return 0.0  # At T_c, return 0 (avoid divergence)
    # dΔ/dT = -Δ₀ × T / (T_c² × sqrt_term)
    return -Delta0 * T / (T_c**2 * sqrt_term)


# =============================================================================
# CFL PHASE: THERMODYNAMICS (Δ² corrections to unpaired)
# =============================================================================
def P_cfl_correction(mu_u: float, mu_d: float, mu_s: float, 
                     T: float, Delta0: float) -> float:
    """
    CFL pressure correction: (μu² + μd² + μs²) × Δ(T)² / (π² × hc³)
    """
    Delta = gap_cfl(T, Delta0)
    mu_sum_sq = mu_u**2 + mu_d**2 + mu_s**2
    return mu_sum_sq * Delta**2 / (PI2 * hc3)


def n_cfl_correction(mu: float, T: float, Delta0: float) -> float:
    """
    CFL density correction for single quark: 2μ × Δ(T)² / (π² × hc³)
    """
    Delta = gap_cfl(T, Delta0)
    return 2.0 * mu * Delta**2 / (PI2 * hc3)


def s_cfl_correction(mu_u: float, mu_d: float, mu_s: float,
                     T: float, Delta0: float) -> float:
    """
    CFL entropy correction: 2(μu² + μd² + μs²) × Δ(T) × dΔ/dT / (π² × hc³)
    """
    Delta = gap_cfl(T, Delta0)
    dDelta_dT = dgap_dT_cfl(T, Delta0)
    mu_sum_sq = mu_u**2 + mu_d**2 + mu_s**2
    return 2.0 * mu_sum_sq * Delta * dDelta_dT / (PI2 * hc3)


def compute_cfl_thermo_from_mu(
    mu_u: float, mu_d: float, mu_s: float, T: float, Delta0: float,
    params: AlphaBagParams = None
) -> CFLThermo:
    """
    Compute CFL phase thermodynamics from chemical potentials.
    
    CFL adds Δ² corrections to the unpaired quark thermodynamics:
        P_CFL = P_unpaired + (μu² + μd² + μs²) × Δ(T)² / π² - B
        n_q_CFL = n_q_unpaired + 2μq × Δ(T)² / π²
        s_CFL = s_unpaired + 2(μu² + μd² + μs²) × Δ(T) × dΔ/dT / π²
        f_CFL = -P + μu×nu + μd×nd + μs×ns
        e_CFL = f + T×s
    
    Args:
        mu_u, mu_d, mu_s: Quark chemical potentials (MeV)
        T: Temperature (MeV)
        Delta0: Zero-temperature pairing gap (MeV)
        params: AlphaBagParams (uses default if None)
        
    Returns:
        CFLThermo with all thermodynamic quantities
    """
    if params is None:
        params = get_alphabag_default()
    
    alpha = params.alpha
    m_u, m_d, m_s = params.m_u, params.m_d, params.m_s
    
    # Gap at this temperature
    Delta = gap_cfl(T, Delta0)
    
    # Unpaired quark thermodynamics
    thermo_u = compute_quark_thermo(mu_u, T, m_u, alpha)
    thermo_d = compute_quark_thermo(mu_d, T, m_d, alpha)
    thermo_s = compute_quark_thermo(mu_s, T, m_s, alpha)
    
    # Unpaired totals
    P_unpaired = thermo_u.P + thermo_d.P + thermo_s.P
    n_u_unpaired = thermo_u.n
    n_d_unpaired = thermo_d.n
    n_s_unpaired = thermo_s.n
    s_unpaired = thermo_u.s + thermo_d.s + thermo_s.s
    
    # CFL corrections
    P_corr = P_cfl_correction(mu_u, mu_d, mu_s, T, Delta0)
    n_u_corr = n_cfl_correction(mu_u, T, Delta0)
    n_d_corr = n_cfl_correction(mu_d, T, Delta0)
    n_s_corr = n_cfl_correction(mu_s, T, Delta0)
    s_corr = s_cfl_correction(mu_u, mu_d, mu_s, T, Delta0)
    
    # CFL totals (quark + corrections - bag)
    B = params.B / hc3  # Bag constant in MeV/fm³
    P = P_unpaired + P_corr - B
    n_u = n_u_unpaired + n_u_corr
    n_d = n_d_unpaired + n_d_corr
    n_s = n_s_unpaired + n_s_corr
    s = s_unpaired + s_corr
    
    # Free energy: f = -P + μu×nu + μd×nd + μs×ns
    f = -P + mu_u * n_u + mu_d * n_d + mu_s * n_s
    
    # Energy: e = f + T×s
    e = f + T * s
    
    # Baryon density
    n_B = (n_u + n_d + n_s) / 3.0
    
    # Fractions
    Y_u = n_u / n_B if n_B > 0 else 1.0/3.0
    Y_d = n_d / n_B if n_B > 0 else 1.0/3.0
    Y_s = n_s / n_B if n_B > 0 else 1.0/3.0
    
    # Common quark chemical potential (average)
    mu = (mu_u + mu_d + mu_s) / 3.0
    
    return CFLThermo(
        n_B=n_B, T=T, mu=mu, Delta=Delta, Delta0=Delta0,
        P=P, e=e, s=s, f=f,
        n_u=n_u, n_d=n_d, n_s=n_s,
        Y_u=Y_u, Y_d=Y_d, Y_s=Y_s
    )


def B_eff_cfl(mu_u: float, mu_d: float, mu_s: float, T: float, Delta0: float,
              params: AlphaBagParams = None) -> float:
    """
    Effective bag constant in CFL phase.
    
    B_eff = B/hc³ - (μu² + μd² + μs²) × Δ(T)² / (π² × hc³)
    """
    if params is None:
        params = get_alphabag_default()
    B = params.B / hc3
    Delta = gap_cfl(T, Delta0)
    mu_sum_sq = mu_u**2 + mu_d**2 + mu_s**2
    return B - mu_sum_sq * Delta**2 / (PI2 * hc3)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("αBag Thermodynamics Test")
    print("=" * 50)
    
    params = get_alphabag_default()
    print(f"Parameters: B^1/4={params.B4} MeV, α_s={params.alpha}, m_s={params.m_s} MeV")
    
    # Test unpaired phase from μ
    print("\n1. Unpaired phase test:")
    print("-" * 40)
    mu_u, mu_d, mu_s = 300.0, 300.0, 350.0
    T = 50.0
    
    result = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, T, params)
    print(f"   μ = ({mu_u}, {mu_d}, {mu_s}) MeV, T = {T} MeV")
    print(f"   n_B = {result.n_B:.4f} fm⁻³")
    print(f"   P   = {result.P:.2f} MeV/fm³")
    print(f"   Y_C = {result.Y_C:.4f}, Y_S = {result.Y_S:.4f}")
    
    # Test massless α-corrections
    print("\n2. Massless α-correction test:")
    print("-" * 40)
    n_alpha0 = n_massless_alpha(300.0, 50.0, 0.0)
    n_alpha03 = n_massless_alpha(300.0, 50.0, 0.3)
    print(f"   n(α=0)   = {n_alpha0:.6f} fm⁻³")
    print(f"   n(α=0.3) = {n_alpha03:.6f} fm⁻³")
    print(f"   Ratio    = {n_alpha03/n_alpha0:.4f} (expected ~{1 - 2*0.3/PI:.4f})")
    
    # Test gluon
    print("\n3. Gluon thermodynamics:")
    print("-" * 40)
    gluon = gluon_thermo(50.0, 0.3)
    print(f"   T = 50 MeV, α = 0.3")
    print(f"   P_g = {gluon.P:.4f} MeV/fm³")
    print(f"   e_g = {gluon.e:.4f} MeV/fm³")
    
    # Test CFL phase
    print("\n4. CFL phase test:")
    print("-" * 40)
    Delta0 = 100.0
    mu_u, mu_d, mu_s = 300.0, 300.0, 300.0
    print(f"   Δ₀ = {Delta0} MeV, T_c = {T_critical(Delta0):.2f} MeV")
    for T in [0, 30, 60]:
        cfl = compute_cfl_thermo_from_mu(mu_u, mu_d, mu_s, T, Delta0, params)
        print(f"   T = {T:3.0f} MeV: Δ = {cfl.Delta:6.2f} MeV, n_B = {cfl.n_B:.4f} fm⁻³, P = {cfl.P:.2f} MeV/fm³")
    
    print("\n✓ All OK")

