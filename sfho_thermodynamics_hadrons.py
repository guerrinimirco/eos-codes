"""
sfho_thermodynamics_hadrons.py
================================
Model-dependent hadron thermodynamics for SFHo-type RMF models.

This module computes thermodynamic quantities for:
- Baryons (nucleons, hyperons, deltas) in mean-field approximation
- Pseudoscalar mesons (pions, kaons, etas) as free Bose gas

Units:
- Energies/masses: MeV
- Lengths: fm
- Number density: fm⁻³
- Pressure/energy density: MeV/fm³
- Entropy density: fm⁻³
- Meson fields: MeV

References:
- Fortin, Oertel, Providência, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from general_physics_constants import hc, hc3
from general_particles import Particle
from general_fermi_integrals import solve_fermi_jel
from general_bose_integrals import solve_bose_jel
from sfho_parameters import SFHoParams


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HadronState:
    """
    Thermodynamic state for a single hadron species.
    
    Attributes:
        n: Number density (fm⁻³) - net baryon number
        ns: Scalar density (fm⁻³) - for σ field equation
        P: Pressure contribution (MeV/fm³)
        e: Energy density contribution (MeV/fm³)
        s: Entropy density contribution (fm⁻³)
        mu_eff: Effective chemical potential (MeV)
        m_eff: Effective mass (MeV)
    """
    n: float      # Number density
    ns: float     # Scalar density
    P: float      # Pressure
    e: float      # Energy density
    s: float      # Entropy density
    mu_eff: float # Effective chemical potential
    m_eff: float  # Effective mass
    
    def __repr__(self):
        return (f"HadronState(n={self.n:.4e}, ns={self.ns:.4e}, "
                f"P={self.P:.4e}, e={self.e:.4e})")


@dataclass
class HadronThermoResult:
    """
    Complete thermodynamic result for all hadrons.
    
    Attributes:
        states: Dictionary of individual hadron states
        n_B: Total baryon number density (fm⁻³)
        n_Q: Total charge density (fm⁻³)
        n_S: Total strangeness density (fm⁻³)
        P_hadrons: Total hadron pressure (MeV/fm³)
        e_hadrons: Total hadron energy density (MeV/fm³)
        s_hadrons: Total hadron entropy density (fm⁻³)
        src_sigma: Source term for σ equation (fm⁻³)
        src_omega: Source term for ω equation (fm⁻³)
        src_rho: Source term for ρ equation (fm⁻³)
        src_phi: Source term for φ equation (fm⁻³)
    """
    states: Dict[str, HadronState]
    n_B: float       # Total baryon density
    n_Q: float       # Total charge density
    n_S: float       # Total strangeness density
    P_hadrons: float # Hadron pressure
    e_hadrons: float # Hadron energy density
    s_hadrons: float # Hadron entropy density
    src_sigma: float # σ source
    src_omega: float # ω source
    src_rho: float   # ρ source
    src_phi: float   # φ source


@dataclass
class MesonThermoResult:
    """
    Thermodynamic result for pseudoscalar mesons (π, K, η).
    
    Mesons are treated as free Bose gas with chemical potentials:
    - π⁺: μ = +μ_Q
    - π⁻: μ = -μ_Q
    - π⁰: μ = 0
    - K⁺: μ = +μ_Q - μ_S
    - K⁰: μ = -μ_S
    - K⁻: μ = -μ_Q + μ_S
    - K̄⁰: μ = +μ_S
    - η, η': μ = 0
    
    Attributes:
        n_Q_mesons: Total meson charge density (fm⁻³)
        n_S_mesons: Total meson strangeness density (fm⁻³)
        P_mesons: Total meson pressure (MeV/fm³)
        e_mesons: Total meson energy density (MeV/fm³)
        s_mesons: Total meson entropy density (fm⁻³)
        densities: Dictionary of individual meson densities
    """
    n_Q_mesons: float  # Meson charge density
    n_S_mesons: float  # Meson strangeness density
    P_mesons: float    # Meson pressure
    e_mesons: float    # Meson energy density
    s_mesons: float    # Meson entropy density
    densities: Dict[str, float]  # Individual meson densities


# =============================================================================
# MAIN THERMODYNAMICS FUNCTIONS
# =============================================================================

def compute_hadron_thermo(
    T: float,
    mu_B: float, mu_Q: float, mu_S: float,
    sigma: float, omega: float, rho: float, phi: float,
    particles: List[Particle],
    params: SFHoParams
) -> HadronThermoResult:
    """
    Compute thermodynamic quantities for all hadron species.
    
    Given temperature, chemical potentials, and meson fields, this function:
    1. Computes effective masses: M*_j = m_j - g_σj × σ
    2. Computes effective chemical potentials: 
       μ*_j = B_j×μ_B + Q_j×μ_Q + S_j×μ_S - g_ωj×ω - g_ρj×I₃j×ρ - g_φj×φ
    3. Evaluates Fermi integrals for (n, P, e, s, n_s)
    4. Computes source terms for field equations
    
    Args:
        T: Temperature (MeV)
        mu_B: Baryon chemical potential (MeV)
        mu_Q: Charge chemical potential (MeV)
        mu_S: Strangeness chemical potential (MeV)
        sigma: σ-meson field (MeV)
        omega: ω-meson field (MeV)
        rho: ρ-meson field (MeV)
        phi: φ-meson field (MeV)
        particles: List of Particle objects to include
        params: SFHoParams with model parameters
        
    Returns:
        HadronThermoResult with all thermodynamic quantities
    """
    states = {}
    
    # Initialize totals
    n_B_tot = 0.0
    n_Q_tot = 0.0
    n_S_tot = 0.0
    P_tot = 0.0
    e_tot = 0.0
    s_tot = 0.0
    
    # Initialize source terms
    src_sigma = 0.0
    src_omega = 0.0
    src_rho = 0.0
    src_phi = 0.0
    
    for p in particles:
        # 1. Get meson-baryon couplings
        g_s = params.get_coupling(p.name, 'sigma')
        g_w = params.get_coupling(p.name, 'omega')
        g_r = params.get_coupling(p.name, 'rho')
        g_p = params.get_coupling(p.name, 'phi')
        
        # 2. Get baryon mass from parametrization (not from Particle object)
        # This allows for different mass values in different parametrizations
        m_baryon = params.get_baryon_mass(p.name)
        if m_baryon == 0.0:
            # Fall back to Particle mass if not in parametrization
            m_baryon = p.mass
        
        # 3. Effective mass: M* = m - g_σ × σ
        m_eff = m_baryon - g_s * sigma
        
        # Ensure positive effective mass (can become negative at high density)
        if m_eff < 0:
            m_eff = 1e-3  # Small positive value
        
        # 4. Effective chemical potential
        # μ_j = B_j×μ_B + Q_j×μ_Q + S_j×μ_S
        mu_nonint = p.baryon_no * mu_B + p.charge * mu_Q + p.strangeness * mu_S
        
        # Vector field shifts
        # μ* = μ - g_ω×ω - g_ρ×I₃×ρ - g_φ×φ
        vector_shift = g_w * omega + g_r * p.isospin_3 * rho + g_p * phi
        mu_eff = mu_nonint - vector_shift
        
        # 5. Compute Fermi integrals
        # solve_fermi_jel returns (n, P, e, s, ns)
        n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m_eff, p.g_degen,
                                          include_antiparticles=True)
        
        # Store individual state
        states[p.name] = HadronState(
            n=n, ns=ns, P=P, e=e, s=s, mu_eff=mu_eff, m_eff=m_eff
        )
        
        # 5. Accumulate totals
        # n is the NET number density (particles - antiparticles)
        n_B_tot += p.baryon_no * n
        n_Q_tot += p.charge * n
        n_S_tot += p.strangeness * n
        P_tot += P
        e_tot += e
        s_tot += s
        
        # 6. Source terms for field equations
        # σ couples to scalar density
        src_sigma += g_s * ns
        # ω, φ couple to number density
        src_omega += g_w * n
        src_phi += g_p * n
        # ρ couples to isospin-weighted density
        src_rho += g_r * p.isospin_3 * n
    
    return HadronThermoResult(
        states=states,
        n_B=n_B_tot,
        n_Q=n_Q_tot,
        n_S=n_S_tot,
        P_hadrons=P_tot,
        e_hadrons=e_tot,
        s_hadrons=s_tot,
        src_sigma=src_sigma,
        src_omega=src_omega,
        src_rho=src_rho,
        src_phi=src_phi
    )


def compute_field_residuals(
    sigma: float, omega: float, rho: float, phi: float,
    src_sigma: float, src_omega: float, src_rho: float, src_phi: float,
    params: SFHoParams
) -> Tuple[float, float, float, float]:
    """
    Compute residuals of the meson field equations.
    
    The field equations are (in mean-field approximation):
    
    σ: m²_σ σ + g₂σ² + g₃σ³ - ∂A/∂σ ρ² = Σⱼ g_σⱼ n^s_j × (ℏc)³
    ω: m²_ω ω + c₃ω³ + ∂A/∂ω ρ² = Σⱼ g_ωⱼ nⱼ × (ℏc)³
    ρ: m²_ρ ρ + c₄ρ³ + 2Aρ = Σⱼ g_ρⱼ I₃ⱼ nⱼ × (ℏc)³
    φ: m²_φ φ = Σⱼ g_φⱼ nⱼ × (ℏc)³
    
    Residual = LHS - RHS (should be zero at solution)
    
    Args:
        sigma, omega, rho, phi: Meson fields (MeV)
        src_sigma, src_omega, src_rho, src_phi: Source terms (fm⁻³)
        params: Model parameters
        
    Returns:
        Tuple of (res_sigma, res_omega, res_rho, res_phi) in MeV³
    """
    # Convert sources from fm⁻³ to MeV³
    # Source × (ℏc)³ gives MeV³
    
    # σ equation
    # m²σ + g₂σ² + g₃σ³ - (∂A/∂σ)ρ² = g_σ n_s × hc³
    dU_dsigma = params.g2 * sigma**2 + params.g3 * sigma**3
    dA_dsigma = params.compute_dA_dsigma(sigma)
    
    lhs_sigma = params.m_sigma**2 * sigma + dU_dsigma - dA_dsigma * rho**2
    rhs_sigma = src_sigma * hc3
    res_sigma = lhs_sigma - rhs_sigma
    
    # ω equation
    # m²ω + c₃ω³ + (∂A/∂ω)ρ² = g_ω n × hc³
    dA_domega = params.compute_dA_domega(omega)
    
    lhs_omega = params.m_omega**2 * omega + params.c3 * omega**3 + dA_domega * rho**2
    rhs_omega = src_omega * hc3
    res_omega = lhs_omega - rhs_omega
    
    # ρ equation
    # m²ρ + c₄ρ³ + 2Aρ = g_ρ I₃ n × hc³
    A = params.compute_A(sigma, omega)
    
    lhs_rho = params.m_rho**2 * rho + params.c4 * rho**3 + 2.0 * A * rho
    rhs_rho = src_rho * hc3
    res_rho = lhs_rho - rhs_rho
    
    # φ equation (linear, no self-interactions)
    # m²φ = g_φ n × hc³
    lhs_phi = params.m_phi**2 * phi
    rhs_phi = src_phi * hc3
    res_phi = lhs_phi - rhs_phi
    
    return res_sigma, res_omega, res_rho, res_phi


def compute_meson_contribution(
    sigma: float, omega: float, rho: float, phi: float,
    params: SFHoParams
) -> Tuple[float, float]:
    """
    Compute meson field contributions to pressure and energy density.
    
    The meson Lagrangian contributes to the thermodynamics:
    
    P_meson = -V(σ) + ½m²_ω ω² + (c₃/4)ω⁴ 
              + ½m²_ρ ρ² + (c₄/4)ρ⁴ + Aρ²
              + ½m²_φ φ²
              
    e_meson = +V(σ) + ½m²_ω ω² + (3c₃/4)ω⁴
              + ½m²_ρ ρ² + (3c₄/4)ρ⁴ + Aρ²
              + ½m²_φ φ²
    
    where V(σ) = ½m²_σ σ² + (g₂/3)σ³ + (g₃/4)σ⁴
    
    Note: The sign conventions follow from the mean-field Lagrangian.
    The attractive σ field contributes negatively to pressure.
    
    Args:
        sigma, omega, rho, phi: Meson fields (MeV)
        params: Model parameters
        
    Returns:
        Tuple of (P_meson, e_meson) in MeV/fm³
    """
    # Full scalar potential V(σ) including mass term
    sigma_sq = sigma**2
    V_sigma = (0.5 * params.m_sigma**2 * sigma_sq 
               + (params.g2 / 3.0) * sigma**3 
               + (params.g3 / 4.0) * sigma**4)
    
    # Vector contributions
    omega_sq = omega**2
    rho_sq = rho**2
    phi_sq = phi**2
    
    # ω contribution
    P_omega = 0.5 * params.m_omega**2 * omega_sq + (params.c3 / 4.0) * omega**4
    e_omega = 0.5 * params.m_omega**2 * omega_sq + (3.0 * params.c3 / 4.0) * omega**4
    
    # ρ contribution (including A-function)
    A = params.compute_A(sigma, omega)
    P_rho = 0.5 * params.m_rho**2 * rho_sq + (params.c4 / 4.0) * rho**4 + A * rho_sq
    e_rho = 0.5 * params.m_rho**2 * rho_sq + (3.0 * params.c4 / 4.0) * rho**4 + A * rho_sq
    
    # φ contribution
    P_phi = 0.5 * params.m_phi**2 * phi_sq
    e_phi = 0.5 * params.m_phi**2 * phi_sq
    
    # Total (in MeV⁴, need to convert to MeV/fm³)
    # Divide by (ℏc)³ to get MeV/fm³
    P_meson = (-V_sigma + P_omega + P_rho + P_phi) / hc3
    e_meson = (V_sigma + e_omega + e_rho + e_phi) / hc3
    
    return P_meson, e_meson


def compute_pseudoscalar_meson_thermo(
    T: float,
    mu_Q: float, mu_S: float,
    omega: float, rho: float,
    params: SFHoParams,
    include_pions: bool = True,
    include_kaons: bool = True,
    include_etas: bool = True
) -> MesonThermoResult:
    """
    Compute thermodynamic quantities for pseudoscalar mesons (π, K, η).
    
    Mesons are treated as free Bose gas with effective chemical potentials
    shifted by the vector meson fields:
    
    Pions (I=1, S=0):
        π⁺: μ_eff = +μ_Q - g_ρN × ρ
        π⁻: μ_eff = -μ_Q + g_ρN × ρ
        π⁰: μ_eff = 0
        
    Kaons (I=1/2, S=±1):
        K⁺:  μ_eff = +μ_Q - μ_S - (g_ωN - g_ωΛ) × ω - (1/2) g_ρN × ρ
        K⁰:  μ_eff = -μ_S - (g_ωN - g_ωΛ) × ω + (1/2) g_ρN × ρ
        K⁻:  μ_eff = -μ_Q + μ_S + (g_ωN - g_ωΛ) × ω + (1/2) g_ρN × ρ
        K̄⁰:  μ_eff = +μ_S + (g_ωN - g_ωΛ) × ω - (1/2) g_ρN × ρ
        
    Eta (I=0, S=0):
        η, η': μ_eff = 0
    
    Note: Bose-Einstein condensation occurs when μ_eff → m. This function
    does not handle the condensed phase.
    
    Args:
        T: Temperature (MeV)
        mu_Q: Charge chemical potential (MeV)
        mu_S: Strangeness chemical potential (MeV)
        omega: ω-meson field (MeV)
        rho: ρ-meson field (MeV)
        params: SFHoParams (for meson masses and couplings)
        include_pions: Include π mesons
        include_kaons: Include K mesons
        include_etas: Include η, η' mesons
        
    Returns:
        MesonThermoResult with all meson thermodynamics
    """
    # Initialize totals
    n_Q_tot = 0.0
    n_S_tot = 0.0
    P_tot = 0.0
    e_tot = 0.0
    s_tot = 0.0
    densities = {}
    
    if T <= 0:
        return MesonThermoResult(
            n_Q_mesons=0.0, n_S_mesons=0.0,
            P_mesons=0.0, e_mesons=0.0, s_mesons=0.0,
            densities={}
        )
    
    # Get relevant couplings from params
    g_rho_N = params.get_coupling('n', 'rho')  # Nucleon-rho coupling
    g_omega_N = params.get_coupling('n', 'omega')  # Nucleon-omega coupling
    g_omega_Lambda = params.get_coupling('Lambda', 'omega')  # Lambda-omega coupling
    
    # Omega shift for kaons: (g_ωN - g_ωΛ)
    delta_g_omega = g_omega_N - g_omega_Lambda
    
    # Pions (g=1 for each, spin-0)
    if include_pions:
        m_pi = params.m_pi_pm
        
        # π⁺: μ_eff = +μ_Q - g_ρN × ρ
        mu_pip_eff = mu_Q - g_rho_N * rho
        if abs(mu_pip_eff) < m_pi:  # No condensation
            n_pip, P_pip, e_pip, s_pip, _ = solve_bose_jel(mu_pip_eff, T, m_pi, g=1.0, include_antiparticles=False)
            densities['pi+'] = n_pip
            n_Q_tot += n_pip  # Q = +1
            P_tot += P_pip
            e_tot += e_pip
            s_tot += s_pip
        else:
            densities['pi+'] = 0.0
        
        # π⁻: μ_eff = -μ_Q + g_ρN × ρ
        mu_pim_eff = -mu_Q + g_rho_N * rho
        if abs(mu_pim_eff) < m_pi:
            n_pim, P_pim, e_pim, s_pim, _ = solve_bose_jel(mu_pim_eff, T, m_pi, g=1.0, include_antiparticles=False)
            densities['pi-'] = n_pim
            n_Q_tot -= n_pim  # Q = -1
            P_tot += P_pim
            e_tot += e_pim
            s_tot += s_pim
        else:
            densities['pi-'] = 0.0
        
        # π⁰: μ_eff = 0
        m_pi0 = params.m_pi_0
        n_pi0, P_pi0, e_pi0, s_pi0, _ = solve_bose_jel(0.0, T, m_pi0, g=1.0, include_antiparticles=False)
        densities['pi0'] = n_pi0
        P_tot += P_pi0
        e_tot += e_pi0
        s_tot += s_pi0
    
    # Kaons (g=1 for each)
    if include_kaons:
        m_k_pm = params.m_kaon_pm
        m_k_0 = params.m_kaon_0
        
        # K⁺ (us̄): Q=+1, S=-1 → μ_eff = μ_Q - μ_S - Δg_ω × ω - (1/2) g_ρN × ρ
        mu_kp_eff = mu_Q - mu_S - delta_g_omega * omega - 0.5 * g_rho_N * rho
        if abs(mu_kp_eff) < m_k_pm:
            n_kp, P_kp, e_kp, s_kp, _ = solve_bose_jel(mu_kp_eff, T, m_k_pm, g=1.0, include_antiparticles=False)
            densities['K+'] = n_kp
            n_Q_tot += n_kp      # Q = +1
            n_S_tot -= n_kp      # S = -1
            P_tot += P_kp
            e_tot += e_kp
            s_tot += s_kp
        else:
            densities['K+'] = 0.0
        
        # K⁰ (ds̄): Q=0, S=-1 → μ_eff = -μ_S - Δg_ω × ω + (1/2) g_ρN × ρ
        mu_k0_eff = -mu_S - delta_g_omega * omega + 0.5 * g_rho_N * rho
        if abs(mu_k0_eff) < m_k_0:
            n_k0, P_k0, e_k0, s_k0, _ = solve_bose_jel(mu_k0_eff, T, m_k_0, g=1.0, include_antiparticles=False)
            densities['K0'] = n_k0
            n_S_tot -= n_k0      # S = -1
            P_tot += P_k0
            e_tot += e_k0
            s_tot += s_k0
        else:
            densities['K0'] = 0.0
        
        # K⁻ (ūs): Q=-1, S=+1 → μ_eff = -μ_Q + μ_S + Δg_ω × ω + (1/2) g_ρN × ρ
        mu_km_eff = -mu_Q + mu_S + delta_g_omega * omega + 0.5 * g_rho_N * rho
        if abs(mu_km_eff) < m_k_pm:
            n_km, P_km, e_km, s_km, _ = solve_bose_jel(mu_km_eff, T, m_k_pm, g=1.0, include_antiparticles=False)
            densities['K-'] = n_km
            n_Q_tot -= n_km      # Q = -1
            n_S_tot += n_km      # S = +1
            P_tot += P_km
            e_tot += e_km
            s_tot += s_km
        else:
            densities['K-'] = 0.0
        
        # K̄⁰ (d̄s): Q=0, S=+1 → μ_eff = +μ_S + Δg_ω × ω - (1/2) g_ρN × ρ
        mu_k0bar_eff = mu_S + delta_g_omega * omega - 0.5 * g_rho_N * rho
        if abs(mu_k0bar_eff) < m_k_0:
            n_k0bar, P_k0bar, e_k0bar, s_k0bar, _ = solve_bose_jel(mu_k0bar_eff, T, m_k_0, g=1.0, include_antiparticles=False)
            densities['K0_bar'] = n_k0bar
            n_S_tot += n_k0bar   # S = +1
            P_tot += P_k0bar
            e_tot += e_k0bar
            s_tot += s_k0bar
        else:
            densities['K0_bar'] = 0.0
    
    # Eta mesons (g=1, μ_eff=0)
    if include_etas:
        # η
        m_eta = params.m_eta
        n_eta, P_eta, e_eta, s_eta, _ = solve_bose_jel(0.0, T, m_eta, g=1.0, include_antiparticles=False)
        densities['eta'] = n_eta
        P_tot += P_eta
        e_tot += e_eta
        s_tot += s_eta
        
        # η'
        m_etap = params.m_eta_prime
        n_etap, P_etap, e_etap, s_etap, _ = solve_bose_jel(0.0, T, m_etap, g=1.0, include_antiparticles=False)
        densities['eta_prime'] = n_etap
        P_tot += P_etap
        e_tot += e_etap
        s_tot += s_etap
    
    return MesonThermoResult(
        n_Q_mesons=n_Q_tot,
        n_S_mesons=n_S_tot,
        P_mesons=P_tot,
        e_mesons=e_tot,
        s_mesons=s_tot,
        densities=densities
    )


def compute_total_pressure(
    T: float,
    mu_B: float, mu_Q: float, mu_S: float,
    sigma: float, omega: float, rho: float, phi: float,
    particles: List[Particle],
    params: SFHoParams,
    include_pseudoscalar_mesons: bool = False
) -> Tuple[float, float, float, HadronThermoResult, Optional[MesonThermoResult]]:
    """
    Compute total hadronic pressure, energy density, and entropy density.
    
    P_total = P_baryons + P_mean_field_mesons + P_pseudoscalar_mesons
    e_total = e_baryons + e_mean_field_mesons + e_pseudoscalar_mesons
    s_total = s_baryons + s_pseudoscalar_mesons
    
    Args:
        T: Temperature (MeV)
        mu_B, mu_Q, mu_S: Chemical potentials (MeV)
        sigma, omega, rho, phi: Mean-field meson fields (MeV)
        particles: List of baryon species
        params: Model parameters
        include_pseudoscalar_mesons: If True, include π, K, η contributions
        
    Returns:
        Tuple of (P_total, e_total, s_total, hadron_result, meson_result)
        meson_result is None if include_pseudoscalar_mesons=False
    """
    # Baryon contributions
    hadron_result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, particles, params
    )
    
    # Mean-field meson contributions (σ, ω, ρ, φ)
    P_mf, e_mf = compute_meson_contribution(sigma, omega, rho, phi, params)
    
    # Total from baryons + mean-field mesons
    P_total = hadron_result.P_hadrons + P_mf
    e_total = hadron_result.e_hadrons + e_mf
    s_total = hadron_result.s_hadrons
    
    # Optional: pseudoscalar mesons (π, K, η)
    meson_result = None
    if include_pseudoscalar_mesons:
        meson_result = compute_pseudoscalar_meson_thermo(T, mu_Q, mu_S, omega, rho, params)
        P_total += meson_result.P_mesons
        e_total += meson_result.e_mesons
        s_total += meson_result.s_mesons
    
    return P_total, e_total, s_total, hadron_result, meson_result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_residual_vector(
    fields: np.ndarray,
    T: float,
    mu_B: float, mu_Q: float, mu_S: float,
    particles: List[Particle],
    params: SFHoParams
) -> np.ndarray:
    """
    Compute residual vector for self-consistent field solver.
    
    This function is designed to be used with scipy.optimize.fsolve or similar.
    
    Args:
        fields: Array [sigma, omega, rho, phi] of meson fields (MeV)
        T: Temperature (MeV)
        mu_B, mu_Q, mu_S: Chemical potentials (MeV)
        particles: List of hadron species
        params: Model parameters
        
    Returns:
        Array of residuals [res_σ, res_ω, res_ρ, res_φ]
    """
    sigma, omega, rho, phi = fields
    
    # Compute hadron thermodynamics to get source terms
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, particles, params
    )
    
    # Compute field equation residuals
    res_sigma, res_omega, res_rho, res_phi = compute_field_residuals(
        sigma, omega, rho, phi,
        result.src_sigma, result.src_omega, result.src_rho, result.src_phi,
        params
    )
    
    return np.array([res_sigma, res_omega, res_rho, res_phi])


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    from particles import Proton, Neutron, Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM
    from particles import DeltaPP, DeltaP, Delta0, DeltaM
    from sfho_parameters import get_sfho_2fam_phi, print_params_summary
    
    print("Hadron Thermodynamics Module (SFHo)")
    print("=" * 70)
    
    # Load parameters
    params = get_sfho_2fam_phi()
    print_params_summary(params)
    
    # Test parameters
    T = 10.0  # MeV
    mu_B = 950.0  # MeV (typical for n ~ n_sat)
    mu_Q = -20.0  # MeV (slightly neutron-rich)
    mu_S = 0.0    # MeV
    
    # Initial guess for fields (MeV)
    sigma = 50.0  # Attractive scalar
    omega = 100.0 # Repulsive vector
    rho = 5.0     # Isovector (small for near-symmetric)
    phi = 0.0     # Strangeness (zero for nucleons only)
    
    # Test with nucleons only
    print("\n" + "=" * 70)
    print("TEST 1: Nucleons only")
    print("-" * 50)
    nucleons = [Proton, Neutron]
    
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, nucleons, params
    )
    
    print(f"T = {T} MeV, μ_B = {mu_B} MeV, μ_Q = {mu_Q} MeV")
    print(f"σ = {sigma} MeV, ω = {omega} MeV, ρ = {rho} MeV")
    print()
    print(f"n_B = {result.n_B:.4e} fm⁻³")
    print(f"n_Q = {result.n_Q:.4e} fm⁻³")
    print(f"P_hadrons = {result.P_hadrons:.4e} MeV/fm³")
    print(f"e_hadrons = {result.e_hadrons:.4e} MeV/fm³")
    print(f"s_hadrons = {result.s_hadrons:.4e} fm⁻³")
    print()
    print("Individual states:")
    for name, state in result.states.items():
        print(f"  {name}: n={state.n:.4e}, m*={state.m_eff:.1f} MeV, μ*={state.mu_eff:.1f} MeV")
    
    # Field residuals
    res = compute_field_residuals(
        sigma, omega, rho, phi,
        result.src_sigma, result.src_omega, result.src_rho, result.src_phi,
        params
    )
    print(f"\nField residuals (should be ~0 at solution):")
    print(f"  res_σ = {res[0]:.4e}")
    print(f"  res_ω = {res[1]:.4e}")
    print(f"  res_ρ = {res[2]:.4e}")
    print(f"  res_φ = {res[3]:.4e}")
    
    # Meson contribution
    P_m, e_m = compute_meson_contribution(sigma, omega, rho, phi, params)
    print(f"\nMeson contributions:")
    print(f"  P_meson = {P_m:.4e} MeV/fm³")
    print(f"  e_meson = {e_m:.4e} MeV/fm³")
    
    # Test with hyperons
    print("\n" + "=" * 70)
    print("TEST 2: Nucleons + Hyperons")
    print("-" * 50)
    
    mu_S = 50.0  # Non-zero strangeness potential
    phi = 10.0   # Non-zero φ field
    
    baryons = [Proton, Neutron, Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
    
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, baryons, params
    )
    
    print(f"T = {T} MeV, μ_B = {mu_B} MeV, μ_Q = {mu_Q} MeV, μ_S = {mu_S} MeV")
    print()
    print(f"n_B = {result.n_B:.4e} fm⁻³")
    print(f"n_S = {result.n_S:.4e} fm⁻³")
    print(f"P_hadrons = {result.P_hadrons:.4e} MeV/fm³")
    print()
    print("Individual states:")
    for name, state in result.states.items():
        if abs(state.n) > 1e-10:
            print(f"  {name:10s}: n={state.n:.4e}, m*={state.m_eff:.1f} MeV")
    
    # Test with Deltas
    print("\n" + "=" * 70)
    print("TEST 3: Full baryon octet + Deltas")
    print("-" * 50)
    
    all_baryons = baryons + [DeltaPP, DeltaP, Delta0, DeltaM]
    
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, all_baryons, params
    )
    
    print(f"Total baryons: {len(all_baryons)}")
    print(f"n_B = {result.n_B:.4e} fm⁻³")
    print()
    print("Non-zero densities:")
    for name, state in result.states.items():
        if abs(state.n) > 1e-10:
            print(f"  {name:10s}: n={state.n:.4e}")
    
    # Test pseudoscalar mesons
    print("\n" + "=" * 70)
    print("TEST 4: Pseudoscalar mesons (π, K, η)")
    print("-" * 50)
    
    T_high = 50.0  # Higher T to have significant meson density
    mu_Q_test = 10.0
    mu_S_test = 20.0
    
    print(f"T = {T_high} MeV, μ_Q = {mu_Q_test} MeV, μ_S = {mu_S_test} MeV")
    print()
    
    meson_result = compute_pseudoscalar_meson_thermo(T_high, mu_Q_test, mu_S_test, params)
    
    print(f"n_Q (mesons) = {meson_result.n_Q_mesons:.4e} fm⁻³")
    print(f"n_S (mesons) = {meson_result.n_S_mesons:.4e} fm⁻³")
    print(f"P_mesons = {meson_result.P_mesons:.4e} MeV/fm³")
    print(f"e_mesons = {meson_result.e_mesons:.4e} MeV/fm³")
    print(f"s_mesons = {meson_result.s_mesons:.4e} fm⁻³")
    print()
    print("Individual meson densities:")
    for name, n in meson_result.densities.items():
        if n > 1e-10:
            print(f"  {name:10s}: n = {n:.4e} fm⁻³")
    
    # Test compute_total_pressure with mesons
    print("\n" + "=" * 70)
    print("TEST 5: Total EOS with baryons and mesons")
    print("-" * 50)
    
    P_tot, e_tot, s_tot, hadron_res, meson_res = compute_total_pressure(
        T_high, mu_B, mu_Q_test, mu_S_test, sigma, omega, rho, phi,
        all_baryons, params, include_pseudoscalar_mesons=True
    )
    
    print(f"P_total = {P_tot:.4e} MeV/fm³")
    print(f"e_total = {e_tot:.4e} MeV/fm³")
    print(f"s_total = {s_tot:.4e} fm⁻³")
    if meson_res:
        print(f"\nMeson contribution to P: {meson_res.P_mesons:.4e} MeV/fm³")
    
    print("\n" + "=" * 70)
    print("All tests completed!")