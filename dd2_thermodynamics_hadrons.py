"""
dd2_thermodynamics_hadrons.py
================================
Model-dependent hadron thermodynamics for DD2-type RMF models
with density-dependent couplings.

Key differences from SFHo:
1. Density-dependent couplings: g_M(n_B) = g_M(n_0) × h_M(x)
2. Rearrangement term R^0 required for thermodynamic consistency
3. No nonlinear self-interactions (g2 = g3 = c3 = c4 = A = 0)

This module computes thermodynamic quantities for:
- Baryons (nucleons, hyperons, deltas) in mean-field approximation
- Pseudoscalar mesons (pions, kaons, etas) as free Bose gas

Key functions:
1. Compute effective masses and chemical potentials for baryons
2. Compute rearrangement term R^0
3. Call Fermi integrals for each baryon species
4. Compute source terms for meson field equations
5. Compute total pressure including all contributions

Units:
- Energies/masses: MeV
- Lengths: fm
- Number density: fm⁻³
- Pressure/energy density: MeV/fm³
- Entropy density: fm⁻³
- Meson fields: MeV

References:
- Typel, Röpke, Klähn, Blaschke, Wolter, Phys. Rev. C 81 (2010) 015803
- Fortin, Oertel, Providência, PASA 35 (2018) e044
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from general_physics_constants import hc, hc3
from general_particles import Particle
from general_fermi_integrals import solve_fermi_jel
from general_bose_integrals import solve_bose_jel
from dd2_parameters import DD2Params


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
    n: float
    ns: float
    P: float
    e: float
    s: float
    mu_eff: float
    m_eff: float
    
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
        R0: Rearrangement term (MeV)
    """
    states: Dict[str, HadronState]
    n_B: float
    n_Q: float
    n_S: float
    P_hadrons: float
    e_hadrons: float
    s_hadrons: float
    src_sigma: float
    src_omega: float
    src_rho: float
    src_phi: float
    R0: float  # Rearrangement term


@dataclass
class MesonThermoResult:
    """
    Thermodynamic result for pseudoscalar mesons (π, K, η).
    """
    n_Q_mesons: float
    n_S_mesons: float
    P_mesons: float
    e_mesons: float
    s_mesons: float
    densities: Dict[str, float]


# =============================================================================
# REARRANGEMENT TERM CALCULATION
# =============================================================================

def compute_rearrangement_term(
    n_B: float,
    sigma: float, omega: float, rho: float, phi: float,
    particle_states: Dict[str, Tuple[float, float, float, Particle]],
    params: DD2Params
) -> float:
    """
    Compute the rearrangement term R^0 for density-dependent couplings.
    
    From Eq. (15) of Fortin et al. 2017:
    
    R^0 = Σⱼ [∂g_ωⱼ/∂nⱼ × ω × nⱼ + ∂g_ρⱼ/∂nⱼ × ρ × I₃ⱼ × nⱼ 
            + ∂g_φⱼ/∂nⱼ × φ × nⱼ - ∂g_σⱼ/∂nⱼ × σ × n^s_j]
    
    Note: The derivatives ∂g_Mⱼ/∂nⱼ = g_M(n_0) × R_Mⱼ × (dh_M/dn_B)
    
    Args:
        n_B: Total baryon density (fm⁻³)
        sigma, omega, rho, phi: Meson fields (MeV)
        particle_states: Dict of (n, ns, I3, Particle) for each species
        params: DD2Params with coupling functions
        
    Returns:
        R^0 in MeV
    """
    if n_B <= 0:
        return 0.0
    
    R0 = 0.0
    
    for p_name, (n_j, ns_j, I3_j, particle) in particle_states.items():
        # Get coupling derivatives ∂g/∂n_B
        dg_sigma_dn = params.get_coupling_derivative(p_name, 'sigma', n_B)
        dg_omega_dn = params.get_coupling_derivative(p_name, 'omega', n_B)
        dg_rho_dn = params.get_coupling_derivative(p_name, 'rho', n_B)
        dg_phi_dn = params.get_coupling_derivative(p_name, 'phi', n_B)
        
        # Rearrangement contribution from this species
        # Note: The sign for σ is negative because it's attractive
        R0 += (dg_omega_dn * omega * n_j 
               + dg_rho_dn * I3_j * rho * n_j
               + dg_phi_dn * phi * n_j
               - dg_sigma_dn * sigma * ns_j)
    
    return R0


# =============================================================================
# MAIN THERMODYNAMICS FUNCTIONS
# =============================================================================

def compute_hadron_thermo(
    T: float,
    mu_B: float, mu_Q: float, mu_S: float,
    sigma: float, omega: float, rho: float, phi: float,
    n_B_input: float,
    particles: List[Particle],
    params: DD2Params
) -> HadronThermoResult:
    """
    Compute thermodynamic quantities for all hadron species.
    
    Key difference from SFHo: couplings depend on n_B, and there's a
    rearrangement term R^0 in the effective chemical potential.
    
    Given temperature, chemical potentials, and meson fields, this function:
    1. Computes effective masses: M*_j = m_j - g_σj(n_B) × σ
    2. Computes effective chemical potentials (including R^0):
       μ*_j = B_j×μ_B + Q_j×μ_Q + S_j×μ_S - g_ωj×ω - g_ρj×I₃j×ρ - g_φj×φ - R^0
    3. Evaluates Fermi integrals for (n, P, e, s, n_s)
    4. Computes source terms for field equations
    
    Note: This requires an iterative approach since couplings depend on n_B,
    but n_B is an output. We use n_B_input as the density for coupling evaluation.
    
    Args:
        T: Temperature (MeV)
        mu_B: Baryon chemical potential (MeV)
        mu_Q: Charge chemical potential (MeV)
        mu_S: Strangeness chemical potential (MeV)
        sigma: σ-meson field (MeV)
        omega: ω-meson field (MeV)
        rho: ρ-meson field (MeV)
        phi: φ-meson field (MeV)
        n_B_input: Input baryon density for coupling evaluation (fm⁻³)
        particles: List of Particle objects to include
        params: DD2Params with model parameters
        
    Returns:
        HadronThermoResult with all thermodynamic quantities
    """
    states = {}
    particle_data = {}  # For rearrangement term
    
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
    
    # Use input n_B for coupling evaluation
    n_B_for_couplings = max(n_B_input, 1e-10)  # Avoid division by zero
    
    # First pass: compute particle densities and thermodynamics
    for p in particles:
        # 1. Get meson-baryon couplings at current density
        g_s = params.get_coupling(p.name, 'sigma', n_B_for_couplings)
        g_w = params.get_coupling(p.name, 'omega', n_B_for_couplings)
        g_r = params.get_coupling(p.name, 'rho', n_B_for_couplings)
        g_p = params.get_coupling(p.name, 'phi', n_B_for_couplings)
        
        # 2. Get baryon mass from parametrization
        m_baryon = params.get_baryon_mass(p.name)
        if m_baryon == 0.0:
            m_baryon = p.mass
        
        # 3. Effective mass: M* = m - g_σ × σ
        m_eff = m_baryon - g_s * sigma
        if m_eff < 0:
            m_eff = 1e-3
        
        # 4. Non-interacting chemical potential
        mu_nonint = p.baryon_no * mu_B + p.charge * mu_Q + p.strangeness * mu_S
        
        # 5. Vector field shifts (rearrangement added later)
        vector_shift = g_w * omega + g_r * p.isospin_3 * rho + g_p * phi
        mu_eff = mu_nonint - vector_shift
        
        # 6. Compute Fermi integrals
        n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m_eff, p.g_degen,
                                          include_antiparticles=True)
        
        # Store for rearrangement calculation
        particle_data[p.name] = (n, ns, p.isospin_3, p)
        
        # Store individual state (without R0 correction for now)
        states[p.name] = HadronState(
            n=n, ns=ns, P=P, e=e, s=s, mu_eff=mu_eff, m_eff=m_eff
        )
        
        # Accumulate totals
        n_B_tot += p.baryon_no * n
        n_Q_tot += p.charge * n
        n_S_tot += p.strangeness * n
        P_tot += P
        e_tot += e
        s_tot += s
        
        # Source terms for field equations (using density-dependent couplings)
        src_sigma += g_s * ns
        src_omega += g_w * n
        src_phi += g_p * n
        src_rho += g_r * p.isospin_3 * n
    
    # Compute rearrangement term
    R0 = compute_rearrangement_term(
        n_B_for_couplings, sigma, omega, rho, phi, particle_data, params
    )
    
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
        src_phi=src_phi,
        R0=R0
    )


def compute_hadron_thermo_with_R0(
    T: float,
    mu_B: float, mu_Q: float, mu_S: float,
    sigma: float, omega: float, rho: float, phi: float,
    n_B_input: float,
    R0: float,
    particles: List[Particle],
    params: DD2Params
) -> HadronThermoResult:
    """
    Compute hadron thermodynamics with pre-computed rearrangement term R^0.
    
    This version includes R^0 in the effective chemical potential from the start,
    which is the correct thermodynamic formulation.
    
    μ*_j = B_j×μ_B + Q_j×μ_Q + S_j×μ_S - g_ωj×ω - g_ρj×I₃j×ρ - g_φj×φ - R^0
    """
    states = {}
    particle_data = {}
    
    n_B_tot = 0.0
    n_Q_tot = 0.0
    n_S_tot = 0.0
    P_tot = 0.0
    e_tot = 0.0
    s_tot = 0.0
    
    src_sigma = 0.0
    src_omega = 0.0
    src_rho = 0.0
    src_phi = 0.0
    
    n_B_for_couplings = max(n_B_input, 1e-10)
    
    for p in particles:
        g_s = params.get_coupling(p.name, 'sigma', n_B_for_couplings)
        g_w = params.get_coupling(p.name, 'omega', n_B_for_couplings)
        g_r = params.get_coupling(p.name, 'rho', n_B_for_couplings)
        g_p = params.get_coupling(p.name, 'phi', n_B_for_couplings)
        
        m_baryon = params.get_baryon_mass(p.name)
        if m_baryon == 0.0:
            m_baryon = p.mass
        
        m_eff = m_baryon - g_s * sigma
        if m_eff < 0:
            m_eff = 1e-3
        
        mu_nonint = p.baryon_no * mu_B + p.charge * mu_Q + p.strangeness * mu_S
        
        # Include R0 in effective chemical potential
        vector_shift = g_w * omega + g_r * p.isospin_3 * rho + g_p * phi + R0
        mu_eff = mu_nonint - vector_shift
        
        n, P, e, s, ns = solve_fermi_jel(mu_eff, T, m_eff, p.g_degen,
                                          include_antiparticles=True)
        
        particle_data[p.name] = (n, ns, p.isospin_3, p)
        
        states[p.name] = HadronState(
            n=n, ns=ns, P=P, e=e, s=s, mu_eff=mu_eff, m_eff=m_eff
        )
        
        n_B_tot += p.baryon_no * n
        n_Q_tot += p.charge * n
        n_S_tot += p.strangeness * n
        P_tot += P
        e_tot += e
        s_tot += s
        
        src_sigma += g_s * ns
        src_omega += g_w * n
        src_phi += g_p * n
        src_rho += g_r * p.isospin_3 * n
    
    # Recompute R0 for consistency check
    R0_new = compute_rearrangement_term(
        n_B_for_couplings, sigma, omega, rho, phi, particle_data, params
    )
    
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
        src_phi=src_phi,
        R0=R0_new
    )


def compute_field_residuals(
    sigma: float, omega: float, rho: float, phi: float,
    src_sigma: float, src_omega: float, src_rho: float, src_phi: float,
    params: DD2Params
) -> Tuple[float, float, float, float]:
    """
    Compute residuals of the meson field equations for DD2.
    
    For DD2 with density-dependent couplings, the field equations are
    LINEAR (no self-interactions):
    
    σ: m²_σ σ = Σⱼ g_σⱼ(n_B) n^s_j × (ℏc)³
    ω: m²_ω ω = Σⱼ g_ωⱼ(n_B) nⱼ × (ℏc)³
    ρ: m²_ρ ρ = Σⱼ g_ρⱼ(n_B) I₃ⱼ nⱼ × (ℏc)³
    φ: m²_φ φ = Σⱼ g_φⱼ(n_B) nⱼ × (ℏc)³
    
    Residual = LHS - RHS (should be zero at solution)
    
    Args:
        sigma, omega, rho, phi: Meson fields (MeV)
        src_sigma, src_omega, src_rho, src_phi: Source terms (fm⁻³)
        params: Model parameters
        
    Returns:
        Tuple of (res_sigma, res_omega, res_rho, res_phi) in MeV³
    """
    # σ equation: m²σ = g_σ n_s × hc³
    lhs_sigma = params.m_sigma**2 * sigma
    rhs_sigma = src_sigma * hc3
    res_sigma = lhs_sigma - rhs_sigma
    
    # ω equation: m²ω = g_ω n × hc³
    lhs_omega = params.m_omega**2 * omega
    rhs_omega = src_omega * hc3
    res_omega = lhs_omega - rhs_omega
    
    # ρ equation: m²ρ = g_ρ I₃ n × hc³
    lhs_rho = params.m_rho**2 * rho
    rhs_rho = src_rho * hc3
    res_rho = lhs_rho - rhs_rho
    
    # φ equation: m²φ = g_φ n × hc³
    lhs_phi = params.m_phi**2 * phi
    rhs_phi = src_phi * hc3
    res_phi = lhs_phi - rhs_phi
    
    return res_sigma, res_omega, res_rho, res_phi


def compute_meson_contribution(
    sigma: float, omega: float, rho: float, phi: float,
    params: DD2Params
) -> Tuple[float, float]:
    """
    Compute meson field contributions to pressure and energy density.
    
    For DD2 with NO self-interactions:
    
    P_meson = -½m²_σ σ² + ½m²_ω ω² + ½m²_ρ ρ² + ½m²_φ φ²
    e_meson = +½m²_σ σ² + ½m²_ω ω² + ½m²_ρ ρ² + ½m²_φ φ²
    
    The sign difference for σ reflects the attractive nature of scalar exchange.
    
    Args:
        sigma, omega, rho, phi: Meson fields (MeV)
        params: Model parameters
        
    Returns:
        Tuple of (P_meson, e_meson) in MeV/fm³
    """
    sigma_sq = sigma**2
    omega_sq = omega**2
    rho_sq = rho**2
    phi_sq = phi**2
    
    # Scalar contribution (attractive → negative pressure)
    V_sigma = 0.5 * params.m_sigma**2 * sigma_sq
    
    # Vector contributions (repulsive → positive pressure)
    V_omega = 0.5 * params.m_omega**2 * omega_sq
    V_rho = 0.5 * params.m_rho**2 * rho_sq
    V_phi = 0.5 * params.m_phi**2 * phi_sq
    
    # Total (in MeV⁴, convert to MeV/fm³)
    P_meson = (-V_sigma + V_omega + V_rho + V_phi) / hc3
    e_meson = (V_sigma + V_omega + V_rho + V_phi) / hc3
    
    return P_meson, e_meson


def compute_rearrangement_contribution(
    n_B: float,
    R0: float
) -> Tuple[float, float]:
    """
    Compute rearrangement term contribution to pressure.
    
    For density-dependent couplings, the rearrangement term R^0
    contributes to both pressure and energy density.
    
    The contribution to pressure from the rearrangement is:
        P_rearr = n_B × R^0
        
    And the contribution to energy density is zero (it cancels).
    
    Args:
        n_B: Total baryon density (fm⁻³)
        R0: Rearrangement term (MeV)
        
    Returns:
        Tuple of (P_rearr, e_rearr) in MeV/fm³
    """
    P_rearr = n_B * R0
    e_rearr = 0.0  # Cancels in energy density
    
    return P_rearr, e_rearr


def compute_pseudoscalar_meson_thermo(
    T: float,
    mu_Q: float, mu_S: float,
    params: DD2Params,
    include_pions: bool = True,
    include_kaons: bool = True,
    include_etas: bool = True
) -> MesonThermoResult:
    """
    Compute thermodynamic quantities for pseudoscalar mesons (π, K, η).
    
    Mesons are treated as free Bose gas. For DD2, we don't include
    vector field shifts for mesons (simpler treatment than SFHo).
    
    Chemical potentials:
        π⁺: μ = +μ_Q
        π⁻: μ = -μ_Q
        π⁰: μ = 0
        K⁺: μ = +μ_Q - μ_S
        K⁰: μ = -μ_S
        K⁻: μ = -μ_Q + μ_S
        K̄⁰: μ = +μ_S
        η, η': μ = 0
    """
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
    
    def add_boson(name: str, mass: float, mu: float, charge: float, strangeness: float):
        nonlocal n_Q_tot, n_S_tot, P_tot, e_tot, s_tot
        
        # Clamp μ to avoid condensation
        if mu >= mass:
            mu = mass * 0.99
        
        n, P, e, s, _ = solve_bose_jel(mu, T, mass, 1.0, include_antiparticles=False)
        
        densities[name] = n
        n_Q_tot += charge * n
        n_S_tot += strangeness * n
        P_tot += P
        e_tot += e
        s_tot += s
    
    # Pions
    if include_pions:
        m_pi = params.m_pi_pm
        add_boson('pi+', m_pi, mu_Q, +1.0, 0.0)
        add_boson('pi-', m_pi, -mu_Q, -1.0, 0.0)
        add_boson('pi0', params.m_pi_0, 0.0, 0.0, 0.0)
    
    # Kaons
    if include_kaons:
        m_K = params.m_kaon_pm
        m_K0 = params.m_kaon_0
        add_boson('K+', m_K, mu_Q - mu_S, +1.0, -1.0)
        add_boson('K0', m_K0, -mu_S, 0.0, -1.0)
        add_boson('K-', m_K, -mu_Q + mu_S, -1.0, +1.0)
        add_boson('K0_bar', m_K0, mu_S, 0.0, +1.0)
    
    # Etas
    if include_etas:
        add_boson('eta', params.m_eta, 0.0, 0.0, 0.0)
        add_boson('eta_prime', params.m_eta_prime, 0.0, 0.0, 0.0)
    
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
    n_B_input: float,
    particles: List[Particle],
    params: DD2Params,
    include_pseudoscalar_mesons: bool = False
) -> Tuple[float, float, float, HadronThermoResult, Optional[MesonThermoResult]]:
    """
    Compute total pressure, energy density, and entropy density.
    
    Includes:
    - Baryon contributions from Fermi integrals
    - Mean-field meson contributions (σ, ω, ρ, φ)
    - Rearrangement contribution
    - Optional pseudoscalar mesons (π, K, η)
    
    Args:
        T: Temperature (MeV)
        mu_B, mu_Q, mu_S: Chemical potentials (MeV)
        sigma, omega, rho, phi: Mean-field meson fields (MeV)
        n_B_input: Input baryon density for coupling evaluation (fm⁻³)
        particles: List of baryon species
        params: Model parameters
        include_pseudoscalar_mesons: If True, include π, K, η contributions
        
    Returns:
        Tuple of (P_total, e_total, s_total, hadron_result, meson_result)
    """
    # Baryon contributions
    hadron_result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_input, particles, params
    )
    
    # Mean-field meson contributions
    P_mf, e_mf = compute_meson_contribution(sigma, omega, rho, phi, params)
    
    # Rearrangement contribution
    P_rearr, e_rearr = compute_rearrangement_contribution(hadron_result.n_B, hadron_result.R0)
    
    # Total from baryons + mean-field + rearrangement
    P_total = hadron_result.P_hadrons + P_mf + P_rearr
    e_total = hadron_result.e_hadrons + e_mf + e_rearr
    s_total = hadron_result.s_hadrons
    
    # Optional: pseudoscalar mesons
    meson_result = None
    if include_pseudoscalar_mesons:
        meson_result = compute_pseudoscalar_meson_thermo(T, mu_Q, mu_S, params)
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
    n_B_input: float,
    particles: List[Particle],
    params: DD2Params
) -> np.ndarray:
    """
    Compute residual vector for self-consistent field solver.
    
    Args:
        fields: Array [sigma, omega, rho, phi] of meson fields (MeV)
        T: Temperature (MeV)
        mu_B, mu_Q, mu_S: Chemical potentials (MeV)
        n_B_input: Input baryon density (fm⁻³)
        particles: List of hadron species
        params: Model parameters
        
    Returns:
        Array of residuals [res_σ, res_ω, res_ρ, res_φ]
    """
    sigma, omega, rho, phi = fields
    
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_input, particles, params
    )
    
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
    from general_particles import Proton, Neutron, Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM
    from dd2_parameters import get_dd2y_fortin, print_params_summary
    
    print("Hadron Thermodynamics Module (DD2)")
    print("=" * 70)
    
    # Load parameters
    params = get_dd2y_fortin()
    print_params_summary(params)
    
    # Test parameters
    T = 10.0  # MeV
    mu_B = 950.0  # MeV
    mu_Q = -20.0  # MeV
    mu_S = 0.0    # MeV
    n_B_input = params.n_sat  # Start at saturation
    
    # Initial guess for fields (MeV)
    sigma = 50.0
    omega = 100.0
    rho = 5.0
    phi = 0.0
    
    # Test with nucleons only
    print("\n" + "=" * 70)
    print("TEST 1: Nucleons only")
    print("-" * 50)
    nucleons = [Proton, Neutron]
    
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_input, nucleons, params
    )
    
    print(f"T = {T} MeV, μ_B = {mu_B} MeV, μ_Q = {mu_Q} MeV")
    print(f"σ = {sigma} MeV, ω = {omega} MeV, ρ = {rho} MeV")
    print(f"n_B (input) = {n_B_input:.4e} fm⁻³")
    print()
    print(f"n_B (output) = {result.n_B:.4e} fm⁻³")
    print(f"n_Q = {result.n_Q:.4e} fm⁻³")
    print(f"P_hadrons = {result.P_hadrons:.4e} MeV/fm³")
    print(f"e_hadrons = {result.e_hadrons:.4e} MeV/fm³")
    print(f"s_hadrons = {result.s_hadrons:.4e} fm⁻³")
    print(f"R^0 = {result.R0:.4e} MeV")
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
    print(f"\nMeson contributions (mean-field):")
    print(f"  P_meson = {P_m:.4e} MeV/fm³")
    print(f"  e_meson = {e_m:.4e} MeV/fm³")
    
    # Rearrangement contribution
    P_r, e_r = compute_rearrangement_contribution(result.n_B, result.R0)
    print(f"\nRearrangement contribution:")
    print(f"  P_rearr = {P_r:.4e} MeV/fm³")
    print(f"  e_rearr = {e_r:.4e} MeV/fm³")
    
    # Test with hyperons
    print("\n" + "=" * 70)
    print("TEST 2: Nucleons + Hyperons")
    print("-" * 50)
    
    mu_S = 50.0
    phi = 10.0
    
    baryons = [Proton, Neutron, Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
    
    result = compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_input, baryons, params
    )
    
    print(f"T = {T} MeV, μ_B = {mu_B} MeV, μ_Q = {mu_Q} MeV, μ_S = {mu_S} MeV")
    print()
    print(f"n_B = {result.n_B:.4e} fm⁻³")
    print(f"n_S = {result.n_S:.4e} fm⁻³")
    print(f"P_hadrons = {result.P_hadrons:.4e} MeV/fm³")
    print(f"R^0 = {result.R0:.4e} MeV")
    print()
    print("Individual states:")
    for name, state in result.states.items():
        if abs(state.n) > 1e-10:
            print(f"  {name:10s}: n={state.n:.4e}, m*={state.m_eff:.1f} MeV")
    
    # Test total pressure
    print("\n" + "=" * 70)
    print("TEST 3: Total EOS")
    print("-" * 50)
    
    P_tot, e_tot, s_tot, hadron_res, meson_res = compute_total_pressure(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi, n_B_input,
        baryons, params, include_pseudoscalar_mesons=False
    )
    
    print(f"P_total = {P_tot:.4e} MeV/fm³")
    print(f"e_total = {e_tot:.4e} MeV/fm³")
    print(f"s_total = {s_tot:.4e} fm⁻³")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
