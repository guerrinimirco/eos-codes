"""
sfho_eos.py
===========
SFHo-specific equation of state implementation.

This module implements the EOSModel interface for the SFHo relativistic
mean field model family, including:
- SFHo (nucleons only)
- SFHoY (nucleons + hyperons)
- SFHo_HD (nucleons + hyperons + deltas)
- SFHo_HDM (nucleons + hyperons + deltas + mesons)

Usage:
    from eos_sfho import SFHoEOS, solve_beta_equilibrium, generate_beta_eq_table
    
    # Single point
    model = SFHoEOS(particle_content='nucleons_hyperons_deltas')
    result = solve_beta_equilibrium(model, n_B=0.16, T=10.0)
    
    # Table
    n_B_values = np.logspace(-3, 0, 100) * 0.16
    results = generate_beta_eq_table(model, n_B_values, T=10.0)

References:
- Fortin, Oertel, Providência, PASA 35 (2018) e044
- Steiner, Hempel, Fischer, ApJ 774 (2013) 17
"""
import numpy as np
from typing import Dict, Tuple, List, Optional

from general_eos_solver import (
    EOSModel, EOSInput, EOSGuess, EOSResult, TableConfig,
    EquilibriumType, ParticleContent, PrintLevel,
    solve_eos_point, generate_eos_table, print_eos_result, save_table_to_file
)

from general_particles import (
    Particle, Proton, Neutron,
    Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM,
    DeltaPP, DeltaP, Delta0, DeltaM
)

from sfho_parameters import (
    SFHoParams, get_sfho_nucleonic, get_sfhoy_fortin, 
    get_sfhoy_star_fortin, get_sfho_2fam_phi, get_sfho_general
)

from sfho_thermodynamics_hadrons import (
    compute_hadron_thermo as compute_hadron_thermo_sfho,
    compute_field_residuals as compute_field_residuals_sfho,
    compute_meson_contribution,
    compute_pseudoscalar_meson_thermo
)


# =============================================================================
# PARTICLE LISTS
# =============================================================================
NUCLEONS = [Proton, Neutron]
HYPERONS = [Lambda, SigmaP, Sigma0, SigmaM, Xi0, XiM]
DELTAS = [DeltaPP, DeltaP, Delta0, DeltaM]

BARYONS_N = NUCLEONS
BARYONS_NY = NUCLEONS + HYPERONS
BARYONS_NYD = NUCLEONS + HYPERONS + DELTAS


# =============================================================================
# SFHO EOS MODEL CLASS
# =============================================================================
class SFHoEOS(EOSModel):
    """
    SFHo relativistic mean field EOS implementation.
    
    Args:
        particle_content: One of 'nucleons', 'nucleons_hyperons',
                         'nucleons_hyperons_deltas', 'nucleons_hyperons_deltas_mesons'
        parametrization: One of 'sfho', 'sfhoy', 'sfhoy_star', 'sfho_hd', 'custom'
        params: Custom SFHoParams (only if parametrization='custom')
    """
    
    def __init__(
        self,
        particle_content: str = 'nucleons',
        parametrization: str = 'sfho',
        params: Optional[SFHoParams] = None
    ):
        # Set particle content
        pc_lower = particle_content.lower()
        if pc_lower == 'nucleons':
            self._particle_content = ParticleContent.NUCLEONS
            self._baryons = BARYONS_N
        elif pc_lower in ['nucleons_hyperons', 'ny']:
            self._particle_content = ParticleContent.NUCLEONS_HYPERONS
            self._baryons = BARYONS_NY
        elif pc_lower in ['nucleons_hyperons_deltas', 'nyd']:
            self._particle_content = ParticleContent.NUCLEONS_HYPERONS_DELTAS
            self._baryons = BARYONS_NYD
        elif pc_lower in ['nucleons_hyperons_deltas_mesons', 'nydm', 'full']:
            self._particle_content = ParticleContent.NUCLEONS_HYPERONS_DELTAS_MESONS
            self._baryons = BARYONS_NYD
        else:
            raise ValueError(f"Unknown particle content: {particle_content}")
        
        # Set parameters
        if params is not None:
            self._params = params
        else:
            par_lower = parametrization.lower()
            if par_lower == 'sfho':
                self._params = get_sfho_nucleonic()
            elif par_lower == 'sfhoy':
                self._params = get_sfhoy_fortin()
            elif par_lower == 'sfhoy_star':
                self._params = get_sfhoy_star_fortin()
            elif par_lower == '2fam_phi':
                self._params = get_sfho_2fam_phi()
            elif par_lower == '2fam':
                from sfho_parameters import get_sfho_2fam
                self._params = get_sfho_2fam()
            else:
                raise ValueError(f"Unknown parametrization: {parametrization}")
        
        self._name = f"{self._params.name}_{pc_lower}"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def params(self) -> SFHoParams:
        return self._params
    
    @property
    def baryons(self) -> List[Particle]:
        return self._baryons
    
    def get_particle_content(self) -> ParticleContent:
        return self._particle_content
    
    def compute_hadron_thermo(
        self, T: float, mu_B: float, mu_Q: float, mu_S: float,
        sigma: float, omega: float, rho: float, phi: float
    ) -> Tuple[Dict, float, float, float, float, float, float, float, float, float, float]:
        """Compute hadronic thermodynamics using SFHo model."""
        result = compute_hadron_thermo_sfho(
            T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi,
            self._baryons, self._params
        )
        
        return (
            result.states,
            result.n_B, result.n_Q, result.n_S,
            result.P_hadrons, result.e_hadrons, result.s_hadrons,
            result.src_sigma, result.src_omega, result.src_rho, result.src_phi
        )
    
    def compute_meson_field_contribution(
        self, sigma: float, omega: float, rho: float, phi: float
    ) -> Tuple[float, float]:
        """Compute mean-field meson contribution."""
        return compute_meson_contribution(sigma, omega, rho, phi, self._params)
    
    def compute_field_residuals(
        self, sigma: float, omega: float, rho: float, phi: float,
        src_sigma: float, src_omega: float, src_rho: float, src_phi: float
    ) -> Tuple[float, float, float, float]:
        """Compute field equation residuals."""
        return compute_field_residuals_sfho(
            sigma, omega, rho, phi,
            src_sigma, src_omega, src_rho, src_phi,
            self._params
        )
    
    def compute_pseudoscalar_meson_thermo(
        self, T: float, mu_Q: float, mu_S: float,
        omega: float = 0.0, rho: float = 0.0
    ) -> Tuple[float, float, float, float, float, Dict]:
        """Compute pseudoscalar meson thermodynamics."""
        result = compute_pseudoscalar_meson_thermo(
            T, mu_Q, mu_S, omega, rho, self._params
        )
        return (
            result.n_Q_mesons, result.n_S_mesons,
            result.P_mesons, result.e_mesons, result.s_mesons,
            result.densities
        )
    
    def get_default_guess(self, n_B: float, T: float) -> EOSGuess:
        """
        Get default initial guess based on density.
        
        Based on self-consistent solutions for symmetric nuclear matter:
        - At saturation (n_B = 0.16 fm⁻³): σ ~ 30 MeV, ω ~ 19 MeV, m*/m ~ 0.76
        - Fields scale roughly linearly with density
        
        The key insight is that g_σN ~ 7.5 and m*/m ~ 0.76 at saturation gives:
        σ = M_N(1 - m*/m) / g_σN = 939 × 0.24 / 7.5 ≈ 30 MeV
        """
        n_sat = 0.158  # fm^-3 (SFHo saturation density)
        ratio = n_B / n_sat
        
        # Field values at saturation (from self-consistent calculation):
        # sigma_sat ~ 30 MeV, omega_sat ~ 19 MeV, mu_B_sat ~ 922 MeV
        
        if ratio < 0.05:
            # Very low density: minimal fields (nearly linear scaling)
            sigma = 30.0 * ratio / 0.5  # ~3 MeV at 0.05 n_sat
            omega = 19.0 * ratio / 0.5  # ~1.9 MeV at 0.05 n_sat
            mu_B = 939.0 + T  # Near free nucleon
            mu_Q = -5.0
        elif ratio < 0.5:
            # Low density: linear scaling from saturation values
            sigma = 30.0 * ratio  # 0-15 MeV
            omega = 19.0 * ratio  # 0-9.5 MeV
            mu_B = 920.0 - 10.0 * (1.0 - ratio)  # Approach 920 from below
            mu_Q = -20.0 * ratio  # More negative as density increases
        elif ratio < 1.5:
            # Around saturation: linear interpolation
            sigma = 30.0 * ratio  # 15-45 MeV
            omega = 19.0 * ratio  # 9.5-28.5 MeV
            mu_B = 922.0 + 50.0 * (ratio - 1.0)  # 922-947 MeV
            mu_Q = -50.0 - 30.0 * (ratio - 1.0)  # -50 to -65 MeV
        else:
            # High density: slower growth due to repulsion
            # Extrapolate from calculated values
            sigma = 30.0 + 25.0 * (ratio - 1.0)  # 30 + 25*(ratio-1)
            omega = 19.0 + 19.0 * (ratio - 1.0)  # Linear in omega
            mu_B = 922.0 + 80.0 * (ratio - 1.0)  # Strong increase
            mu_Q = -80.0 - 30.0 * (ratio - 1.5)  # More negative
            
            # Cap sigma to prevent m_eff < 0
            # m_eff = 939 - 7.5 * sigma > 0 => sigma < 125 MeV
            sigma = min(sigma, 110.0)
        
        return EOSGuess(
            sigma=sigma,
            omega=omega,
            rho=0.0,  # Small for symmetric/nearly symmetric matter
            phi=0.0,  # Zero without strange hadrons
            mu_B=mu_B,
            mu_Q=mu_Q,
            mu_S=0.0,
            mu_L=0.0
        )


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON USE CASES
# =============================================================================
def solve_beta_equilibrium(
    model: SFHoEOS,
    n_B: float,
    T: float,
    include_muons: bool = True,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for beta equilibrium at given density and temperature.
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        T: Temperature (MeV)
        include_muons: Include muons
        guess: Initial guess (uses default if None)
        verbose: Print result
        
    Returns:
        EOSResult
    """
    eos_input = EOSInput(n_B=n_B, T=T)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.BETA_EQ, guess,
        include_muons=include_muons
    )
    
    if verbose:
        print(f"\nBeta equilibrium: n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


def solve_fixed_YQ(
    model: SFHoEOS,
    n_B: float,
    Y_Q: float,
    T: float,
    include_muons: bool = True,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for fixed charge fraction.
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        Y_Q: Charge fraction n_Q/n_B
        T: Temperature (MeV)
        include_muons: Include muons
        guess: Initial guess
        verbose: Print result
        
    Returns:
        EOSResult
    """
    eos_input = EOSInput(n_B=n_B, T=T, Y_Q=Y_Q)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.FIXED_YQ, guess,
        include_muons=include_muons
    )
    
    if verbose:
        print(f"\nFixed Y_Q = {Y_Q}: n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


def solve_fixed_YQ_YS(
    model: SFHoEOS,
    n_B: float,
    Y_Q: float,
    Y_S: float,
    T: float,
    include_muons: bool = True,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for fixed charge and strangeness fractions.
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        Y_Q: Charge fraction n_Q/n_B
        Y_S: Strangeness fraction n_S/n_B
        T: Temperature (MeV)
        include_muons: Include muons
        guess: Initial guess
        verbose: Print result
        
    Returns:
        EOSResult
    """
    eos_input = EOSInput(n_B=n_B, T=T, Y_Q=Y_Q, Y_S=Y_S)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.FIXED_YQ_YS, guess,
        include_muons=include_muons
    )
    
    if verbose:
        print(f"\nFixed Y_Q = {Y_Q}, Y_S = {Y_S}: n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


def solve_trapped_neutrinos(
    model: SFHoEOS,
    n_B: float,
    Y_L: float,
    T: float,
    include_muons: bool = True,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for beta equilibrium with trapped neutrinos (fixed lepton fraction).
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        Y_L: Lepton fraction (n_e + n_νe)/n_B
        T: Temperature (MeV)
        include_muons: Include muons
        guess: Initial guess
        verbose: Print result
        
    Returns:
        EOSResult
    """
    eos_input = EOSInput(n_B=n_B, T=T, Y_L=Y_L)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
        guess.mu_L = 50.0  # Initial guess for lepton chemical potential
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.BETA_EQ_TRAPPED, guess,
        include_muons=include_muons
    )
    
    if verbose:
        print(f"\nTrapped neutrinos Y_L = {Y_L}: n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


def solve_fixed_YC_hadrons_only(
    model: SFHoEOS,
    n_B: float,
    Y_C: float,
    T: float,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for fixed hadronic charge fraction WITHOUT leptons/photons.
    
    This mode fixes the hadronic charge fraction Y_C = n_Q_had/n_B.
    No electrons, muons, or photons are included. Strangeness β-equilibrium
    (μ_S = 0) is assumed.
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        Y_C: Hadronic charge fraction n_Q_had/n_B
        T: Temperature (MeV)
        guess: Initial guess
        verbose: Print result
        
    Returns:
        EOSResult with Y_Q = Y_C (no leptons)
    """
    eos_input = EOSInput(n_B=n_B, T=T, Y_C=Y_C)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.FIXED_YC_HADRONS_ONLY, guess,
        include_muons=False  # No muons in this mode
    )
    
    if verbose:
        print(f"\nFixed Y_C = {Y_C} (hadrons only): n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


def solve_fixed_YC_neutral(
    model: SFHoEOS,
    n_B: float,
    Y_C: float,
    T: float,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for fixed hadronic charge fraction WITH charge neutrality.
    
    This mode fixes the hadronic charge fraction Y_C = n_Q_had/n_B and
    adds electrons (and photons) to maintain charge neutrality:
    Y_e = Y_C (so that n_Q_total = 0).
    
    No muons are included. Strangeness β-equilibrium (μ_S = 0) is assumed.
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        Y_C: Hadronic charge fraction n_Q_had/n_B
        T: Temperature (MeV)
        guess: Initial guess
        verbose: Print result
        
    Returns:
        EOSResult with Y_e = Y_C and n_Q = 0 (charge neutral)
    """
    eos_input = EOSInput(n_B=n_B, T=T, Y_C=Y_C)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.FIXED_YC_NEUTRAL, guess,
        include_muons=False  # No muons in this mode
    )
    
    if verbose:
        print(f"\nFixed Y_C = {Y_C} (charge neutral): n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


# =============================================================================
# TABLE GENERATION FUNCTIONS
# =============================================================================
def generate_beta_eq_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    T: float,
    include_muons: bool = True,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """
    Generate beta equilibrium EOS table.
    
    Args:
        model: SFHoEOS model
        n_B_values: Array of baryon densities
        T: Temperature (MeV)
        include_muons: Include muons
        initial_guess: Guess for first point
        print_level: Output verbosity
        print_first_n: Number of first points to print
        error_threshold: Error threshold for printing
        
    Returns:
        List of EOSResult
    """
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.BETA_EQ,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons)


def generate_fixed_YQ_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    Y_Q: float,
    T: float,
    include_muons: bool = True,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """Generate EOS table with fixed charge fraction."""
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.FIXED_YQ,
        Y_Q=Y_Q,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons)


def generate_fixed_YQ_YS_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    Y_Q: float,
    Y_S: float,
    T: float,
    include_muons: bool = True,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """Generate EOS table with fixed charge and strangeness fractions."""
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.FIXED_YQ_YS,
        Y_Q=Y_Q,
        Y_S=Y_S,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons)


def generate_trapped_neutrino_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    Y_L: float,
    T: float,
    include_muons: bool = True,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """Generate EOS table with trapped neutrinos (fixed lepton fraction)."""
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
        initial_guess.mu_L = 50.0
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.BETA_EQ_TRAPPED,
        Y_L=Y_L,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons)


def generate_fixed_YC_hadrons_only_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    Y_C: float,
    T: float,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """Generate EOS table with fixed hadronic charge, no leptons."""
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.FIXED_YC_HADRONS_ONLY,
        Y_C=Y_C,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons=False)


def generate_fixed_YC_neutral_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    Y_C: float,
    T: float,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """Generate EOS table with fixed hadronic charge and charge neutrality (Y_e = Y_C)."""
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.FIXED_YC_NEUTRAL,
        Y_C=Y_C,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons=False)


def solve_fixed_YC_YS(
    model: SFHoEOS,
    n_B: float,
    Y_C: float,
    Y_S: float,
    T: float,
    guess: Optional[EOSGuess] = None,
    verbose: bool = False
) -> EOSResult:
    """
    Solve for fixed hadronic charge AND strangeness fractions WITHOUT leptons.
    
    This mode fixes both:
    - Y_C = n_Q_had/n_B (hadronic charge fraction)  
    - Y_S = n_S/n_B (strangeness fraction)
    
    No leptons or photons are included. μ_S is determined self-consistently.
    
    Args:
        model: SFHoEOS model instance
        n_B: Baryon density (fm⁻³)
        Y_C: Hadronic charge fraction n_Q_had/n_B
        Y_S: Strangeness fraction n_S/n_B
        T: Temperature (MeV)
        guess: Initial guess
        verbose: Print result
        
    Returns:
        EOSResult with specified Y_C and Y_S (no leptons)
    """
    eos_input = EOSInput(n_B=n_B, T=T, Y_C=Y_C, Y_S=Y_S)
    
    if guess is None:
        guess = model.get_default_guess(n_B, T)
    
    result = solve_eos_point(
        model, eos_input, EquilibriumType.FIXED_YC_YS, guess,
        include_muons=False  # No muons in this mode
    )
    
    if verbose:
        print(f"\nFixed Y_C = {Y_C}, Y_S = {Y_S} (hadrons only): n_B = {n_B:.4e} fm⁻³, T = {T} MeV")
        print_eos_result(result, detailed=True)
    
    return result


def generate_fixed_YC_YS_table(
    model: SFHoEOS,
    n_B_values: np.ndarray,
    Y_C: float,
    Y_S: float,
    T: float,
    initial_guess: Optional[EOSGuess] = None,
    print_level: PrintLevel = PrintLevel.NONE,
    print_first_n: int = 5,
    error_threshold: float = 1e-6
) -> List[EOSResult]:
    """Generate EOS table with fixed hadronic charge AND strangeness (no leptons)."""
    if initial_guess is None:
        initial_guess = model.get_default_guess(n_B_values[0], T)
    
    config = TableConfig(
        n_B_values=n_B_values,
        T=T,
        eq_type=EquilibriumType.FIXED_YC_YS,
        Y_C=Y_C,
        Y_S=Y_S,
        print_level=print_level,
        print_first_n=print_first_n,
        error_threshold=error_threshold
    )
    
    return generate_eos_table(model, config, initial_guess, include_muons=False)


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("SFHo EOS Implementation")
    print("=" * 70)
    
    # Test 1: Beta equilibrium with nucleons only
    print("\n" + "=" * 70)
    print("TEST 1: Beta equilibrium, nucleons only")
    print("-" * 50)
    
    model_n = SFHoEOS(particle_content='nucleons', parametrization='sfho')
    print(f"Model: {model_n.name}")
    
    result = solve_beta_equilibrium(model_n, n_B=0.16, T=10.0, verbose=True)
    
    # Test 2: Beta equilibrium with hyperons
    print("\n" + "=" * 70)
    print("TEST 2: Beta equilibrium, nucleons + hyperons")
    print("-" * 50)
    
    model_ny = SFHoEOS(particle_content='nucleons_hyperons', parametrization='sfhoy')
    print(f"Model: {model_ny.name}")
    
    result = solve_beta_equilibrium(model_ny, n_B=0.32, T=10.0, verbose=True)
    
    # Test 3: Fixed Y_Q (total charge)
    print("\n" + "=" * 70)
    print("TEST 3: Fixed Y_Q = 0.3 (total charge fraction)")
    print("-" * 50)
    
    result = solve_fixed_YQ(model_ny, n_B=0.16, Y_Q=0.3, T=10.0, verbose=True)
    
    # Test 4: Fixed Y_C hadrons only (no leptons)
    print("\n" + "=" * 70)
    print("TEST 4: Fixed Y_C = 0.3 (hadrons only, no leptons)")
    print("-" * 50)
    
    result = solve_fixed_YC_hadrons_only(model_ny, n_B=0.16, Y_C=0.3, T=10.0, verbose=True)
    print(f"\nVerification: Y_Q = {result.Y_Q:.4f} (should equal Y_C = 0.3)")
    print(f"              n_e = {result.n_e:.4e} (should be 0)")
    
    # Test 5: Fixed Y_C with charge neutrality
    print("\n" + "=" * 70)
    print("TEST 5: Fixed Y_C = 0.3 (charge neutral, Y_e = Y_C)")
    print("-" * 50)
    
    result = solve_fixed_YC_neutral(model_ny, n_B=0.16, Y_C=0.3, T=10.0, verbose=True)
    print(f"\nVerification: Y_e = {result.Y_e:.4f} (should equal Y_C = 0.3)")
    print(f"              n_Q = {result.n_Q:.4e} (should be ~0, charge neutral)")
    
    # Test 6: Small table generation
    print("\n" + "=" * 70)
    print("TEST 6: Generate small beta equilibrium table")
    print("-" * 50)
    
    n_sat = 0.16
    n_B_values = np.array([0.5, 1.0, 1.5, 2.0]) * n_sat
    
    model_hd = SFHoEOS(particle_content='nucleons_hyperons_deltas', parametrization='sfho_hd')
    print(f"Model: {model_hd.name}")
    
    results = generate_beta_eq_table(
        model_hd, n_B_values, T=10.0,
        print_level=PrintLevel.FIRST_AND_ERRORS,
        print_first_n=10
    )
    
    print("\nTable summary:")
    print(f"{'n_B [fm^-3]':>12} {'P [MeV/fm³]':>14} {'Y_e':>10} {'converged':>10}")
    print("-" * 50)
    for r in results:
        status = "OK" if r.converged else "FAIL"
        print(f"{r.n_B:12.4e} {r.P_total:14.4e} {r.Y_e:10.4e} {status:>10}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")