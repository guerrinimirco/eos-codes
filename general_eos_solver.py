"""
general_eos_solver.py
====================
Model-independent framework for equation of state calculations.

This module provides:
1. Abstract base class for EOS models
2. Single-point solver for various thermodynamic conditions
3. Table generator with adaptive initial guesses
4. Output/printing utilities

Supported conditions:
- Beta equilibrium (nB, T) with/without muons
- Fixed charge fraction Y_Q (nB, Y_Q, T)
- Fixed charge and strangeness fractions (nB, Y_Q, Y_S, T)
- Beta equilibrium + neutrino trapping (nB, Y_L, T)

Units:
- Energies/masses/chemical potentials: MeV
- Densities: fm⁻³
- Pressure/energy density: MeV/fm³
- Entropy density: fm⁻³
- Temperature: MeV

"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
from scipy.optimize import root
import warnings


# =============================================================================
# ENUMERATIONS
# =============================================================================
class EquilibriumType(Enum):
    """Type of thermodynamic equilibrium condition."""
    BETA_EQ = auto()              # Beta equilibrium: μ_S = 0, charge neutrality
    FIXED_YQ = auto()             # Fixed total charge fraction Y_Q = n_Q/n_B
    FIXED_YQ_YS = auto()          # Fixed Y_Q and Y_S = n_S/n_B
    BETA_EQ_TRAPPED = auto()      # Beta eq + neutrino trapping (fixed Y_L)
    FIXED_YC_HADRONS_ONLY = auto() # Fixed hadronic charge Y_C, no leptons/photons
    FIXED_YC_NEUTRAL = auto()      # Fixed hadronic charge Y_C, Y_e=Y_C (neutral), no muons
    FIXED_YC_YS = auto()           # Fixed hadronic charge Y_C and strangeness Y_S, no leptons


class ParticleContent(Enum):
    """Particle species included in the EOS."""
    NUCLEONS = auto()
    NUCLEONS_HYPERONS = auto()
    NUCLEONS_HYPERONS_DELTAS = auto()
    NUCLEONS_HYPERONS_DELTAS_MESONS = auto()


class PrintLevel(Enum):
    """Output verbosity for table calculations."""
    NONE = 0           # No output
    FIRST_ONLY = 1     # Print first few points
    FIRST_AND_ERRORS = 2  # Print first few + high error points


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class EOSInput:
    """
    Input parameters for EOS calculation.
    
    Attributes:
        n_B: Baryon number density (fm⁻³)
        T: Temperature (MeV)
        Y_Q: Charge fraction n_Q/n_B (only for FIXED_YQ modes)
        Y_S: Strangeness fraction n_S/n_B (only for FIXED_YQ_YS mode)
        Y_L: Lepton fraction (n_e + n_νe)/n_B (only for BETA_EQ_TRAPPED)
    """
    n_B: float
    T: float
    Y_Q: Optional[float] = None
    Y_S: Optional[float] = None
    Y_L: Optional[float] = None
    Y_C: Optional[float] = None  # Hadronic charge fraction (for FIXED_YC modes)


@dataclass
class EOSGuess:
    """
    Initial guess for the solver.
    
    Attributes:
        sigma: σ field (MeV)
        omega: ω field (MeV)
        rho: ρ field (MeV)
        phi: φ field (MeV)
        mu_B: Baryon chemical potential (MeV)
        mu_Q: Charge chemical potential (MeV)
        mu_S: Strangeness chemical potential (MeV), default 0 for beta eq
        mu_L: Lepton chemical potential (MeV), only for neutrino trapping
    """
    sigma: float = 50.0
    omega: float = 100.0
    rho: float = 0.0
    phi: float = 0.0
    mu_B: float = 939.0
    mu_Q: float = -50.0
    mu_S: float = 0.0
    mu_L: float = 0.0
    
    def to_array(self, eq_type: EquilibriumType) -> np.ndarray:
        """Convert to array for solver based on equilibrium type."""
        if eq_type in [EquilibriumType.BETA_EQ, EquilibriumType.FIXED_YQ,
                       EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_NEUTRAL]:
            # 6 unknowns: σ, ω, ρ, φ, μ_B, μ_Q (μ_S = 0 for strangeness β-eq)
            return np.array([self.sigma, self.omega, self.rho, self.phi,
                           self.mu_B, self.mu_Q])
        elif eq_type in [EquilibriumType.FIXED_YQ_YS, EquilibriumType.FIXED_YC_YS]:
            # 7 unknowns: σ, ω, ρ, φ, μ_B, μ_Q, μ_S
            return np.array([self.sigma, self.omega, self.rho, self.phi,
                           self.mu_B, self.mu_Q, self.mu_S])
        elif eq_type == EquilibriumType.BETA_EQ_TRAPPED:
            # 7 unknowns: σ, ω, ρ, φ, μ_B, μ_Q, μ_L (μ_S = 0)
            return np.array([self.sigma, self.omega, self.rho, self.phi,
                           self.mu_B, self.mu_Q, self.mu_L])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, eq_type: EquilibriumType) -> 'EOSGuess':
        """Create EOSGuess from solver array."""
        guess = cls()
        guess.sigma = arr[0]
        guess.omega = arr[1]
        guess.rho = arr[2]
        guess.phi = arr[3]
        guess.mu_B = arr[4]
        guess.mu_Q = arr[5]
        
        if eq_type in [EquilibriumType.BETA_EQ, EquilibriumType.FIXED_YQ,
                       EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_NEUTRAL]:
            guess.mu_S = 0.0
        elif eq_type in [EquilibriumType.FIXED_YQ_YS, EquilibriumType.FIXED_YC_YS]:
            guess.mu_S = arr[6]
        elif eq_type == EquilibriumType.BETA_EQ_TRAPPED:
            guess.mu_S = 0.0
            guess.mu_L = arr[6]
        
        return guess


@dataclass
class EOSResult:
    """
    Complete result from EOS calculation.
    
    Attributes:
        converged: Whether solver converged
        error: Maximum residual error
        
        # Meson fields (MeV)
        sigma, omega, rho, phi: float
        
        # Chemical potentials (MeV)
        mu_B, mu_Q, mu_S, mu_L: float
        
        # Conserved densities (fm⁻³)
        n_B, n_Q, n_S: float
        
        # Lepton densities (fm⁻³)
        n_e, n_mu, n_nu_e, n_nu_mu, n_nu_tau: float
        
        # Total thermodynamics (MeV/fm³ for P, e; fm⁻³ for s)
        P_total, e_total, s_total: float
        
        # Component contributions
        P_hadrons, P_leptons, P_photons, P_mesons: float
        
        # Fractions
        Y_Q, Y_S, Y_L, Y_e: float
        
        # Individual baryon densities
        baryon_densities: Dict[str, float]
        
        # Effective masses
        m_eff: Dict[str, float]
    """
    converged: bool = False
    error: float = 1e10
    
    # Fields
    sigma: float = 0.0
    omega: float = 0.0
    rho: float = 0.0
    phi: float = 0.0
    
    # Chemical potentials
    mu_B: float = 0.0
    mu_Q: float = 0.0
    mu_S: float = 0.0
    mu_L: float = 0.0
    
    # Densities
    n_B: float = 0.0
    n_Q: float = 0.0
    n_S: float = 0.0
    n_e: float = 0.0
    n_mu: float = 0.0
    n_nu_e: float = 0.0
    n_nu_mu: float = 0.0
    n_nu_tau: float = 0.0
    
    # Thermodynamics
    P_total: float = 0.0
    e_total: float = 0.0
    s_total: float = 0.0
    P_hadrons: float = 0.0
    P_leptons: float = 0.0
    P_photons: float = 0.0
    P_mesons: float = 0.0
    e_hadrons: float = 0.0
    e_leptons: float = 0.0
    e_photons: float = 0.0
    e_mesons: float = 0.0
    s_hadrons: float = 0.0
    s_leptons: float = 0.0
    s_photons: float = 0.0
    s_mesons: float = 0.0
    
    # Fractions
    Y_Q: float = 0.0
    Y_S: float = 0.0
    Y_L: float = 0.0
    Y_e: float = 0.0
    
    # Detailed info
    baryon_densities: Dict[str, float] = field(default_factory=dict)
    meson_densities: Dict[str, float] = field(default_factory=dict)
    m_eff: Dict[str, float] = field(default_factory=dict)


@dataclass 
class TableConfig:
    """Configuration for table generation."""
    n_B_values: np.ndarray  # Array of baryon densities
    T: float                # Temperature (fixed for table)
    eq_type: EquilibriumType
    Y_Q: Optional[float] = None
    Y_S: Optional[float] = None
    Y_L: Optional[float] = None
    Y_C: Optional[float] = None  # Hadronic charge fraction (FIXED_YC modes)
    
    # Printing options
    print_level: PrintLevel = PrintLevel.NONE
    print_first_n: int = 5
    error_threshold: float = 1e-6
    
    # Solver options
    max_iterations: int = 1000
    tolerance: float = 1e-10


# =============================================================================
# ABSTRACT BASE CLASS FOR EOS MODELS
# =============================================================================
class EOSModel(ABC):
    """
    Abstract base class for equation of state models.
    
    Subclasses must implement the model-specific methods for computing
    thermodynamics and field equations.
    """
    
    @abstractmethod
    def get_particle_content(self) -> ParticleContent:
        """Return the particle content of this model."""
        pass
    
    @abstractmethod
    def compute_hadron_thermo(
        self, T: float, mu_B: float, mu_Q: float, mu_S: float,
        sigma: float, omega: float, rho: float, phi: float
    ) -> Tuple[Dict, float, float, float, float, float, float, float, float, float, float]:
        """
        Compute hadronic thermodynamics.
        
        Returns:
            Tuple of (baryon_states, n_B, n_Q, n_S, P_baryons, e_baryons, s_baryons,
                     src_sigma, src_omega, src_rho, src_phi)
        """
        pass
    
    @abstractmethod
    def compute_meson_field_contribution(
        self, sigma: float, omega: float, rho: float, phi: float
    ) -> Tuple[float, float]:
        """
        Compute mean-field meson contribution to P and e.
        
        Returns:
            (P_meson_field, e_meson_field) in MeV/fm³
        """
        pass
    
    @abstractmethod
    def compute_field_residuals(
        self, sigma: float, omega: float, rho: float, phi: float,
        src_sigma: float, src_omega: float, src_rho: float, src_phi: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute field equation residuals.
        
        Returns:
            (res_sigma, res_omega, res_rho, res_phi)
        """
        pass
    
    @abstractmethod
    def compute_pseudoscalar_meson_thermo(
        self, T: float, mu_Q: float, mu_S: float,
        omega: float = 0.0, rho: float = 0.0
    ) -> Tuple[float, float, float, float, float, Dict]:
        """
        Compute pseudoscalar meson (π, K, η) thermodynamics.
        
        Returns:
            (n_Q_mesons, n_S_mesons, P_mesons, e_mesons, s_mesons, densities)
        """
        pass
    
    @abstractmethod
    def get_default_guess(self, n_B: float, T: float) -> EOSGuess:
        """Get default initial guess for given density and temperature."""
        pass
    
    def includes_pseudoscalar_mesons(self) -> bool:
        """Whether model includes thermal pseudoscalar mesons."""
        return self.get_particle_content() == ParticleContent.NUCLEONS_HYPERONS_DELTAS_MESONS


# =============================================================================
# LEPTON AND PHOTON THERMODYNAMICS (MODEL-INDEPENDENT)
# =============================================================================
def compute_lepton_photon_thermo(
    T: float,
    mu_Q: float,
    mu_L: float = 0.0,
    include_muons: bool = True,
    neutrino_trapped: bool = False
) -> Dict[str, Any]:
    """
    Compute lepton and photon thermodynamics.
    
    Chemical potential conventions:
    - Electron: μ_e = -μ_Q (beta equilibrium) or μ_e = μ_L - μ_Q (trapped)
    - Muon: μ_μ = μ_e (in chemical equilibrium with electrons)
    - Electron neutrino: μ_νe = 0 (escaped) or μ_νe = μ_L (trapped)
    - Muon/tau neutrino: μ_νμ = μ_ντ = 0 (thermal)
    
    Args:
        T: Temperature (MeV)
        mu_Q: Charge chemical potential (MeV)
        mu_L: Lepton chemical potential (MeV), only used if trapped
        include_muons: Include muons in calculation
        neutrino_trapped: If True, neutrinos are trapped with μ_νe = μ_L
        
    Returns:
        Dictionary with all lepton/photon thermodynamic quantities
    """
    from general_thermodynamics_leptons import (
        photon_thermo, electron_thermo, muon_thermo, neutrino_thermo
    )
    
    result = {
        'n_e': 0.0, 'n_mu': 0.0,
        'n_nu_e': 0.0, 'n_nu_mu': 0.0, 'n_nu_tau': 0.0,
        'P_leptons': 0.0, 'e_leptons': 0.0, 's_leptons': 0.0,
        'P_photons': 0.0, 'e_photons': 0.0, 's_photons': 0.0,
        'n_Q_leptons': 0.0, 'Y_L': 0.0
    }
    
    # Photons (always included)
    gamma = photon_thermo(T)
    result['P_photons'] = gamma.P
    result['e_photons'] = gamma.e
    result['s_photons'] = gamma.s
    
    # Electron chemical potential
    if neutrino_trapped:
        mu_e = mu_L - mu_Q
    else:
        mu_e = -mu_Q
    
    # Electrons
    e_thermo = electron_thermo(mu_e, T, include_antiparticles=True)
    result['n_e'] = e_thermo.n
    result['P_leptons'] += e_thermo.P
    result['e_leptons'] += e_thermo.e
    result['s_leptons'] += e_thermo.s
    result['n_Q_leptons'] += -e_thermo.n  # Q = -1 for electrons
    
    # Muons (same chemical potential as electrons in equilibrium)
    if include_muons:
        mu_thermo = muon_thermo(mu_e, T, include_antiparticles=True)
        result['n_mu'] = mu_thermo.n
        result['P_leptons'] += mu_thermo.P
        result['e_leptons'] += mu_thermo.e
        result['s_leptons'] += mu_thermo.s
        result['n_Q_leptons'] += -mu_thermo.n  # Q = -1 for muons
    
    # Neutrinos
    # Electron neutrino: trapped or thermal
    if neutrino_trapped:
        mu_nu_e = mu_L
    else:
        mu_nu_e = 0.0
    
    nu_e_thermo = neutrino_thermo(mu_nu_e, T, include_antiparticles=True)
    result['n_nu_e'] = nu_e_thermo.n
    result['P_leptons'] += nu_e_thermo.P
    result['e_leptons'] += nu_e_thermo.e
    result['s_leptons'] += nu_e_thermo.s
    
    # Muon neutrino: thermal (μ = 0)
    nu_mu_thermo = neutrino_thermo(0.0, T, include_antiparticles=True)
    result['n_nu_mu'] = nu_mu_thermo.n
    result['P_leptons'] += nu_mu_thermo.P
    result['e_leptons'] += nu_mu_thermo.e
    result['s_leptons'] += nu_mu_thermo.s
    
    # Tau neutrino: thermal (μ = 0)
    nu_tau_thermo = neutrino_thermo(0.0, T, include_antiparticles=True)
    result['n_nu_tau'] = nu_tau_thermo.n
    result['P_leptons'] += nu_tau_thermo.P
    result['e_leptons'] += nu_tau_thermo.e
    result['s_leptons'] += nu_tau_thermo.s
    
    return result


def _generate_alternative_guesses(n_B: float, T: float, Y_C: float = None, Y_S: float = None) -> List[EOSGuess]:
    """
    Generate a list of alternative initial guesses for difficult convergence cases.
    
    The guesses span a range of field values and chemical potentials to find
    one that leads to convergence.
    """
    n_sat = 0.158
    ratio = n_B / n_sat
    
    # For high strangeness, we need non-zero mu_S
    # Positive mu_S favors strange quarks (negative strangeness)
    mu_S_guess = 0.0
    phi_guess = 0.0
    if Y_S is not None and abs(Y_S) > 0.1:
        # For Y_S > 0 (positive strangeness = lots of strange baryons)
        # we need negative mu_S to favor s-quarks
        mu_S_guess = -100.0 * Y_S  # Rough scaling
        phi_guess = -10.0 * ratio * Y_S  # phi field couples to strangeness
    
    guesses = []
    
    # Guess 1: Conservative (smaller fields)
    guesses.append(EOSGuess(
        sigma=20.0 * ratio,
        omega=12.0 * ratio,
        rho=-5.0 * ratio if Y_C is not None and Y_C < 0.5 else 0.0,
        phi=phi_guess,
        mu_B=920.0 + 30.0 * ratio,
        mu_Q=-30.0 - 20.0 * (0.5 - (Y_C if Y_C else 0.3)),
        mu_S=mu_S_guess,
        mu_L=0.0
    ))
    
    # Guess 2: Larger fields (for higher density behavior)
    guesses.append(EOSGuess(
        sigma=35.0 * ratio,
        omega=22.0 * ratio,
        rho=-3.0 * ratio,
        phi=phi_guess * 1.5,
        mu_B=930.0 + 50.0 * ratio,
        mu_Q=-50.0,
        mu_S=mu_S_guess * 1.2,
        mu_L=0.0
    ))
    
    # Guess 3: Very small fields (for low density limit)
    guesses.append(EOSGuess(
        sigma=10.0 * ratio,
        omega=6.0 * ratio,
        rho=0.0,
        phi=phi_guess * 0.5,
        mu_B=938.0 + T,  # Near free nucleon plus thermal
        mu_Q=-20.0,
        mu_S=mu_S_guess * 0.5,
        mu_L=0.0
    ))
    
    # Guess 4: Saturation-like (good for n_B ~ n_sat)
    guesses.append(EOSGuess(
        sigma=30.0,
        omega=19.0,
        rho=-5.0 if Y_C is not None and Y_C < 0.4 else 0.0,
        phi=phi_guess,
        mu_B=922.0,
        mu_Q=-50.0 if Y_C is not None and Y_C < 0.4 else -1.0,
        mu_S=mu_S_guess,
        mu_L=0.0
    ))
    
    # Guess 5: High strangeness specific (for Y_S ~ 1)
    if Y_S is not None and Y_S > 0.5:
        # Strange matter dominated by Λ, Σ, Ξ hyperons
        # Larger sigma (more binding), significant phi field
        guesses.append(EOSGuess(
            sigma=50.0 * ratio,
            omega=30.0 * ratio,
            rho=-2.0 * ratio,
            phi=-20.0 * ratio,  # Strong phi field for strangeness
            mu_B=900.0 + 80.0 * ratio,
            mu_Q=-80.0 + 60.0 * (Y_C if Y_C else 0.0),
            mu_S=-150.0,  # Strong strangeness chemical potential
            mu_L=0.0
        ))
        
        # Guess 6: Alternative for extreme strangeness
        guesses.append(EOSGuess(
            sigma=60.0 * ratio,
            omega=35.0 * ratio,
            rho=-1.0 * ratio,
            phi=-30.0 * ratio,
            mu_B=880.0 + 100.0 * ratio,
            mu_Q=-60.0 + 40.0 * (Y_C if Y_C else 0.0),
            mu_S=-200.0,
            mu_L=0.0
        ))
        
        # Guess 7: Try different mu_S range
        guesses.append(EOSGuess(
            sigma=45.0 * ratio,
            omega=28.0 * ratio,
            rho=-3.0 * ratio,
            phi=-15.0 * ratio,
            mu_B=910.0 + 60.0 * ratio,
            mu_Q=-40.0,
            mu_S=-80.0,
            mu_L=0.0
        ))
    
    # Guess 8: For moderate strangeness (Y_S ~ 0.1-0.3) with Y_C ~ 0.5
    if Y_S is not None and 0.1 < Y_S < 0.4:
        guesses.append(EOSGuess(
            sigma=40.0 * ratio,
            omega=25.0 * ratio,
            rho=0.0,  # Symmetric matter
            phi=-8.0 * ratio,
            mu_B=930.0 + 40.0 * ratio,
            mu_Q=-5.0 if Y_C is not None and Y_C > 0.4 else -40.0,
            mu_S=-50.0,
            mu_L=0.0
        ))
        
        # Another variant for moderate strangeness
        guesses.append(EOSGuess(
            sigma=35.0 * ratio,
            omega=20.0 * ratio,
            rho=0.0,
            phi=-5.0 * ratio,
            mu_B=940.0 + 30.0 * ratio,
            mu_Q=-2.0,
            mu_S=-30.0,
            mu_L=0.0
        ))
    
    # Guess 9: For Y_C = 0.5 (symmetric) with any strangeness
    if Y_C is not None and Y_C > 0.45:
        guesses.append(EOSGuess(
            sigma=30.0 * ratio,
            omega=18.0 * ratio,
            rho=0.0,
            phi=-10.0 * ratio * (Y_S if Y_S else 0.0),
            mu_B=925.0 + 35.0 * ratio,
            mu_Q=-1.0,
            mu_S=-30.0 * (Y_S if Y_S else 0.0),
            mu_L=0.0
        ))
        
        # Additional guess for symmetric + moderate strangeness
        guesses.append(EOSGuess(
            sigma=45.0 * ratio / 2,  # ~ 45 for n_B=0.3
            omega=28.0 * ratio / 2,
            rho=0.0,
            phi=-6.0 * ratio / 2,
            mu_B=940.0,
            mu_Q=-1.0,
            mu_S=-30.0,
            mu_L=0.0
        ))
    
    # Guess 10-14: For low Y_C (0 < Y_C < 0.15) - very neutron-rich matter
    # These require precisely tuned mu_Q (very negative but not at boundary)
    if Y_C is not None and 0.0 < Y_C < 0.15:
        # The challenge: mu_Q must be negative enough to suppress protons
        # but not so negative that proton fraction goes to exactly 0
        # Scale mu_Q with density - higher density needs even more negative mu_Q
        
        # Guess 10: Moderate negative mu_Q with strong rho field
        guesses.append(EOSGuess(
            sigma=40.0 * ratio,
            omega=25.0 * ratio,
            rho=-12.0 * ratio,  # Strong rho for high isospin asymmetry
            phi=0.0,
            mu_B=920.0 + 40.0 * ratio,
            mu_Q=-100.0 - 30.0 * ratio,  # Scale with density
            mu_S=0.0,
            mu_L=0.0
        ))
        
        # Guess 11: Even more negative mu_Q for very high density
        guesses.append(EOSGuess(
            sigma=50.0 * ratio,
            omega=30.0 * ratio,
            rho=-15.0 * ratio,
            phi=0.0,
            mu_B=910.0 + 50.0 * ratio,
            mu_Q=-120.0 - 40.0 * ratio,
            mu_S=0.0,
            mu_L=0.0
        ))
        
        # Guess 12: Interpolate between Y_C~0.1 and Y_C~0 behavior
        # Use Y_C to scale mu_Q: smaller Y_C → more negative mu_Q
        mu_Q_scaled = -80.0 - 50.0 * (0.1 - Y_C) / 0.1 * ratio
        guesses.append(EOSGuess(
            sigma=35.0 * ratio,
            omega=22.0 * ratio,
            rho=-10.0 * ratio,
            phi=0.0,
            mu_B=925.0 + 35.0 * ratio,
            mu_Q=mu_Q_scaled,
            mu_S=0.0,
            mu_L=0.0
        ))
        
        # Guess 13: Try with smaller fields but very negative mu_Q
        guesses.append(EOSGuess(
            sigma=30.0 * ratio,
            omega=18.0 * ratio,
            rho=-8.0 * ratio,
            phi=0.0,
            mu_B=930.0 + 30.0 * ratio,
            mu_Q=-150.0 - 20.0 * ratio,
            mu_S=0.0,
            mu_L=0.0
        ))
        
        # Guess 14: High density specific - very large fields
        if ratio > 5:  # n_B > 5*n_sat
            guesses.append(EOSGuess(
                sigma=70.0 * ratio / 5,
                omega=45.0 * ratio / 5,
                rho=-20.0 * ratio / 5,
                phi=0.0,
                mu_B=900.0 + 80.0 * ratio / 5,
                mu_Q=-180.0 - 30.0 * ratio / 5,
                mu_S=0.0,
                mu_L=0.0
            ))
    
    return guesses


# =============================================================================
# SINGLE-POINT SOLVER
# =============================================================================
def solve_eos_point(
    model: EOSModel,
    eos_input: EOSInput,
    eq_type: EquilibriumType,
    guess: EOSGuess,
    include_muons: bool = True,
    tol: float = 1e-10,
    max_iter: int = 1000,
    retry_on_failure: bool = True
) -> EOSResult:
    """
    Solve EOS for a single thermodynamic point.
    
    If retry_on_failure is True and the initial guess doesn't converge,
    alternative guesses will be tried automatically.
    
    Args:
        model: EOS model implementing EOSModel interface
        eos_input: Input parameters (n_B, T, Y_Q, Y_S, Y_L)
        eq_type: Type of equilibrium condition
        guess: Initial guess for solver
        include_muons: Include muons in lepton sector
        tol: Solver tolerance
        max_iter: Maximum iterations
        retry_on_failure: Try alternative guesses if first fails
        
    Returns:
        EOSResult with all thermodynamic quantities
    """
    # Try with provided guess first
    result = _solve_eos_point_single(
        model, eos_input, eq_type, guess, include_muons, tol, max_iter
    )
    
    # If converged, return
    if result.converged:
        return result
    
    # If not converged and retry is enabled, try alternative guesses
    if retry_on_failure:
        Y_C = eos_input.Y_C if eos_input.Y_C is not None else eos_input.Y_Q
        Y_S = eos_input.Y_S
        alt_guesses = _generate_alternative_guesses(eos_input.n_B, eos_input.T, Y_C, Y_S)
        
        best_result = result
        best_error = result.error
        
        for alt_guess in alt_guesses:
            alt_result = _solve_eos_point_single(
                model, eos_input, eq_type, alt_guess, include_muons, tol, max_iter
            )
            
            if alt_result.converged:
                return alt_result
            
            # Keep track of best attempt even if not fully converged
            if alt_result.error < best_error:
                best_result = alt_result
                best_error = alt_result.error
        
        return best_result
    
    return result


def _solve_eos_point_single(
    model: EOSModel,
    eos_input: EOSInput,
    eq_type: EquilibriumType,
    guess: EOSGuess,
    include_muons: bool = True,
    tol: float = 1e-10,
    max_iter: int = 1000
) -> EOSResult:
    """Internal solver - single attempt with one guess."""
    n_B_target = eos_input.n_B
    T = eos_input.T
    include_ps_mesons = model.includes_pseudoscalar_mesons()
    neutrino_trapped = (eq_type == EquilibriumType.BETA_EQ_TRAPPED)
    
    def residual_function(x: np.ndarray) -> np.ndarray:
        """Compute residual vector for solver."""
        # Unpack variables based on equilibrium type
        sigma, omega, rho, phi = x[0], x[1], x[2], x[3]
        mu_B, mu_Q = x[4], x[5]
        
        if eq_type in [EquilibriumType.BETA_EQ, EquilibriumType.FIXED_YQ,
                       EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_NEUTRAL]:
            mu_S = 0.0
            mu_L = 0.0
        elif eq_type in [EquilibriumType.FIXED_YQ_YS, EquilibriumType.FIXED_YC_YS]:
            mu_S = x[6]
            mu_L = 0.0
        elif eq_type == EquilibriumType.BETA_EQ_TRAPPED:
            mu_S = 0.0
            mu_L = x[6]
        
        # Compute hadronic thermodynamics
        (baryon_states, n_B_had, n_Q_had, n_S_had, 
         P_had, e_had, s_had,
         src_sigma, src_omega, src_rho, src_phi) = model.compute_hadron_thermo(
            T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi
        )
        
        # Pseudoscalar meson contributions (if included)
        n_Q_ps = 0.0
        n_S_ps = 0.0
        if include_ps_mesons:
            n_Q_ps, n_S_ps, P_ps, e_ps, s_ps, _ = model.compute_pseudoscalar_meson_thermo(
                T, mu_Q, mu_S, omega, rho
            )
        
        # Lepton/photon contributions
        # For FIXED_YC_HADRONS_ONLY and FIXED_YC_YS: no leptons at all
        # For FIXED_YC_NEUTRAL: electrons + photons, no muons
        if eq_type in [EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_YS]:
            n_Q_lep = 0.0
        else:
            # For FIXED_YC_NEUTRAL, we don't include muons
            use_muons = include_muons and (eq_type != EquilibriumType.FIXED_YC_NEUTRAL)
            lep_result = compute_lepton_photon_thermo(
                T, mu_Q, mu_L, use_muons, neutrino_trapped
            )
            n_Q_lep = lep_result['n_Q_leptons']
        
        # Total conserved quantities
        n_B_total = n_B_had  # Only baryons contribute to n_B
        n_Q_total = n_Q_had + n_Q_ps + n_Q_lep
        n_S_total = n_S_had + n_S_ps
        
        # Field equation residuals
        res_sigma, res_omega, res_rho, res_phi = model.compute_field_residuals(
            sigma, omega, rho, phi,
            src_sigma, src_omega, src_rho, src_phi
        )
        
        # Normalize field residuals for better conditioning
        scale = 1e6  # Typical scale for field equations
        res_sigma /= scale
        res_omega /= scale
        res_rho /= scale
        res_phi /= scale
        
        # Build residual vector
        residuals = [res_sigma, res_omega, res_rho, res_phi]
        
        # Baryon number constraint
        residuals.append((n_B_total - n_B_target) / max(n_B_target, 1e-10))
        
        # Charge constraint
        if eq_type in [EquilibriumType.BETA_EQ, EquilibriumType.BETA_EQ_TRAPPED]:
            # Charge neutrality: n_Q = 0
            residuals.append(n_Q_total / max(n_B_target, 1e-10))
        elif eq_type in [EquilibriumType.FIXED_YQ, EquilibriumType.FIXED_YQ_YS]:
            # Fixed total Y_Q
            Y_Q_target = eos_input.Y_Q if eos_input.Y_Q is not None else 0.0
            residuals.append((n_Q_total - Y_Q_target * n_B_target) / max(n_B_target, 1e-10))
        elif eq_type in [EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_YS]:
            # Fixed hadronic charge Y_C, no leptons
            Y_C_target = eos_input.Y_C if eos_input.Y_C is not None else 0.0
            residuals.append((n_Q_had - Y_C_target * n_B_target) / max(n_B_target, 1e-10))
        elif eq_type == EquilibriumType.FIXED_YC_NEUTRAL:
            # Fixed hadronic charge Y_C, with charge neutrality (n_Q_total = 0)
            # This automatically gives Y_e = Y_C (since n_e = n_Q_had)
            Y_C_target = eos_input.Y_C if eos_input.Y_C is not None else 0.0
            residuals.append((n_Q_had - Y_C_target * n_B_target) / max(n_B_target, 1e-10))
        
        # Additional constraints
        if eq_type in [EquilibriumType.FIXED_YQ_YS, EquilibriumType.FIXED_YC_YS]:
            # Fixed Y_S
            Y_S_target = eos_input.Y_S if eos_input.Y_S is not None else 0.0
            residuals.append((n_S_total - Y_S_target * n_B_target) / max(n_B_target, 1e-10))
        elif eq_type == EquilibriumType.BETA_EQ_TRAPPED:
            # Fixed Y_L (lepton fraction)
            Y_L_target = eos_input.Y_L if eos_input.Y_L is not None else 0.4
            n_lep = lep_result['n_e'] + lep_result['n_nu_e']
            if include_muons:
                n_lep += lep_result['n_mu']
            Y_L_actual = n_lep / max(n_B_target, 1e-10)
            residuals.append(Y_L_actual - Y_L_target)
        
        return np.array(residuals)
    
    # Initial guess array
    x0 = guess.to_array(eq_type)
    
    # Solve
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = root(residual_function, x0, method='hybr',
                       options={'maxfev': max_iter, 'xtol': tol})
    
    x_sol = solution.x
    error = np.max(np.abs(solution.fun))
    
    # Convergence check: Either scipy reports success with small error,
    # OR the error is very small regardless of scipy status
    # (solver may hit max iterations but still find a good solution)
    converged = (solution.success and error < 1e-4) or (error < 1e-8)
    
    # Extract solution
    sigma, omega, rho, phi = x_sol[0], x_sol[1], x_sol[2], x_sol[3]
    mu_B, mu_Q = x_sol[4], x_sol[5]
    
    if eq_type in [EquilibriumType.BETA_EQ, EquilibriumType.FIXED_YQ,
                   EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_NEUTRAL]:
        mu_S = 0.0
        mu_L = 0.0
    elif eq_type in [EquilibriumType.FIXED_YQ_YS, EquilibriumType.FIXED_YC_YS]:
        mu_S = x_sol[6]
        mu_L = 0.0
    elif eq_type == EquilibriumType.BETA_EQ_TRAPPED:
        mu_S = 0.0
        mu_L = x_sol[6]
    
    # Compute final thermodynamics
    (baryon_states, n_B_had, n_Q_had, n_S_had,
     P_had, e_had, s_had,
     src_sigma, src_omega, src_rho, src_phi) = model.compute_hadron_thermo(
        T, mu_B, mu_Q, mu_S, sigma, omega, rho, phi
    )
    
    # Mean-field meson contribution
    P_mf, e_mf = model.compute_meson_field_contribution(sigma, omega, rho, phi)
    
    # Pseudoscalar mesons
    P_ps, e_ps, s_ps = 0.0, 0.0, 0.0
    meson_densities = {}
    n_Q_ps, n_S_ps = 0.0, 0.0
    if include_ps_mesons:
        n_Q_ps, n_S_ps, P_ps, e_ps, s_ps, meson_densities = model.compute_pseudoscalar_meson_thermo(
            T, mu_Q, mu_S, omega, rho
        )
    
    # Leptons and photons
    # For FIXED_YC_HADRONS_ONLY and FIXED_YC_YS: no leptons/photons at all
    # For FIXED_YC_NEUTRAL: set n_e = n_Q_had for exact charge neutrality, add photons
    if eq_type in [EquilibriumType.FIXED_YC_HADRONS_ONLY, EquilibriumType.FIXED_YC_YS]:
        lep_result = {
            'n_e': 0.0, 'n_mu': 0.0,
            'n_nu_e': 0.0, 'n_nu_mu': 0.0, 'n_nu_tau': 0.0,
            'P_leptons': 0.0, 'e_leptons': 0.0, 's_leptons': 0.0,
            'P_photons': 0.0, 'e_photons': 0.0, 's_photons': 0.0,
            'n_Q_leptons': 0.0
        }
    elif eq_type == EquilibriumType.FIXED_YC_NEUTRAL:
        # For FIXED_YC_NEUTRAL: set n_e = n_Q_had for exact charge neutrality
        # Electrons are prescribed, not determined by equilibrium
        from general_thermodynamics_leptons import photon_thermo, electron_thermo_from_density
        
        # Photons
        gamma = photon_thermo(T)
        
        # Electrons: set n_e = n_Q_had (including meson contributions)
        n_e_prescribed = n_Q_had + n_Q_ps
        e_result = electron_thermo_from_density(n_e_prescribed, T)
        
        lep_result = {
            'n_e': n_e_prescribed, 'n_mu': 0.0,
            'n_nu_e': 0.0, 'n_nu_mu': 0.0, 'n_nu_tau': 0.0,
            'P_leptons': e_result.P, 'e_leptons': e_result.e, 's_leptons': e_result.s,
            'P_photons': gamma.P, 'e_photons': gamma.e, 's_photons': gamma.s,
            'n_Q_leptons': -n_e_prescribed  # Electrons have Q = -1
        }
    else:
        # For other modes, compute from equilibrium thermodynamics
        use_muons = include_muons
        lep_result = compute_lepton_photon_thermo(
            T, mu_Q, mu_L, use_muons, neutrino_trapped
        )
    
    # Build result
    result = EOSResult(
        converged=converged,
        error=error,
        sigma=sigma, omega=omega, rho=rho, phi=phi,
        mu_B=mu_B, mu_Q=mu_Q, mu_S=mu_S, mu_L=mu_L,
        n_B=n_B_had,
        n_Q=n_Q_had + n_Q_ps + lep_result['n_Q_leptons'],
        n_S=n_S_had + n_S_ps,
        n_e=lep_result['n_e'],
        n_mu=lep_result['n_mu'],
        n_nu_e=lep_result['n_nu_e'],
        n_nu_mu=lep_result['n_nu_mu'],
        n_nu_tau=lep_result['n_nu_tau'],
        P_hadrons=P_had + P_mf,
        P_leptons=lep_result['P_leptons'],
        P_photons=lep_result['P_photons'],
        P_mesons=P_ps,
        e_hadrons=e_had + e_mf,
        e_leptons=lep_result['e_leptons'],
        e_photons=lep_result['e_photons'],
        e_mesons=e_ps,
        s_hadrons=s_had,
        s_leptons=lep_result['s_leptons'],
        s_photons=lep_result['s_photons'],
        s_mesons=s_ps,
        meson_densities=meson_densities
    )
    
    # Totals
    result.P_total = result.P_hadrons + result.P_leptons + result.P_photons + result.P_mesons
    result.e_total = result.e_hadrons + result.e_leptons + result.e_photons + e_ps
    result.s_total = result.s_hadrons + result.s_leptons + result.s_photons + result.s_mesons
    
    # Fractions
    if result.n_B > 0:
        result.Y_Q = result.n_Q / result.n_B
        result.Y_S = result.n_S / result.n_B
        result.Y_e = result.n_e / result.n_B
        n_lep = result.n_e + result.n_nu_e
        if include_muons:
            n_lep += result.n_mu
        result.Y_L = n_lep / result.n_B
    
    # Baryon densities and effective masses
    result.baryon_densities = {name: state.n for name, state in baryon_states.items()}
    result.m_eff = {name: state.m_eff for name, state in baryon_states.items()}
    
    return result


# =============================================================================
# TABLE GENERATOR
# =============================================================================
def generate_eos_table(
    model: EOSModel,
    config: TableConfig,
    initial_guess: EOSGuess,
    include_muons: bool = True
) -> List[EOSResult]:
    """
    Generate EOS table over a range of baryon densities.
    
    Uses adaptive initial guessing: each point uses the solution from
    the previous point (or extrapolation from previous 2-3 points).
    
    Args:
        model: EOS model
        config: Table configuration
        initial_guess: Guess for first point
        include_muons: Include muons
        
    Returns:
        List of EOSResult for each density point
    """
    results = []
    previous_solutions = []  # Store last few solutions for extrapolation
    
    n_points = len(config.n_B_values)
    
    for i, n_B in enumerate(config.n_B_values):
        # Create input
        eos_input = EOSInput(
            n_B=n_B,
            T=config.T,
            Y_Q=config.Y_Q,
            Y_S=config.Y_S,
            Y_L=config.Y_L,
            Y_C=config.Y_C
        )
        
        # Determine initial guess
        if i == 0:
            guess = initial_guess
        else:
            guess = _extrapolate_guess(
                previous_solutions, 
                config.n_B_values[:i],
                n_B,
                config.eq_type
            )
        
        # Solve
        result = solve_eos_point(
            model, eos_input, config.eq_type, guess,
            include_muons=include_muons,
            tol=config.tolerance,
            max_iter=config.max_iterations
        )
        
        results.append(result)
        
        # Store solution for extrapolation
        sol_array = np.array([
            result.sigma, result.omega, result.rho, result.phi,
            result.mu_B, result.mu_Q, result.mu_S, result.mu_L
        ])
        previous_solutions.append(sol_array)
        if len(previous_solutions) > 3:
            previous_solutions.pop(0)
        
        # Print output based on print level
        _print_point(i, n_B, result, config)
    
    return results


def _extrapolate_guess(
    previous_solutions: List[np.ndarray],
    previous_nB: np.ndarray,
    current_nB: float,
    eq_type: EquilibriumType
) -> EOSGuess:
    """
    Extrapolate initial guess from previous solutions.
    
    Uses linear extrapolation if 2+ points available,
    otherwise uses the last solution.
    """
    n_prev = len(previous_solutions)
    
    if n_prev == 0:
        return EOSGuess()
    elif n_prev == 1:
        # Use last solution
        sol = previous_solutions[-1]
    elif n_prev == 2:
        # Linear extrapolation
        n1, n2 = previous_nB[-2], previous_nB[-1]
        s1, s2 = previous_solutions[-2], previous_solutions[-1]
        if abs(n2 - n1) > 1e-15:
            slope = (s2 - s1) / (n2 - n1)
            sol = s2 + slope * (current_nB - n2)
        else:
            sol = s2
    else:
        # Quadratic extrapolation using last 3 points
        n1, n2, n3 = previous_nB[-3], previous_nB[-2], previous_nB[-1]
        s1, s2, s3 = previous_solutions[-3], previous_solutions[-2], previous_solutions[-1]
        
        # Fit quadratic: s = a*n^2 + b*n + c
        # Use Lagrange interpolation for simplicity
        denom1 = (n1 - n2) * (n1 - n3)
        denom2 = (n2 - n1) * (n2 - n3)
        denom3 = (n3 - n1) * (n3 - n2)
        
        # Avoid division by zero
        if abs(denom1) < 1e-30 or abs(denom2) < 1e-30 or abs(denom3) < 1e-30:
            sol = s3
        else:
            L1 = ((current_nB - n2) * (current_nB - n3)) / denom1
            L2 = ((current_nB - n1) * (current_nB - n3)) / denom2
            L3 = ((current_nB - n1) * (current_nB - n2)) / denom3
            sol = L1 * s1 + L2 * s2 + L3 * s3
    
    # Create EOSGuess from extrapolated values
    guess = EOSGuess(
        sigma=sol[0], omega=sol[1], rho=sol[2], phi=sol[3],
        mu_B=sol[4], mu_Q=sol[5], mu_S=sol[6], mu_L=sol[7]
    )
    
    return guess


def _print_point(
    index: int,
    n_B: float,
    result: EOSResult,
    config: TableConfig
):
    """Print point information based on print level."""
    if config.print_level == PrintLevel.NONE:
        return
    
    should_print = False
    reason = ""
    
    if config.print_level == PrintLevel.FIRST_ONLY:
        if index < config.print_first_n:
            should_print = True
            reason = "first"
    elif config.print_level == PrintLevel.FIRST_AND_ERRORS:
        if index < config.print_first_n:
            should_print = True
            reason = "first"
        elif result.error > config.error_threshold:
            should_print = True
            reason = "HIGH ERROR"
        elif not result.converged:
            should_print = True
            reason = "NOT CONVERGED"
    
    if should_print:
        status = "OK" if result.converged else "FAILED"
        print(f"[{index:4d}] n_B={n_B:.4e} ({reason}) [{status}]")
        print(f"       σ={result.sigma:.2f} ω={result.omega:.2f} "
              f"ρ={result.rho:.3f} φ={result.phi:.3f}")
        print(f"       μ_B={result.mu_B:.2f} μ_Q={result.mu_Q:.2f} "
              f"μ_S={result.mu_S:.2f} err={result.error:.2e}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def save_table_to_file(
    results: List[EOSResult],
    config: TableConfig,
    filename: str,
    model_name: str = "Unknown"
):
    """
    Save EOS table to file.
    
    Args:
        results: List of EOSResult
        config: Table configuration
        filename: Output filename
        model_name: Name of the model for header
    """
    with open(filename, 'w') as f:
        # Header
        f.write(f"# EOS Table: {model_name}\n")
        f.write(f"# Equilibrium: {config.eq_type.name}\n")
        f.write(f"# Temperature: {config.T} MeV\n")
        if config.Y_Q is not None:
            f.write(f"# Y_Q: {config.Y_Q}\n")
        if config.Y_S is not None:
            f.write(f"# Y_S: {config.Y_S}\n")
        if config.Y_L is not None:
            f.write(f"# Y_L: {config.Y_L}\n")
        f.write("#\n")
        
        # Column headers
        f.write("# Columns:\n")
        f.write("# 1:n_B[fm^-3] 2:P[MeV/fm^3] 3:e[MeV/fm^3] 4:s[fm^-3] "
                "5:mu_B[MeV] 6:mu_Q[MeV] 7:mu_S[MeV] 8:sigma[MeV] "
                "9:omega[MeV] 10:rho[MeV] 11:phi[MeV] 12:Y_e 13:Y_Q 14:error\n")
        
        # Data
        for r in results:
            f.write(f"{r.n_B:.6e} {r.P_total:.6e} {r.e_total:.6e} {r.s_total:.6e} "
                    f"{r.mu_B:.6f} {r.mu_Q:.6f} {r.mu_S:.6f} "
                    f"{r.sigma:.6f} {r.omega:.6f} {r.rho:.6f} {r.phi:.6f} "
                    f"{r.Y_e:.6e} {r.Y_Q:.6e} {r.error:.6e}\n")


def print_eos_result(result: EOSResult, detailed: bool = False):
    """Pretty print an EOS result."""
    status = "CONVERGED" if result.converged else "NOT CONVERGED"
    print(f"EOS Result [{status}] (error = {result.error:.2e})")
    print("-" * 60)
    
    print(f"Fields (MeV):")
    print(f"  σ = {result.sigma:.4f}")
    print(f"  ω = {result.omega:.4f}")
    print(f"  ρ = {result.rho:.4f}")
    print(f"  φ = {result.phi:.4f}")
    
    print(f"\nChemical potentials (MeV):")
    print(f"  μ_B = {result.mu_B:.4f}")
    print(f"  μ_Q = {result.mu_Q:.4f}")
    print(f"  μ_S = {result.mu_S:.4f}")
    if result.mu_L != 0:
        print(f"  μ_L = {result.mu_L:.4f}")
    
    print(f"\nDensities (fm⁻³):")
    print(f"  n_B = {result.n_B:.4e}")
    print(f"  n_Q = {result.n_Q:.4e}")
    print(f"  n_S = {result.n_S:.4e}")
    print(f"  n_e = {result.n_e:.4e}")
    if result.n_mu != 0:
        print(f"  n_μ = {result.n_mu:.4e}")
    
    print(f"\nThermodynamics:")
    print(f"  P_total = {result.P_total:.4e} MeV/fm³")
    print(f"  e_total = {result.e_total:.4e} MeV/fm³")
    print(f"  s_total = {result.s_total:.4e} fm⁻³")
    
    print(f"\nFractions:")
    print(f"  Y_e = {result.Y_e:.4e}")
    print(f"  Y_Q = {result.Y_Q:.4e}")
    print(f"  Y_S = {result.Y_S:.4e}")
    print(f"  Y_L = {result.Y_L:.4e}")
    
    if detailed and result.baryon_densities:
        print(f"\nBaryon densities (fm⁻³):")
        for name, n in sorted(result.baryon_densities.items()):
            if abs(n) > 1e-15:
                print(f"  {name:10s}: {n:.4e}")
    
    if detailed and result.m_eff:
        print(f"\nEffective masses (MeV):")
        for name, m in sorted(result.m_eff.items()):
            print(f"  {name:10s}: {m:.2f}")


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("EOS Solver Framework")
    print("=" * 60)
    print("This module provides the abstract framework for EOS calculations.")
    print("Use eos_sfho.py for SFHo-specific implementations.")
    print()
    print("Available equilibrium types:")
    for eq in EquilibriumType:
        print(f"  - {eq.name}")
    print()
    print("Available particle contents:")
    for pc in ParticleContent:
        print(f"  - {pc.name}")