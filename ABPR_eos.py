"""
ABPR_eos.py
===========
ABPR (Alford-Braby-Paris-Reddy) equation of state for CFL quark matter at T=0.

The ABPR model uses analytical expressions for pressure and density at T=0,
parametrized by:
- a4: Perturbative QCD factor (related to α_s via α = π/2 × (1 - a4))
- Δ: Pairing gap (MeV)
- B: Bag constant (MeV⁴)
- ms: Strange quark mass (MeV)

P(μ) = 3 a4 μ⁴/(4π²ℏc³) + 3(Δ² - ms²/4) μ²/(π²ℏc³) - B/ℏc³
n_B(μ) = [3 a4 μ³/π² + 6(Δ² - ms²/4) μ/π²] / (3ℏc³)

Units:
- Energy/mass/chemical potentials: MeV
- Densities: fm⁻³
- Pressure/energy density: MeV/fm³

References:
- M. Alford, M. Braby, M. Paris, S. Reddy, Astrophys. J. 629 (2005)
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from general_physics_constants import hc, hc3, PI, PI2


# =============================================================================
# PARAMETERS DATACLASS
# =============================================================================
@dataclass
class ABPRParams:
    """
    Parameters for the ABPR CFL quark EOS at T=0.
    
    Attributes:
        name: Parameter set identifier
        ms: Strange quark mass (MeV), typically ~100-150 MeV
        Delta: Pairing gap Δ (MeV), typically ~50-150 MeV
        a4: QCD factor (dimensionless), related to α via α = π/2 × (1 - a4)
            Typical range: 0.6 - 1.0 (a4=1 is free quarks)
        B4: Bag constant B^(1/4) (MeV), typical 145-180 MeV
    """
    name: str = "abpr_default"
    ms: float = 150.0        # MeV (strange quark mass)
    Delta: float = 80.0     # MeV (pairing gap)
    a4: float = 0.7          # dimensionless (QCD factor)
    B4: float = 135.0        # MeV (bag constant B^1/4)
    
    @property
    def B(self) -> float:
        """Bag constant B = (B^1/4)^4/hc^3 in MeV/fm^3."""
        return self.B4**4/hc3
    
    @property
    def alpha(self) -> float:
        """QCD coupling α_s = π/2 × (1 - a4)."""
        return PI / 2.0 * (1.0 - self.a4)
    
    

def get_abpr_default() -> ABPRParams:
    """Get default ABPR parameter set."""
    return ABPRParams(name="abpr_default")


def get_abpr_custom(
    ms: float = 150.0,
    Delta: float = 100.0,
    a4: float = 0.7,
    B4: float = 145.0,
    name: str = "abpr_custom"
) -> ABPRParams:
    """
    Create custom ABPR parameter set.
    
    Args:
        ms: Strange quark mass (MeV)
        Delta: Pairing gap (MeV)
        a4: QCD factor (dimensionless)
        B4: Bag constant B^(1/4) (MeV)
        name: Parameter set name
        
    Returns:
        ABPRParams with specified values
    """
    return ABPRParams(name=name, ms=ms, Delta=Delta, a4=a4, B4=B4)


# =============================================================================
# RESULT DATACLASS
# =============================================================================
@dataclass
class ABPREOSResult:
    """Complete result from ABPR EOS calculation at one point."""
    # Convergence info
    converged: bool = True
    error: float = 0.0
    
    # Input conditions
    n_B: float = 0.0        # Baryon density (fm⁻³)
    
    # Parameters used
    ms: float = 0.0         # Strange quark mass (MeV)
    Delta: float = 0.0      # Pairing gap (MeV)
    a4: float = 0.0         # QCD factor
    B4: float = 0.0         # Bag constant B^(1/4) (MeV)
    
    # Chemical potential
    mu: float = 0.0         # Common quark chemical potential μ (MeV)
    mu_B: float = 0.0       # Baryon chemical potential μ_B = 3μ (MeV)
    
    # Thermodynamic quantities
    P: float = 0.0          # Pressure (MeV/fm³)
    e: float = 0.0          # Energy density (MeV/fm³)
    f: float = 0.0          # Free energy density = e at T=0 (MeV/fm³)

    


# =============================================================================
# CORE ANALYTICAL FUNCTIONS (OPTIMIZED)
# =============================================================================
def pressure_abpr(mu: float, params: ABPRParams) -> float:
    """
    ABPR pressure at T=0.
    
    P(μ) = 3 a4 μ⁴/(4π²ℏc³) + 3(Δ² - ms²/4) μ²/(π²ℏc³) - B/ℏc³
    Args:
        mu: Quark chemical potential (MeV)
        params: ABPR parameters
        
    Returns:
        Pressure P (MeV/fm³)
    """
    a4 = params.a4
    Delta2 = params.Delta**2
    ms2 = params.ms**2
    B = params.B
    
    # Precompute powers
    mu2 = mu**2
    mu4 = mu**4
    
    coeff_mu4 = 3.0 * a4 / (4.0 * PI2 * hc3)
    coeff_mu2 = 3.0 * (Delta2 - ms2 / 4.0) / (PI2 * hc3)
    
    return coeff_mu4 * mu4 + coeff_mu2 * mu2 - B


def baryon_density_abpr(mu: float, params: ABPRParams) -> float:
    """
    ABPR baryon density at T=0.
    
    n_B(μ) = 3*n_q/3;  n_B = dP/dµ_B = dP/d(3μ) = dP/dμ/3 
           = [3 a4 μ³/π² + 6(Δ² - ms²/4) μ/π²] / (3ℏc³)
           = a4 μ³/(π²ℏc³) + 2(Δ² - ms²/4) μ/(π²ℏc³)
    
    Args:
        mu: Quark chemical potential (MeV)
        params: ABPR parameters
        
    Returns:
        Baryon density n_B (fm⁻³)
    """
    a4 = params.a4
    Delta2 = params.Delta**2
    ms2 = params.ms**2
    mu3 = mu**3
    
    coeff_mu3 = a4 / (PI2 * hc3)
    coeff_mu1 = 2.0 * (Delta2 - ms2 / 4.0) / (PI2 * hc3)
    
    return coeff_mu3 * mu3 + coeff_mu1 * mu


def energy_density_abpr(mu: float, params: ABPRParams) -> float:
    """
    ABPR energy density at T=0.
    
    From thermodynamic identity: ε = -P + μ_B × n_B = -P + 3 μ × n_B = -P + 3 μ × n_q
    
    Args:
        mu: Quark chemical potential (MeV)
        params: ABPR parameters
        
    Returns:
        Energy density ε (MeV/fm³)
    """
    P = pressure_abpr(mu, params)
    n_B = baryon_density_abpr(mu, params)
    return -P + 3.0 * mu * n_B


# =============================================================================
# CHEMICAL POTENTIAL INVERSION (using scipy.optimize.root)
# =============================================================================
from scipy.optimize import root


def mu_from_nB_abpr(
    n_B: float,
    params: ABPRParams,
    mu_guess: Optional[float] = None
) -> tuple:
    """
    Solve for μ given n_B using scipy.optimize.root.
    
    Args:
        n_B: Target baryon density (fm⁻³)
        params: ABPR parameters
        mu_guess: Optional initial guess for μ (MeV)
        
    Returns:
        (mu, converged)
    """
    # Initial guess
    if mu_guess is not None and mu_guess > 0:
        mu0 = mu_guess
    else:
        mu0 = (n_B * PI2 * hc3)**(1.0/3.0)
    
    def residual(mu):
        return baryon_density_abpr(mu[0], params) - n_B
    
    sol = root(residual, [mu0], method='hybr')
    mu = sol.x[0]
    return mu, sol.success


def mu_from_P_abpr(
    P_target: float,
    params: ABPRParams,
    mu_guess: Optional[float] = None
) -> tuple:
    """
    Solve for μ given pressure P using scipy.optimize.root.
    
    Args:
        P_target: Target pressure (MeV/fm³)
        params: ABPR parameters
        mu_guess: Optional initial guess for μ (MeV)
        
    Returns:
        (mu, converged)
    """
    B = params.B
    
    # Initial guess
    if mu_guess is not None and mu_guess > 0:
        mu0 = mu_guess
    else:
        mu0 =(max(P_target + B, 0.0) * 4.0 * PI2 * hc3 / (3.0 * params.a4))**(0.25)
    
    def residual(mu):
        return pressure_abpr(mu[0], params) - P_target
    
    sol = root(residual, [mu0], method='hybr')
    mu = sol.x[0]
    return mu, sol.success


def mu_from_epsilon_abpr(
    epsilon_target: float,
    params: ABPRParams,
    mu_guess: Optional[float] = None
) -> tuple:
    """
    Solve for μ given energy density ε using scipy.optimize.root.
    
    Args:
        epsilon_target: Target energy density (MeV/fm³)
        params: ABPR parameters
        mu_guess: Optional initial guess for μ (MeV)
        
    Returns:
        (mu, converged)
    """
    B = params.B
    
    # Initial guess
    if mu_guess is not None and mu_guess > 0:
        mu0 = mu_guess
    else:
        mu0 = max((max(epsilon_target - B, 0.0) * 4.0 * PI2 * hc3 / (9.0 * params.a4))**(0.25), 50.0)
    
    def residual(mu):
        return energy_density_abpr(mu[0], params) - epsilon_target
    
    sol = root(residual, [mu0], method='hybr')
    mu = sol.x[0]
    return mu, sol.success


# =============================================================================
# TABLE GENERATION
# =============================================================================
def generate_abpr_tables(
    input_type: str,
    input_values: np.ndarray,
    a4_list: list,
    B4_list: list,
    Delta_list: list,
    ms_list: list,
    output_dir: str = "./output",
    single_table: bool = True,
    filename_prefix: str = "abpr_eos"
) -> dict:
    """
    Generate ABPR EOS tables for multiple parameter sets.
    
    Args:
        input_type: 'mu', 'nB', 'P', or 'epsilon' - specifies input variable
        input_values: Array of input values (MeV for mu, fm⁻³ for nB, MeV/fm³ for P/epsilon)
        a4_list: List of a4 values to use
        B4_list: List of B^(1/4) values (MeV) to use
        Delta_list: List of Δ values (MeV) to use
        ms_list: List of strange quark mass values (MeV) to use
        output_dir: Directory for output files
        single_table: If True, output one combined table; if False, one table per parameter set
        filename_prefix: Prefix for output filenames
        
    Returns:
        Dictionary with tables (combined or per-parameter-set)
    """
    import os
    import time
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # Count total parameter combinations
    total_sets = len(a4_list) * len(B4_list) * len(Delta_list) * len(ms_list)
    current_set = 0
    total_start = time.time()
    
    print(f"Starting table generation for {total_sets} parameter set(s)...")
    print("-" * 70)
    
    # Loop over all parameter combinations
    for a4 in a4_list:
        for B4 in B4_list:
            for Delta in Delta_list:
                for ms in ms_list:
                    current_set += 1
                    set_start = time.time()
                    
                    print(f"[{current_set}/{total_sets}] a4={a4}, B4={B4}, Δ={Delta}, ms={ms}...", end=" ", flush=True)
                    
                    params = ABPRParams(
                        name=f"a4={a4}_B4={B4}_D={Delta}_ms={ms}",
                        ms=ms,
                        Delta=Delta,
                        a4=a4,
                        B4=B4
                    )
                    
                    n_points = len(input_values)
                    mu_arr = np.zeros(n_points)
                    nB_arr = np.zeros(n_points)
                    P_arr = np.zeros(n_points)
                    eps_arr = np.zeros(n_points)
                    converged_arr = np.zeros(n_points, dtype=bool)
                    
                    mu_prev = None
                    
                    for i, val in enumerate(input_values):
                        # Solve for mu based on input type
                        if input_type == 'mu':
                            mu = val
                            converged = True
                        elif input_type == 'nB':
                            mu, converged = mu_from_nB_abpr(val, params, mu_guess=mu_prev)
                        elif input_type == 'P':
                            mu, converged = mu_from_P_abpr(val, params, mu_guess=mu_prev)
                        elif input_type == 'epsilon':
                            mu, converged = mu_from_epsilon_abpr(val, params, mu_guess=mu_prev)
                        else:
                            raise ValueError(f"Unknown input_type: {input_type}. Use 'mu', 'nB', 'P', or 'epsilon'.")
                        
                        # Compute all thermodynamic quantities
                        mu_arr[i] = mu
                        P_arr[i] = pressure_abpr(mu, params)
                        nB_arr[i] = baryon_density_abpr(mu, params)
                        eps_arr[i] = energy_density_abpr(mu, params)
                        converged_arr[i] = converged
                        
                        # Use this mu as guess for next step
                        if converged:
                            mu_prev = mu
                    
                    set_elapsed = time.time() - set_start
                    n_converged = np.sum(converged_arr)
                    print(f"done in {set_elapsed*1000:.1f} ms ({n_converged}/{n_points} converged)")
                    
                    # Store results for this parameter set
                    result = {
                        'a4': a4,
                        'B4': B4,
                        'Delta': Delta,
                        'ms': ms,
                        'mu': mu_arr,
                        'nB': nB_arr,
                        'P': P_arr,
                        'epsilon': eps_arr,
                        'converged': converged_arr
                    }
                    all_results.append(result)
                    
                    # If separate tables, write immediately
                    if not single_table:
                        fname = f"{filename_prefix}_a4{a4}_B4{B4}_D{Delta}_ms{ms}.dat"
                        fpath = os.path.join(output_dir, fname)
                        _write_table(fpath, result)
                        print(f"    → Saved: {fpath}")
    
    # If single combined table, write all results
    if single_table:
        fname = f"{filename_prefix}_combined.dat"
        fpath = os.path.join(output_dir, fname)
        _write_combined_table(fpath, all_results)
        print(f"  Saved: {fpath}")
    
    total_elapsed = time.time() - total_start
    print("-" * 70)
    print(f"Total time: {total_elapsed:.2f} s for {total_sets} parameter set(s)")
    
    return {'results': all_results}


def _write_table(fpath: str, result: dict):
    """Write a single parameter set table to file."""
    with open(fpath, 'w') as f:
        # Header with parameters
        f.write(f"# ABPR EOS Table (CFL at T=0)\n")
        f.write(f"# a4 = {result['a4']}\n")
        f.write(f"# B4 = {result['B4']} MeV\n")
        f.write(f"# Delta = {result['Delta']} MeV\n")
        f.write(f"# ms = {result['ms']} MeV\n")
        f.write(f"# Columns: mu [MeV], nB [fm^-3], P [MeV/fm^3], epsilon [MeV/fm^3]\n")
        f.write(f"#{'mu':>14s} {'nB':>15s} {'P':>15s} {'epsilon':>15s}\n")
        
        for i in range(len(result['mu'])):
            f.write(f"{result['mu'][i]:15.8e} {result['nB'][i]:15.8e} "
                   f"{result['P'][i]:15.8e} {result['epsilon'][i]:15.8e}\n")


def _write_combined_table(fpath: str, all_results: list):
    """Write combined table with all parameter sets."""
    with open(fpath, 'w') as f:
        f.write(f"# ABPR EOS Combined Table (CFL at T=0)\n")
        f.write(f"# Columns: a4, B4 [MeV], Delta [MeV], mu [MeV], nB [fm^-3], P [MeV/fm^3], epsilon [MeV/fm^3]\n")
        f.write(f"#{'a4':>10s} {'B4':>12s} {'Delta':>12s} {'mu':>15s} {'nB':>15s} {'P':>15s} {'epsilon':>15s}\n")
        
        for result in all_results:
            for i in range(len(result['mu'])):
                f.write(f"{result['a4']:12.4f} {result['B4']:12.4f} {result['Delta']:12.4f} "
                       f"{result['mu'][i]:15.8e} {result['nB'][i]:15.8e} "
                       f"{result['P'][i]:15.8e} {result['epsilon'][i]:15.8e}\n")



# =============================================================================
# EXAMPLE: GENERATING ABPR EOS TABLES
# =============================================================================
# 
# USAGE:
# ------
# Run this script directly to generate EOS tables:
#   python ABPR_eos.py
#
# Or import and use the functions in your own code:
#   from ABPR_eos import generate_abpr_tables, pressure_abpr, baryon_density_abpr
#
# MAIN FUNCTIONS:
# ---------------
# - pressure_abpr(mu, params)         : P(μ) in MeV/fm³
# - baryon_density_abpr(mu, params)   : n_B(μ) in fm⁻³
# - energy_density_abpr(mu, params)   : ε(μ) in MeV/fm³
# - mu_from_nB_abpr(n_B, params)      : Solve μ from n_B
# - mu_from_P_abpr(P, params)         : Solve μ from P
# - mu_from_epsilon_abpr(eps, params) : Solve μ from ε
# - generate_abpr_tables(...)         : Generate EOS tables for parameter scan
#
# INPUT TYPES for generate_abpr_tables:
# --------------------------------------
# - 'mu'      : Input is quark chemical potential μ (MeV)
# - 'nB'      : Input is baryon density n_B (fm⁻³)
# - 'P'       : Input is pressure P (MeV/fm³)
# - 'epsilon' : Input is energy density ε (MeV/fm³)
#
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ABPR EOS Table Generator (CFL Quark Matter at T=0)")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Define parameter ranges
    # -------------------------------------------------------------------------
    # n0 = 0.16 fm⁻³ (nuclear saturation density)
    n0 = 0.16
    
    nB_values = np.linspace(1.0, 10.0, 300) * n0  # fm⁻³
    P_values = np.linspace(0, 400.0, 400)   # MeV/fm³
    
    input_values = P_values
    
    # Parameter lists to scan
    a4_list = [0.6, 0.7, 0.8, 0.9, 1.0]           # QCD factor (dimensionless)
    B4_list = [135,165]         # Bag constant B^(1/4) (MeV)
    Delta_list = [80,100,150,200]      # Pairing gap Δ (MeV)
    ms_list = [150]         # Strange quark mass (MeV)
    
    # -------------------------------------------------------------------------
    # Generate tables
    # -------------------------------------------------------------------------
    print("\nGenerating ABPR EOS tables...")
    print(f"  Input type: nB")
    print(f"  nB range: [{nB_values[0]/n0:.1f}, {nB_values[-1]/n0:.1f}] n0")
    print(f"  a4 values: {a4_list}")
    print(f"  B4 values: {B4_list} MeV")
    print(f"  Delta values: {Delta_list} MeV")
    print(f"  ms values: {ms_list} MeV")
    print()
    
    results = generate_abpr_tables(
        input_type='P',           # input type: nB, mu, P, epsilon
        input_values=input_values,    # Array of nB values
        a4_list=a4_list,
        B4_list=B4_list,
        Delta_list=Delta_list,
        ms_list=ms_list,
        output_dir="./output",
        single_table=False,        # Separate file for each parameter set
        filename_prefix="abpr_eos"
    )
    
    print("\n" + "=" * 70)
    print("✓ Table generation complete!")
    print("=" * 70)
