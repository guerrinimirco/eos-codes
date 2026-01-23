"""
alphabag_compute_tables.py
==========================
User-friendly script for generating AlphaBag quark EOS tables.

Supports two phases:
- 'unpaired': Normal quark matter with perturbative α-corrections
- 'cfl': Color-Flavor Locked phase with pairing gap Δ

Equilibrium types for unpaired phase:
- 'beta_eq': Beta equilibrium with charge neutrality
- 'fixed_yc': Fixed charge fraction Y_C
- 'fixed_yc_ys': Fixed charge fraction Y_C and strangeness fraction Y_S

For CFL phase, flavor-locking constrains n_u = n_d = n_s.

Usage:
    from alphabag_compute_tables import compute_alphabag_table, AlphaBagTableSettings
    
    settings = AlphaBagTableSettings(
        phase='unpaired',
        equilibrium='beta_eq',
        T_values=[10.0, 30.0, 50.0],
        n_B_values=np.linspace(0.1, 10, 100) * 0.16
    )
    results = compute_alphabag_table(settings)
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union

from alphabag_parameters import AlphaBagParams, get_alphabag_default, get_alphabag_custom

from alphabag_eos import (
    AlphaBagEOSResult, CFLEOSResult,
    solve_alphabag_beta_eq, solve_alphabag_fixed_yc, solve_alphabag_fixed_yc_ys,
    solve_cfl
)
from alphabag_thermodynamics_quarks import CFLThermo


# =============================================================================
# SETTINGS DATACLASS
# =============================================================================
@dataclass
class AlphaBagTableSettings:
    """
    Configuration for AlphaBag EOS table generation.
    
    Phases:
    - 'unpaired': Normal quark matter (αBag)
    - 'cfl': Color-Flavor Locked quark matter
    
    Equilibrium types (unpaired only):
    - 'beta_eq': Beta equilibrium with charge neutrality
    - 'fixed_yc': Fixed charge fraction Y_C
    - 'fixed_yc_ys': Fixed Y_C and Y_S
    """
    # Model parameters (if None, uses defaults)
    params: Optional[AlphaBagParams] = None
    
    # Easy-access parameters (used if params is None)
    alpha: Optional[float] = None  # QCD coupling α_s
    B4: Optional[float] = None     # Bag constant B^(1/4) in MeV
    m_s: Optional[float] = None    # Strange quark mass in MeV
    
    # Phase selection
    phase: str = 'unpaired'  # 'unpaired' or 'cfl'
    
    # Equilibrium type (for unpaired phase)
    equilibrium: str = 'beta_eq'
    
    # CFL gap (for CFL phase)
    Delta0_values: List[float] = field(default_factory=lambda: [100.0])  # MeV
    
    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 10, 100) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    
    # Constraint parameters (for unpaired phase)
    Y_C_values: List[float] = field(default_factory=lambda: [0.0])
    Y_S_values: List[float] = field(default_factory=lambda: [0.0])
    
    # Options
    include_photons: bool = True
    include_gluons: bool = True
    include_electrons: bool = False
    include_thermal_neutrinos: bool = True  
    
    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True
    
    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None


# =============================================================================
# TABLE GENERATOR FOR UNPAIRED PHASE
# =============================================================================
def _compute_unpaired_table(settings: AlphaBagTableSettings) -> Dict[Tuple, List[AlphaBagEOSResult]]:
    """Compute table for unpaired αBag quark matter."""
    # Build params from settings or use default
    if settings.params is not None:
        params = settings.params
    elif settings.alpha is not None or settings.B4 is not None or settings.m_s is not None:
        params = get_alphabag_custom(
            alpha=settings.alpha or 0.3,
            B4=settings.B4 or 165.0,
            m_s=settings.m_s or 150.0
        )
    else:
        params = get_alphabag_default()
    
    eq_type = settings.equilibrium.lower()
    
    n_B_arr = np.asarray(settings.n_B_values)
    T_list = list(settings.T_values)
    
    # Build parameter grid
    if eq_type == 'beta_eq':
        grid_params = [(T,) for T in T_list]
        param_names = ['T']
    elif eq_type == 'fixed_yc':
        Y_C_list = list(settings.Y_C_values)
        grid_params = [(T, Y_C) for T in T_list for Y_C in Y_C_list]
        param_names = ['T', 'Y_C']
    elif eq_type == 'fixed_yc_ys':
        Y_C_list = list(settings.Y_C_values)
        Y_S_list = list(settings.Y_S_values)
        grid_params = [(T, Y_C, Y_S) for T in T_list for Y_C in Y_C_list for Y_S in Y_S_list]
        param_names = ['T', 'Y_C', 'Y_S']
    else:
        raise ValueError(f"Unknown equilibrium type: {eq_type}")
    
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print(f"\nPhase: UNPAIRED (αBag)")
        print(f"Parameters: B^1/4={params.B4} MeV, α_s={params.alpha}, m_s={params.m_s} MeV")
        print(f"Equilibrium: {eq_type}")
        print(f"Density grid: {n_points} points, n_B = [{n_B_arr[0]:.4e}, {n_B_arr[-1]:.4e}] fm⁻³")
        print(f"Parameter grid: {n_tables} tables\n")
    
    all_results = {}
    total_start = time.time()
    
    for idx, grid_param in enumerate(grid_params):
        T = grid_param[0]
        Y_C = grid_param[1] if len(grid_param) > 1 else None
        Y_S = grid_param[2] if len(grid_param) > 2 else None
        
        if settings.print_results:
            print("-" * 60)
            param_str = f"T={T} MeV"
            if Y_C is not None:
                param_str += f", Y_C={Y_C}"
            if Y_S is not None:
                param_str += f", Y_S={Y_S}"
            print(f"[{idx+1}/{n_tables}] {param_str}")
        
        start_time = time.time()
        results = []
        guess = None
        
        for i, n_B in enumerate(n_B_arr):
            # Call appropriate solver
            if eq_type == 'beta_eq':
                r = solve_alphabag_beta_eq(
                    n_B, T, params,
                    include_photons=settings.include_photons,
                    include_gluons=settings.include_gluons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    initial_guess=guess
                )
            elif eq_type == 'fixed_yc':
                r = solve_alphabag_fixed_yc(
                    n_B, Y_C, T, params,
                    include_photons=settings.include_photons,
                    include_gluons=settings.include_gluons,
                    include_electrons=settings.include_electrons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    initial_guess=guess
                )
            elif eq_type == 'fixed_yc_ys':
                r = solve_alphabag_fixed_yc_ys(
                    n_B, Y_C, Y_S, T, params,
                    include_photons=settings.include_photons,
                    include_gluons=settings.include_gluons,
                    include_electrons=settings.include_electrons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    initial_guess=guess
                )
            
            results.append(r)
            
            # Update guess for next point
            if r.converged:
                guess = np.array([r.mu_u, r.mu_d, r.mu_s, r.mu_e])[:3 if eq_type != 'beta_eq' else 4]
            
            # Print progress
            if settings.print_results:
                should_print = (i < settings.print_first_n or 
                               (settings.print_errors and not r.converged))
                if should_print:
                    status = "OK" if r.converged else "FAILED"
                    print(f"[{i:4d}] n_B={n_B:.4e} [{status}] P={r.P_total:.2f} Y_C={r.Y_C:.4f}")
        
        elapsed = time.time() - start_time
        all_results[grid_param] = results
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            print(f"\n  {elapsed:.2f}s, Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    return all_results


# =============================================================================
# TABLE GENERATOR FOR CFL PHASE
# =============================================================================
def _compute_cfl_table(settings: AlphaBagTableSettings) -> Dict[Tuple, List[CFLThermo]]:
    """Compute table for CFL quark matter."""
    # Build params from settings or use default
    if settings.params is not None:
        params = settings.params
    elif settings.alpha is not None or settings.B4 is not None or settings.m_s is not None:
        params = get_alphabag_custom(
            alpha=settings.alpha or 0.3,
            B4=settings.B4 or 165.0,
            m_s=settings.m_s or 150.0
        )
    else:
        params = get_alphabag_default()
    
    n_B_arr = np.asarray(settings.n_B_values)
    T_list = list(settings.T_values)
    Delta0_list = list(settings.Delta0_values)
    
    grid_params = [(T, Delta0) for T in T_list for Delta0 in Delta0_list]
    
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print(f"\nPhase: CFL (Color-Flavor Locked)")
        print(f"Parameters: B^1/4={params.B4} MeV, m_s={params.m_s} MeV")
        print(f"Gap values: Δ₀ = {Delta0_list} MeV")
        print(f"Density grid: {n_points} points, n_B = [{n_B_arr[0]:.4e}, {n_B_arr[-1]:.4e}] fm⁻³")
        print(f"Parameter grid: {n_tables} tables\n")
    
    all_results = {}
    total_start = time.time()
    
    for idx, (T, Delta0) in enumerate(grid_params):
        if settings.print_results:
            print("-" * 60)
            print(f"[{idx+1}/{n_tables}] T={T} MeV, Δ₀={Delta0} MeV")
        
        start_time = time.time()
        results = []
        guess = None
        
        for i, n_B in enumerate(n_B_arr):
            # CFL constraint: n_u = n_d = n_s = n_B (charge-neutral by construction)
            r = solve_cfl(
                n_B, T=T, Delta0=Delta0, params=params,
                include_photons=settings.include_photons,
                include_gluons=settings.include_gluons,
                initial_guess=guess
            )
            results.append(r)
            
            # Update guess for next point
            if r.converged:
                guess = np.array([r.mu_u, r.mu_d, r.mu_s])
            
            # Print progress
            if settings.print_results and i < settings.print_first_n:
                mu_avg = (r.mu_u + r.mu_d + r.mu_s) / 3.0
                print(f"[{i:4d}] n_B={n_B:.4e} μ={mu_avg:.2f} Δ={r.Delta:.2f} P={r.P_total:.2f}")
        
        elapsed = time.time() - start_time
        all_results[(T, Delta0)] = results
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            print(f"\n  {elapsed:.2f}s, Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    return all_results


# =============================================================================
# MAIN TABLE GENERATOR
# =============================================================================
def compute_alphabag_table(settings: AlphaBagTableSettings) -> Dict[Tuple, List]:
    """
    Compute AlphaBag EOS table.
    
    Dispatches to unpaired or CFL table generator based on settings.phase.
    
    Args:
        settings: AlphaBagTableSettings configuration
        
    Returns:
        Dictionary mapping parameter tuple to list of results
    """
    phase = settings.phase.lower()
    
    if settings.print_results:
        print("=" * 70)
        print("AlphaBag EOS TABLE GENERATOR")
        print("=" * 70)
    
    total_start = time.time()
    
    if phase == 'unpaired':
        all_results = _compute_unpaired_table(settings)
    elif phase == 'cfl':
        all_results = _compute_cfl_table(settings)
    else:
        raise ValueError(f"Unknown phase: {phase}. Use 'unpaired' or 'cfl'.")
    
    total_elapsed = time.time() - total_start
    
    if settings.print_timing:
        n_total = sum(len(v) for v in all_results.values())
        print("\n" + "=" * 70)
        print(f"Total: {total_elapsed:.2f}s, {n_total} points, Avg: {total_elapsed*1000/n_total:.1f}ms/pt")
    
    if settings.save_to_file:
        save_alphabag_results(all_results, settings, phase)
    
    return all_results


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_alphabag_results(
    all_results: Dict[Tuple, List],
    settings: AlphaBagTableSettings,
    phase: str
):
    """Save results to file."""
    params = settings.params if settings.params is not None else get_alphabag_default()
    
    if settings.output_filename:
        filename = settings.output_filename
    else:
        filename = f"/Users/mircoguerrini/Desktop/Research/Python_codes/output/alphabag_{phase}_B{int(params.B4)}_alpha{params.alpha}.dat"
    
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(f"# AlphaBag EOS Table ({phase})\n")
        f.write(f"# Parameters: B^1/4={params.B4} MeV, α_s={params.alpha}, m_s={params.m_s} MeV\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")
        
        # Write component flags
        components = []
        if settings.include_photons:
            components.append("photons")
        if settings.include_gluons:
            components.append("gluons")
        if settings.include_electrons:
            components.append("electrons")
        if settings.include_thermal_neutrinos:
            components.append("thermal_neutrinos")
        f.write(f"# Components: {', '.join(components) if components else 'quarks only'}\n")
        
        if phase == 'cfl':
            columns = ['n_B', 'T', 'Delta0', 'Delta', 'mu_u', 'mu_d', 'mu_s', 'P', 'e', 's', 'f']
        else:
            eq_type = settings.equilibrium.lower()
            if eq_type == 'beta_eq':
                columns = ['n_B', 'T', 'mu_u', 'mu_d', 'mu_s', 'mu_e', 'Y_u', 'Y_d', 'Y_s', 
                           'P_total', 'e_total', 's_total', 'converged']
            else:
                columns = ['n_B', 'Y_C', 'T', 'mu_u', 'mu_d', 'mu_s', 'Y_u', 'Y_d', 'Y_s',
                           'P_total', 'e_total', 's_total', 'converged']
        
        f.write("# " + " ".join(f"{col:>14}" for col in columns) + "\n")
        
        for params_tuple, results in all_results.items():
            for r in results:
                if phase == 'cfl':
                    row = [r.n_B, r.T, r.Delta0, r.Delta, r.mu_u, r.mu_d, r.mu_s, r.P_total, r.e_total, r.s_total, r.f_total]
                else:
                    if hasattr(r, 'converged') and not r.converged:
                        continue
                    if settings.equilibrium.lower() == 'beta_eq':
                        row = [r.n_B, r.T, r.mu_u, r.mu_d, r.mu_s, r.mu_e,
                               r.Y_u, r.Y_d, r.Y_s, r.P_total, r.e_total, r.s_total, 1]
                    else:
                        row = [r.n_B, r.Y_C, r.T, r.mu_u, r.mu_d, r.mu_s,
                               r.Y_u, r.Y_d, r.Y_s, r.P_total, r.e_total, r.s_total, 1]
                
                f.write(" ".join(f"{v:>14.6e}" if isinstance(v, float) else f"{v:>14}" for v in row) + "\n")
    
    print(f"\nSaved to: {filename}")


# =============================================================================
# HELPER: CONVERT TO ARRAYS
# =============================================================================
def results_to_arrays(results: List, phase: str = 'unpaired') -> Dict[str, np.ndarray]:
    """Convert list of results to dictionary of numpy arrays."""
    if phase == 'cfl':
        attrs = ['n_B', 'T', 'mu', 'Delta', 'Delta0', 'P', 'e', 's', 'f',
                 'n_u', 'n_d', 'n_s', 'Y_u', 'Y_d', 'Y_s']
        arrays = {}
        for attr in attrs:
            try:
                arrays[attr] = np.array([getattr(r, attr) for r in results])
            except AttributeError:
                pass
    else:
        # Filter converged only
        results = [r for r in results if r.converged]
        attrs = ['n_B', 'T', 'Y_C', 'Y_S', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
                 'Y_u', 'Y_d', 'Y_s', 'Y_e', 'P_total', 'e_total', 's_total']
        arrays = {}
        for attr in attrs:
            try:
                arrays[attr] = np.array([getattr(r, attr) for r in results])
            except AttributeError:
                pass
        arrays['converged'] = np.array([r.converged for r in results])
    
    return arrays


# =============================================================================
# CONFIGURATION (EDIT THIS SECTION)
# =============================================================================
settings = AlphaBagTableSettings(
    # Model parameters (easy access - if set, override defaults)
    alpha=0.3*np.pi/2,     # QCD coupling α_s
    B4=165.0,      # Bag constant B^(1/4) in MeV
    m_s=100.0,     # Strange quark mass in MeV
    
    # Phase: 'unpaired' or 'cfl'
    phase='cfl',
    
    # For unpaired phase: 'beta_eq', 'fixed_yc', 'fixed_yc_ys'
    equilibrium='beta',
    
    # For CFL phase: pairing gap values
    Delta0_values=[80.0],
    
    # Grid
    n_B_values=np.linspace(0.1, 12, 300) * 0.16,
    T_values=np.concatenate([[0.1],np.linspace(2.5, 120, 48)]),
    Y_C_values=[0.0,0.1,0.2,0.3,0.4,0.5],
    Y_S_values=[0.0, 0.1,0.2,0.4,0.6,0.8,1],
    
    # Options
    include_photons=True,
    include_gluons=True,
    include_electrons=True,
    include_thermal_neutrinos=True,
    
    # Output
    print_results=True,
    print_first_n=3,
    print_errors=True,
    print_timing=True,
    save_to_file=True,
)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    all_results = compute_alphabag_table(settings)
    print("\nDONE!")
