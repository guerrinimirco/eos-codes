"""
sfho_compute_tables.py
=====================
User-friendly script for generating SFHo EOS tables with OPTIMIZED initial guesses.

Key speed optimization: Uses solutions from previous parameter values (T, Y_C)
as initial guesses for the next, dramatically reducing solver iterations.

Usage:
    1. Edit the CONFIGURATION section below
    2. Run: python sfho_compute_tables.py
    
    OR import and use programmatically:
    
    from sfho_compute_tables import compute_table, TableSettings
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
from itertools import product

# Import SFHo EOS modules
from sfho_eos import (
    SFHoEOSResult,
    solve_sfho_beta_eq,
    solve_sfho_fixed_yc,
    solve_sfho_fixed_yc_ys,
    solve_sfho_trapped_neutrinos,
    solve_sfho_isentropic_beta_eq,
    solve_sfho_isentropic_trapped,
    result_to_guess,
    get_default_guess_beta_eq,
    get_default_guess_fixed_yc,
    get_default_guess_fixed_yc_ys,
    get_default_guess_trapped,
    BARYONS_N, BARYONS_NY, BARYONS_NYD
)
from sfho_parameters import (
    SFHoParams, 
    get_sfho_nucleonic, 
    get_sfhoy_fortin,
    get_sfhoy_star_fortin,
    get_sfho_2fam_phi,
    get_sfho_2fam
)


#==============================================================================
# SETTINGS DATACLASS
#==============================================================================
@dataclass
class TableSettings:
    """
    Configuration for SFHo EOS table generation (supports multi-dimensional grids).
    
    Speed optimizations:
    - Within n_B: uses quadratic extrapolation from previous 3 points
    - Across parameters: uses solution at same n_B from previous (T, Y_C, ...) as guess
    
    Equilibrium types:
    - 'beta_eq': Beta equilibrium with charge neutrality
    - 'fixed_yc': Fixed charge fraction Y_C (hadrons only)
    - 'fixed_yc_ys': Fixed Y_C and Y_S (requires hyperons)
    - 'trapped_neutrinos': Trapped neutrinos with fixed Y_L
    
    Custom parametrization:
        Use custom_params to pass a SFHoParams object directly. Example:
        
        from sfho_parameters import create_custom_parametrization
        
        my_params = create_custom_parametrization(
            U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
            name="MyCustom"
        )
        settings = TableSettings(
            custom_params=my_params,
            particle_content='nucleons_hyperons'
        )
    """
    # Model selection
    parametrization: str = 'sfho'        # 'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi'
    particle_content: str = 'nucleons'   # 'nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas'
    equilibrium: str = 'beta_eq'         # 'beta_eq', 'fixed_yc', 'fixed_yc_ys', 'trapped_neutrinos',
                                         # 'isentropic_beta_eq', 'isentropic_trapped'
    custom_params: Any = None            # SFHoParams object for custom parametrization
    
    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.logspace(-2, 0, 50) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    S_values: List[float] = field(default_factory=lambda: [1.0])  # Entropy per baryon for isentropic
    
    # Constraint parameters - can be single values OR arrays for multidimensional tables
    Y_C_values: Union[float, List[float], None] = None
    Y_S_values: Union[float, List[float], None] = None
    Y_L_values: Union[float, List[float], None] = None
    
    # Options
    include_muons: bool = False
    include_photons: bool = True
    include_electrons: bool = False      # For fixed_yc modes: add electrons for charge neutrality
    include_thermal_neutrinos: bool = False  # Add thermal neutrinos with μ_ν=0
    include_pseudoscalar_mesons: bool = False
    
    # Output control
    print_results: bool = True
    print_first_n: int = 5
    print_errors: bool = True
    print_timing: bool = True
    
    # File output
    save_to_file: bool = False
    output_filename: Optional[str] = None
    output_columns: List[str] = field(default_factory=lambda: [
        'n_B', 'T', 
        'sigma', 'omega', 'rho', 'phi',
        'mu_B', 'mu_C', 'mu_S', 'mu_e', 'mu_nu',
        'P_total', 'e_total', 's_total',
        'Y_C', 'Y_S', 'Y_L', 
        'converged'
    ])


def _to_list(val):
    """Convert value to list if not None."""
    if val is None:
        return [None]
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    return [val]


def _get_params(settings: TableSettings) -> SFHoParams:
    """Get SFHoParams from settings."""
    if settings.custom_params is not None:
        return settings.custom_params
    
    param_map = {
        'sfho': get_sfho_nucleonic,
        'sfhoy': get_sfhoy_fortin,
        'sfhoy_star': get_sfhoy_star_fortin,
        '2fam_phi': get_sfho_2fam_phi,
        '2fam': get_sfho_2fam,
    }
    
    if settings.parametrization.lower() in param_map:
        return param_map[settings.parametrization.lower()]()
    else:
        raise ValueError(f"Unknown parametrization: {settings.parametrization}")


def _get_particles(settings: TableSettings) -> list:
    """Get particle list from settings."""
    particle_map = {
        'nucleons': BARYONS_N,
        'nucleons_hyperons': BARYONS_NY,
        'nucleons_hyperons_deltas': BARYONS_NYD,
    }
    
    if settings.particle_content.lower() in particle_map:
        return particle_map[settings.particle_content.lower()]
    else:
        raise ValueError(f"Unknown particle content: {settings.particle_content}")


def _result_to_guess_array(result: SFHoEOSResult, eq_type: str) -> np.ndarray:
    """Convert SFHoEOSResult to guess array."""
    return result_to_guess(result, eq_type)


def _get_guess_linear_extrapolation(
    previous_solutions: List[np.ndarray],
    previous_nB: np.ndarray,
    current_nB: float
) -> Optional[np.ndarray]:
    """Linear extrapolation from last 2 non-None points."""
    # Get last 2 non-None solutions with their indices
    valid = [(i, s) for i, s in enumerate(previous_solutions) if s is not None]
    if len(valid) < 2:
        return None
    
    idx1, s1 = valid[-2]
    idx2, s2 = valid[-1]
    n1, n2 = previous_nB[idx1], previous_nB[idx2]
    
    if abs(n2 - n1) > 1e-15:
        slope = (s2 - s1) / (n2 - n1)
        return s2 + slope * (current_nB - n2)
    return None


def _get_guess_previous(
    previous_solutions: List[np.ndarray]
) -> Optional[np.ndarray]:
    """Return the most recent non-None solution."""
    # Find last non-None solution
    for sol in reversed(previous_solutions):
        if sol is not None:
            return sol.copy()
    return None


def _get_guess_quadratic_extrapolation(
    previous_solutions: List[np.ndarray],
    previous_nB: np.ndarray,
    current_nB: float
) -> Optional[np.ndarray]:
    """Quadratic extrapolation from last 3 non-None points (Lagrange)."""
    # Get last 3 non-None solutions with their indices
    valid = [(i, s) for i, s in enumerate(previous_solutions) if s is not None]
    if len(valid) < 3:
        return None
    
    idx1, s1 = valid[-3]
    idx2, s2 = valid[-2]
    idx3, s3 = valid[-1]
    n1, n2, n3 = previous_nB[idx1], previous_nB[idx2], previous_nB[idx3]
    
    denom1 = (n1 - n2) * (n1 - n3)
    denom2 = (n2 - n1) * (n2 - n3)  
    denom3 = (n3 - n1) * (n3 - n2)
    
    if abs(denom1) < 1e-30 or abs(denom2) < 1e-30 or abs(denom3) < 1e-30:
        return None
    
    L1 = ((current_nB - n2) * (current_nB - n3)) / denom1
    L2 = ((current_nB - n1) * (current_nB - n3)) / denom2
    L3 = ((current_nB - n1) * (current_nB - n2)) / denom3
    
    return L1 * s1 + L2 * s2 + L3 * s3


def _try_guess_strategies(
    previous_solutions: List[np.ndarray],
    previous_nB: np.ndarray,
    current_nB: float
) -> Optional[np.ndarray]:
    """
    Try guess strategies in order:
    1. Linear extrapolation (best for smooth continuation)
    2. Previous n_B result (safe fallback)
    3. Quadratic extrapolation (for non-linear regions)
    """
    # Strategy 1: Linear extrapolation
    guess = _get_guess_linear_extrapolation(previous_solutions, previous_nB, current_nB)
    if guess is not None:
        return guess
    
    # Strategy 2: Previous result
    guess = _get_guess_previous(previous_solutions)
    if guess is not None:
        return guess
    
    # Strategy 3: Quadratic extrapolation
    guess = _get_guess_quadratic_extrapolation(previous_solutions, previous_nB, current_nB)
    if guess is not None:
        return guess
    
    return None


#==============================================================================
# OPTIMIZED TABLE GENERATOR
#==============================================================================
def compute_table(settings: TableSettings) -> Dict[Tuple, List[SFHoEOSResult]]:
    """
    Compute SFHo EOS table(s) with OPTIMIZED initial guesses.
    
    Speed optimizations:
    1. Within n_B sweep: quadratic extrapolation from last 3 points
    2. Across parameters: use solution at same n_B index from previous table
       (e.g., use T=10 solutions as guess for T=20)
    """
    params = _get_params(settings)
    particles = _get_particles(settings)
    eq_type_str = settings.equilibrium.lower()
    
    # Build parameter grid
    T_list = list(settings.T_values)
    S_list = list(settings.S_values)
    Y_C_list = _to_list(settings.Y_C_values)
    Y_S_list = _to_list(settings.Y_S_values)
    Y_L_list = _to_list(settings.Y_L_values)
    
    # Determine grid structure
    # Order: composition constraints (Y_C, Y_S, Y_L) outer, T/S inner
    # This way cross-parameter guesses come from previous T/S at same composition,
    # which is physically more sensible (T/S changes are more continuous)
    if eq_type_str == 'beta_eq':
        grid_params = list(product(T_list))
        param_names = ['T']
    elif eq_type_str == 'fixed_yc':
        grid_params = list(product(Y_C_list, T_list))
        param_names = ['Y_C', 'T']
    elif eq_type_str == 'fixed_yc_ys':
        grid_params = list(product(Y_C_list, Y_S_list, T_list))
        param_names = ['Y_C', 'Y_S', 'T']
    elif eq_type_str == 'trapped_neutrinos':
        grid_params = list(product(Y_L_list, T_list))
        param_names = ['Y_L', 'T']
    elif eq_type_str == 'isentropic_beta_eq':
        grid_params = list(product(S_list))
        param_names = ['S']
    elif eq_type_str == 'isentropic_trapped':
        grid_params = list(product(Y_L_list, S_list))
        param_names = ['Y_L', 'S']
    else:
        raise ValueError(f"Unknown equilibrium type: {settings.equilibrium}")
    
    n_B_arr = np.asarray(settings.n_B_values)
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print("=" * 70)
        print("SFHo EOS TABLE GENERATION (OPTIMIZED)")
        print("=" * 70)
        print(f"\nModel: {params.name if hasattr(params, 'name') else settings.parametrization}")
        print(f"Particles: {settings.particle_content}")
        print(f"Equilibrium: {settings.equilibrium}")
        print(f"\nDensity grid: {n_points} points")
        print(f"  n_B range: [{n_B_arr[0]:.4e}, {n_B_arr[-1]:.4e}] fm^-3")
        print(f"\nParameter grid: {n_tables} tables")
        print(f"  Parameters: {param_names}")
        print(f"\nTotal points: {n_points * n_tables}")
        print(f"\n[Using optimized cross-parameter guess propagation]")
        print()
    
    all_results = {}
    previous_table_results = None  # Store results from previous parameter combination
    total_start = time.time()
    
    for idx, grid_param in enumerate(grid_params):
        param_dict = dict(zip(param_names, grid_param))
        T = param_dict.get('T')
        S = param_dict.get('S')  # For isentropic modes
        Y_C = param_dict.get('Y_C')
        Y_S = param_dict.get('Y_S')
        Y_L = param_dict.get('Y_L')
        
        if settings.print_results:
            param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items() if v is not None)
            print("-" * 70)
            print(f"[{idx+1}/{n_tables}] Computing table for {param_str}...")
        
        start_time = time.time()
        results = []
        previous_solutions = []  # For within-table extrapolation
        previous_nB_values = []  # n_B values corresponding to previous_solutions
        
        for i, n_B in enumerate(n_B_arr):
            # Determine initial guess (OPTIMIZED)
            # Priority for n_B > first point:
            #   1. Linear extrapolation from previous n_B steps
            #   2. Previous n_B result
            #   3. Quadratic extrapolation
            # Fallback: Cross-parameter guess or default
            
            guess = None
            
            # Priority 1-3: Within-table strategies (linear, previous, quadratic)
            if len(previous_solutions) > 0:
                guess = _try_guess_strategies(
                    previous_solutions, np.array(previous_nB_values), n_B
                )
            
            # Fallback: Cross-parameter guess (from previous table, same n_B index)
            if guess is None and previous_table_results is not None and i < len(previous_table_results):
                prev_result = previous_table_results[i]
                if prev_result.converged:
                    guess = _result_to_guess_array(prev_result, eq_type_str)
            
            # Call appropriate solver
            if eq_type_str == 'beta_eq':
                default_guess = get_default_guess_beta_eq(n_B, T, params)
                if guess is None:
                    guess = default_guess
                result = solve_sfho_beta_eq(
                    n_B, T, params, particles,
                    include_photons=settings.include_photons,
                    include_muons=settings.include_muons,
                    include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                    initial_guess=guess
                )
                # Retry with default guess if first attempt failed
                if not result.converged and not np.allclose(guess, default_guess):
                    result = solve_sfho_beta_eq(
                        n_B, T, params, particles,
                        include_photons=settings.include_photons,
                        include_muons=settings.include_muons,
                        include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                        initial_guess=default_guess
                    )
            elif eq_type_str == 'fixed_yc':
                default_guess = get_default_guess_fixed_yc(n_B, Y_C, T, params)
                if guess is None:
                    guess = default_guess
                result = solve_sfho_fixed_yc(
                    n_B, Y_C, T, params, particles,
                    include_electrons=settings.include_electrons,
                    include_photons=settings.include_photons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                    initial_guess=guess
                )
                # Retry with default guess if first attempt failed
                if not result.converged and not np.allclose(guess, default_guess):
                    result = solve_sfho_fixed_yc(
                        n_B, Y_C, T, params, particles,
                        include_electrons=settings.include_electrons,
                        include_photons=settings.include_photons,
                        include_thermal_neutrinos=settings.include_thermal_neutrinos,
                        include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                        initial_guess=default_guess
                    )
            elif eq_type_str == 'fixed_yc_ys':
                default_guess = get_default_guess_fixed_yc_ys(n_B, Y_C, Y_S, T, params)
                if guess is None:
                    guess = default_guess
                result = solve_sfho_fixed_yc_ys(
                    n_B, Y_C, Y_S, T, params, particles,
                    include_electrons=settings.include_electrons,
                    include_photons=settings.include_photons,
                    include_thermal_neutrinos=settings.include_thermal_neutrinos,
                    include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                    initial_guess=guess
                )
                # Retry with default guess if first attempt failed
                if not result.converged and not np.allclose(guess, default_guess):
                    result = solve_sfho_fixed_yc_ys(
                        n_B, Y_C, Y_S, T, params, particles,
                        include_electrons=settings.include_electrons,
                        include_photons=settings.include_photons,
                        include_thermal_neutrinos=settings.include_thermal_neutrinos,
                        include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                        initial_guess=default_guess
                    )
            elif eq_type_str == 'trapped_neutrinos':
                default_guess = get_default_guess_trapped(n_B, Y_L, T, params)
                if guess is None:
                    guess = default_guess
                result = solve_sfho_trapped_neutrinos(
                    n_B, Y_L, T, params, particles,
                    include_photons=settings.include_photons,
                    include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                    initial_guess=guess
                )
                # Retry with default guess if first attempt failed
                if not result.converged and not np.allclose(guess, default_guess):
                    result = solve_sfho_trapped_neutrinos(
                        n_B, Y_L, T, params, particles,
                        include_photons=settings.include_photons,
                        include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                        initial_guess=default_guess
                    )
            elif eq_type_str == 'isentropic_beta_eq':
                # Isentropic beta equilibrium: T is unknown, S is fixed
                default_guess = np.append(get_default_guess_beta_eq(n_B, 10.0, params), 10.0)
                if guess is None:
                    guess = default_guess
                result = solve_sfho_isentropic_beta_eq(
                    n_B, S, params, particles,
                    include_photons=settings.include_photons,
                    include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                    initial_guess=guess
                )
                # Retry with default guess if first attempt failed
                if not result.converged and not np.allclose(guess, default_guess):
                    result = solve_sfho_isentropic_beta_eq(
                        n_B, S, params, particles,
                        include_photons=settings.include_photons,
                        include_pseudoscalar_mesons=settings.include_pseudoscalar_mesons,
                        initial_guess=default_guess
                    )
            elif eq_type_str == 'isentropic_trapped':
                # Isentropic trapped: T is unknown, S and Y_L are fixed
                default_guess = np.append(get_default_guess_trapped(n_B, Y_L, 10.0, params), 10.0)
                if guess is None:
                    guess = default_guess
                result = solve_sfho_isentropic_trapped(
                    n_B, S, Y_L, params, particles,
                    include_photons=settings.include_photons,
                    initial_guess=guess
                )
                # Retry with default guess if first attempt failed
                if not result.converged and not np.allclose(guess, default_guess):
                    result = solve_sfho_isentropic_trapped(
                        n_B, S, Y_L, params, particles,
                        include_photons=settings.include_photons,
                        initial_guess=default_guess
                    )
            
            results.append(result)
            
            # Store for within-table extrapolation
            if result.converged:
                sol_array = _result_to_guess_array(result, eq_type_str)
                previous_solutions.append(sol_array)
                previous_nB_values.append(n_B)
                if len(previous_solutions) > 3:
                    previous_solutions.pop(0)
                    previous_nB_values.pop(0)
            
            # Print if requested
            if settings.print_results:
                should_print = False
                if i < settings.print_first_n:
                    should_print = True
                elif settings.print_errors and not result.converged:
                    should_print = True
                
                if should_print:
                    status = "OK" if result.converged else "FAILED"
                    print(f"[{i:4d}] n_B={n_B:.4e} [{status}] err={result.error:.2e}")
        
        elapsed = time.time() - start_time
        all_results[grid_param] = results
        previous_table_results = results  # Store for next table
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            # Build parameter string for display
            param_parts = []
            if Y_C is not None:
                param_parts.append(f"Y_C={Y_C:.2f}")
            if Y_S is not None:
                param_parts.append(f"Y_S={Y_S:.2f}")
            if Y_L is not None:
                param_parts.append(f"Y_L={Y_L:.2f}")
            if T is not None:
                param_parts.append(f"T={T:.1f}")
            if S is not None:
                param_parts.append(f"S={S:.2f}")
            param_str = " ".join(param_parts)
            
            print(f"\n  Completed {param_str} in {elapsed:.2f} s ({elapsed*1000/n_points:.1f} ms/point)")
            print(f"  Converged: {n_converged}/{n_points} ({100*n_converged/n_points:.1f}%)")
    
    total_elapsed = time.time() - total_start
    
    if settings.print_timing:
        print("\n" + "=" * 70)
        print(f"Total time: {total_elapsed:.2f} s")
        print(f"Average: {total_elapsed*1000/(n_points * n_tables):.1f} ms/point")
    
    if settings.save_to_file:
        save_results(all_results, settings, param_names)
    
    return all_results


def save_results(all_results: Dict[Tuple, List[SFHoEOSResult]], 
                 settings: TableSettings,
                 param_names: List[str]):
    """Save results to file."""
    params = _get_params(settings)
    
    if settings.output_filename:
        filename = settings.output_filename
    else:
        # Auto-generate filename with all relevant info
        filename = f"sfho_tables_output/eos_{settings.parametrization}_{settings.particle_content}_{settings.equilibrium}.dat"
    
    import os
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write(f"# SFHo EOS Table: {settings.parametrization}, {settings.particle_content}\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")
        
        # Components included
        components = []
        if settings.include_photons:
            components.append("photons")
        if settings.include_electrons:
            components.append("electrons")
        if settings.include_thermal_neutrinos:
            components.append("thermal_neutrinos")
        if settings.include_pseudoscalar_mesons:
            components.append("pseudoscalar_mesons")
        f.write(f"# Components: {', '.join(components) if components else 'hadrons only'}\n")
        
        # Build column list with independent variables first
        all_columns = list(settings.output_columns)
        
        # Ensure n_B is first
        if 'n_B' in all_columns:
            all_columns.remove('n_B')
        # Ensure T is present
        if 'T' in all_columns:
            all_columns.remove('T')
            
        # Add independent variables at the beginning: n_B, constraint params, T
        ind_vars = ['n_B']
        for pname in param_names:
            if pname not in ['n_B', 'T'] and pname not in ind_vars:
                ind_vars.append(pname)
        ind_vars.append('T')
        
        all_columns = ind_vars + [c for c in all_columns if c not in ind_vars]
        
        f.write("# " + " ".join(f"{col:>14}" for col in all_columns) + "\n")
        
        for grid_param, results in all_results.items():
            param_dict = dict(zip(param_names, grid_param))
            for r in results:
                if r.converged:
                    row = []
                    for col in all_columns:
                        if col in param_dict:
                            val = param_dict[col]
                        elif col == 'Y_C':
                            val = r.Y_C
                        elif col == 'Y_S':
                            val = r.Y_S
                        elif col == 'Y_L':
                            val = getattr(r, 'Y_L', 0.0)
                        elif col == 'mu_e':
                            val = getattr(r, 'mu_e', 0.0)
                        elif col == 'mu_C':
                            val = r.mu_C
                        elif col == 'mu_nu':
                            val = getattr(r, 'mu_nu', 0.0)
                        else:
                            val = getattr(r, col, 0.0)
                        if val is None:
                            val = 0.0
                        if isinstance(val, bool):
                            val = 1 if val else 0
                        row.append(f"{val:>14.6e}" if isinstance(val, float) else f"{val:>14}")
                    f.write(" ".join(row) + "\n")
    
    print(f"\nSaved to: {filename}")


def results_to_arrays(results: List[SFHoEOSResult]) -> Dict[str, np.ndarray]:
    """Convert list of SFHoEOSResult to dictionary of numpy arrays."""
    attrs = [
        'n_B', 'T', 'P_total', 'e_total', 's_total', 'f_total',
        'sigma', 'omega', 'rho', 'phi', 'mu_B', 'mu_C', 'mu_S', 'mu_L', 'mu_e', 'mu_nu',
        'Y_C', 'Y_S', 'n_C', 'n_e', 'error'
    ]
    arrays = {}
    
    # Filter to converged only
    converged_results = [r for r in results if r.converged]
    
    for attr in attrs:
        vals = []
        for r in converged_results:
            val = getattr(r, attr, np.nan)
            vals.append(val if val is not None else np.nan)
        arrays[attr] = np.array(vals)
    
    arrays['converged'] = np.array([r.converged for r in results])
    
    return arrays


#==============================================================================
# CONFIGURATION (EDIT THIS SECTION)
#==============================================================================
settings = TableSettings(
    # ===================== MODEL =====================
    parametrization='2fam_phi',  # 'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi', 
    particle_content='nucleons',  # 'nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas'
    
    # ===================== EQUILIBRIUM =====================
    # Options: 'beta_eq', 'fixed_yc', 'fixed_yc_ys', 'trapped_neutrinos'
    equilibrium='beta_eq',
    
    # ===================== GRID =====================
    n_B_values=np.linspace(0.1, 10, 300) * 0.1583,
    T_values=[0,0.1],#np.concatenate((np.array([0.1]), np.arange(2.5, 100, 2.5))),
    S_values=np.arange(0.5, 3, 0.5),
    # ===================== CONSTRAINTS =====================
    Y_C_values=np.arange(0., .5, 0.05),     # For fixed_yc modes
    Y_S_values=np.linspace(0., 1, 11),     # For fixed_yc_ys mode
    Y_L_values=np.arange(0.1, 0.4, 0.05),     # For trapped_neutrinos mode
    
    # ===================== OPTIONS =====================
    include_photons=False,
    include_muons=False,
    include_electrons=True,
    include_thermal_neutrinos=False,
    include_pseudoscalar_mesons=False,
    
    # ===================== OUTPUT =====================
    print_results=True, 
    print_first_n=3,
    print_errors=True,
    print_timing=True,
    save_to_file=True,
    output_filename="testT0.dat",  # Auto-generate
)


#==============================================================================
# MAIN
#==============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SFHo EOS TABLE GENERATOR (OPTIMIZED)")
    print("=" * 70 + "\n")
    
    all_results = compute_table(settings)
    
    if len(all_results) == 1:
        key = list(all_results.keys())[0]
        data = results_to_arrays(all_results[key])
        print("\n" + "=" * 70)
        print("DONE!")
        print(f"  n_B: [{data['n_B'].min():.4e}, {data['n_B'].max():.4e}] fm^-3")
        print(f"  P:   [{data['P_total'].min():.4e}, {data['P_total'].max():.4e}] MeV/fm^3")
    else:
        print(f"\nGenerated {len(all_results)} tables")
    
    print("=" * 70 + "\n")
