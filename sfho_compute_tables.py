"""
sfho_compute_tables.py
=====================
User-friendly script for generating EOS tables with OPTIMIZED initial guesses.

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

# Import EOS modules
from sfho_eos import SFHoEOS
from general_eos_solver import (
    EOSResult, EOSInput, EOSGuess, TableConfig, PrintLevel, EquilibriumType,
    solve_eos_point
)


#==============================================================================
# SETTINGS DATACLASS
#==============================================================================
@dataclass
class TableSettings:
    """
    Configuration for EOS table generation (supports multi-dimensional grids).
    
    Speed optimizations:
    - Within n_B: uses quadratic extrapolation from previous 3 points
    - Across parameters: uses solution at same n_B from previous (T, Y_C, ...) as guess
    
    Custom parametrization:
        Use custom_params to pass a SFHoParams object directly. Example:
        
        from sfho_parameters import create_custom_parametrization
        
        my_params = create_custom_parametrization(
            U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
            name="MyCustom"
        )
        settings = TableSettings(
            custom_params=my_params,
            particle_content='nucleons_hyperons_deltas'
        )
    """
    # Model selection
    parametrization: str = 'sfho'        # Used if custom_params is None
    particle_content: str = 'nucleons'
    equilibrium: str = 'beta_eq'
    custom_params: Any = None            # SFHoParams object for custom parametrization
    
    # Grid definition
    n_B_values: np.ndarray = field(default_factory=lambda: np.logspace(-2, 0, 50) * 0.16)
    T_values: List[float] = field(default_factory=lambda: [10.0])
    
    # Constraint parameters - can be single values OR arrays for multidimensional tables
    Y_Q_values: Union[float, List[float], None] = None
    Y_S_values: Union[float, List[float], None] = None
    Y_L_values: Union[float, List[float], None] = None
    Y_C_values: Union[float, List[float], None] = None
    
    # Options
    include_muons: bool = True
    
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
        'mu_B', 'mu_C', 'mu_S', 'mu_Q', 'mu_L',
        'P_total', 'e_total', 's_total',
        'Y_C', 'Y_S', 'Y_Q', 'Y_L', 
        'converged'
    ])


def _to_list(val):
    """Convert value to list if not None."""
    if val is None:
        return [None]
    if isinstance(val, (list, np.ndarray)):
        return list(val)
    return [val]


def _result_to_guess(result: EOSResult) -> EOSGuess:
    """Convert EOSResult to EOSGuess for initial guess propagation."""
    return EOSGuess(
        sigma=result.sigma,
        omega=result.omega,
        rho=result.rho,
        phi=result.phi,
        mu_B=result.mu_B,
        mu_Q=result.mu_Q,
        mu_S=result.mu_S,
        mu_L=result.mu_L
    )


def _get_eq_type(eq_str: str) -> EquilibriumType:
    """Convert equilibrium string to EquilibriumType."""
    eq_map = {
        'beta_eq': EquilibriumType.BETA_EQ,
        'fixed_yq': EquilibriumType.FIXED_YQ,
        'fixed_yq_ys': EquilibriumType.FIXED_YQ_YS,
        'trapped_neutrinos': EquilibriumType.BETA_EQ_TRAPPED,
        'fixed_yc_hadrons_only': EquilibriumType.FIXED_YC_HADRONS_ONLY,
        'fixed_yc_neutral': EquilibriumType.FIXED_YC_NEUTRAL,
        'fixed_yc_ys': EquilibriumType.FIXED_YC_YS,
    }
    return eq_map[eq_str.lower()]


def _extrapolate_guess_from_array(
    previous_solutions: List[np.ndarray],
    previous_nB: np.ndarray,
    current_nB: float
) -> EOSGuess:
    """Extrapolate initial guess from previous solutions using quadratic interpolation."""
    n_prev = len(previous_solutions)
    
    if n_prev == 0:
        return EOSGuess()
    elif n_prev == 1:
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
        # Quadratic extrapolation using last 3 points (Lagrange)
        n1, n2, n3 = previous_nB[-3], previous_nB[-2], previous_nB[-1]
        s1, s2, s3 = previous_solutions[-3], previous_solutions[-2], previous_solutions[-1]
        
        denom1 = (n1 - n2) * (n1 - n3)
        denom2 = (n2 - n1) * (n2 - n3)  
        denom3 = (n3 - n1) * (n3 - n2)
        
        if abs(denom1) < 1e-30 or abs(denom2) < 1e-30 or abs(denom3) < 1e-30:
            sol = s3
        else:
            L1 = ((current_nB - n2) * (current_nB - n3)) / denom1
            L2 = ((current_nB - n1) * (current_nB - n3)) / denom2
            L3 = ((current_nB - n1) * (current_nB - n2)) / denom3
            sol = L1 * s1 + L2 * s2 + L3 * s3
    
    return EOSGuess(
        sigma=sol[0], omega=sol[1], rho=sol[2], phi=sol[3],
        mu_B=sol[4], mu_Q=sol[5], mu_S=sol[6], mu_L=sol[7]
    )


#==============================================================================
# OPTIMIZED TABLE GENERATOR
#==============================================================================
def compute_table(settings: TableSettings) -> Dict[Tuple, List[EOSResult]]:
    """
    Compute EOS table(s) with OPTIMIZED initial guesses.
    
    Speed optimizations:
    1. Within n_B sweep: quadratic extrapolation from last 3 points
    2. Across parameters: use solution at same n_B index from previous table
       (e.g., use T=10 solutions as guess for T=20)
    """
    # Create model - use custom_params if provided, else use parametrization name
    if settings.custom_params is not None:
        model = SFHoEOS(
            particle_content=settings.particle_content,
            params=settings.custom_params
        )
    else:
        model = SFHoEOS(
            particle_content=settings.particle_content,
            parametrization=settings.parametrization
        )
    
    eq_type_str = settings.equilibrium.lower()
    eq_type = _get_eq_type(eq_type_str)
    
    # Build parameter grid
    T_list = list(settings.T_values)
    Y_C_list = _to_list(settings.Y_C_values)
    Y_S_list = _to_list(settings.Y_S_values)
    Y_Q_list = _to_list(settings.Y_Q_values)
    Y_L_list = _to_list(settings.Y_L_values)
    
    # Determine grid structure
    if eq_type_str == 'beta_eq':
        grid_params = list(product(T_list))
        param_names = ['T']
    elif eq_type_str == 'fixed_yq':
        grid_params = list(product(T_list, Y_Q_list))
        param_names = ['T', 'Y_Q']
    elif eq_type_str == 'fixed_yq_ys':
        grid_params = list(product(T_list, Y_Q_list, Y_S_list))
        param_names = ['T', 'Y_Q', 'Y_S']
    elif eq_type_str == 'trapped_neutrinos':
        grid_params = list(product(T_list, Y_L_list))
        param_names = ['T', 'Y_L']
    elif eq_type_str == 'fixed_yc_hadrons_only':
        grid_params = list(product(T_list, Y_C_list))
        param_names = ['T', 'Y_C']
    elif eq_type_str == 'fixed_yc_neutral':
        grid_params = list(product(T_list, Y_C_list))
        param_names = ['T', 'Y_C']
    elif eq_type_str == 'fixed_yc_ys':
        grid_params = list(product(T_list, Y_C_list, Y_S_list))
        param_names = ['T', 'Y_C', 'Y_S']
    else:
        raise ValueError(f"Unknown equilibrium type: {settings.equilibrium}")
    
    n_B_arr = np.asarray(settings.n_B_values)
    n_points = len(n_B_arr)
    n_tables = len(grid_params)
    
    if settings.print_results:
        print("=" * 70)
        print("EOS TABLE GENERATION (OPTIMIZED)")
        print("=" * 70)
        print(f"\nModel: {model.name}")
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
    
    for idx, params in enumerate(grid_params):
        param_dict = dict(zip(param_names, params))
        T = param_dict.get('T')
        Y_C = param_dict.get('Y_C')
        Y_S = param_dict.get('Y_S')
        Y_Q = param_dict.get('Y_Q')
        Y_L = param_dict.get('Y_L')
        
        if settings.print_results:
            param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items() if v is not None)
            print("-" * 70)
            print(f"[{idx+1}/{n_tables}] Computing table for {param_str}...")
        
        start_time = time.time()
        results = []
        previous_solutions = []  # For within-table extrapolation
        
        for i, n_B in enumerate(n_B_arr):
            # Build EOSInput
            eos_input = EOSInput(n_B=n_B, T=T, Y_Q=Y_Q, Y_S=Y_S, Y_L=Y_L, Y_C=Y_C)
            
            # Determine initial guess (OPTIMIZED)
            # Priority: 1) Previous table at same n_B (if converged)
            #           2) Extrapolation within current table
            #           3) Model default
            
            guess = None
            
            # Priority 1: Within-table extrapolation (smoothest continuation)
            # This prevents jumps from propagating between Y_C slices
            if len(previous_solutions) > 0:
                guess = _extrapolate_guess_from_array(
                    previous_solutions, n_B_arr[:i], n_B
                )
            
            # Priority 2: Cross-parameter guess (from previous table, same n_B index)
            # Only used for first few points of new table
            if guess is None and previous_table_results is not None and i < len(previous_table_results):
                prev_result = previous_table_results[i]
                if prev_result.converged:
                    guess = _result_to_guess(prev_result)
            
            # Fallback to model default
            if guess is None:
                guess = model.get_default_guess(n_B, T)
            
            # Solve
            include_muons = settings.include_muons and eq_type not in [
                EquilibriumType.FIXED_YC_HADRONS_ONLY,
                EquilibriumType.FIXED_YC_NEUTRAL,
                EquilibriumType.FIXED_YC_YS
            ]
            
            result = solve_eos_point(
                model, eos_input, eq_type, guess,
                include_muons=include_muons
            )
            results.append(result)
            
            # Store for within-table extrapolation
            sol_array = np.array([
                result.sigma, result.omega, result.rho, result.phi,
                result.mu_B, result.mu_Q, result.mu_S, result.mu_L
            ])
            previous_solutions.append(sol_array)
            if len(previous_solutions) > 2:  # Changed from 3 to 2 for linear extrapolation
                previous_solutions.pop(0)
            
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
        all_results[params] = results
        previous_table_results = results  # Store for next table
        
        if settings.print_timing:
            n_converged = sum(1 for r in results if r.converged)
            # Build parameter string for display
            param_parts = []
            if Y_C is not None:
                param_parts.append(f"Y_C={Y_C:.2f}")
            if Y_S is not None:
                param_parts.append(f"Y_S={Y_S:.2f}")
            if Y_Q is not None:
                param_parts.append(f"Y_Q={Y_Q:.2f}")
            if Y_L is not None:
                param_parts.append(f"Y_L={Y_L:.2f}")
            param_parts.append(f"T={T:.1f}")
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


def save_results(all_results: Dict[Tuple, List[EOSResult]], 
                 settings: TableSettings,
                 param_names: List[str]):
    """Save results to file."""
    if settings.output_filename:
        filename = settings.output_filename
    else:
        # Auto-generate filename with all relevant info
        muon_tag = "with_muons" if settings.include_muons else "no_muons"
        filename = f"sfho_tables_output/eos_{settings.parametrization}_{settings.particle_content}_{settings.equilibrium}_{muon_tag}.dat"
    
    with open(filename, 'w') as f:
        muon_str = "with muons" if settings.include_muons else "electrons only (no muons)"
        f.write(f"# EOS Table: {settings.parametrization}, {settings.particle_content}, {muon_str}\n")
        f.write(f"# Equilibrium: {settings.equilibrium}\n")
        # Build column list with independent variables first
        # Order: n_B, [Y_C/Y_L/Y_Q/Y_S], T, then rest
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
        
        for params, results in all_results.items():
            param_dict = dict(zip(param_names, params))
            for r in results:
                if r.converged:
                    row = []
                    for col in all_columns:
                        if col in param_dict:
                            val = param_dict[col]
                        elif col == 'Y_C':
                            # Y_C = Y_Q + Y_e (hadronic charge fraction)
                            val = r.Y_Q + r.Y_e
                        elif col == 'mu_e':
                            # mu_e = mu_L - mu_Q (or -mu_Q if no neutrinos)
                            val = r.mu_L - r.mu_Q if r.mu_L != 0 else -r.mu_Q
                        elif col == 'mu_C':
                            # mu_C = -mu_Q (charge chemical potential)
                            val = -r.mu_Q
                        else:
                            val = getattr(r, col, 0.0)
                        if val is None:
                            val = 0.0
                        if isinstance(val, bool):
                            val = 1 if val else 0
                        row.append(f"{val:>14.6e}" if isinstance(val, float) else f"{val:>14}")
                    f.write(" ".join(row) + "\n")
    
    print(f"\nSaved to: {filename}")


def results_to_arrays(results: List[EOSResult]) -> Dict[str, np.ndarray]:
    """Convert list of EOSResult to dictionary of numpy arrays."""
    attrs = [
        'n_B', 'T', 'P_total', 'e_total', 's_total',
        'sigma', 'omega', 'rho', 'phi', 'mu_B', 'mu_Q', 'mu_S', 'mu_L',
        'Y_Q', 'Y_S', 'Y_L', 'Y_e', 'n_e', 'n_mu', 'error'
    ]
    arrays = {}
    for attr in attrs:
        arrays[attr] = np.array([getattr(r, attr, np.nan) for r in results if r.converged])
    arrays['converged'] = np.array([r.converged for r in results])
    
    # Compute derived quantities
    # Y_C = Y_Q + Y_e (hadronic charge fraction)
    arrays['Y_C'] = arrays['Y_Q'] + arrays['Y_e']
    # mu_e = -mu_Q (electron chemical potential, no trapped neutrinos)
    # In general: mu_e = mu_L - mu_Q for trapped neutrinos
    arrays['mu_e'] = np.where(arrays['mu_L'] != 0, 
                               arrays['mu_L'] - arrays['mu_Q'], 
                               -arrays['mu_Q'])
    # mu_C = -mu_Q (charge chemical potential, same as mu_e without neutrinos)
    arrays['mu_C'] = -arrays['mu_Q']
    
    return arrays


#==============================================================================
# CONFIGURATION (EDIT THIS SECTION)
#==============================================================================
settings = TableSettings(
    # ===================== MODEL =====================
    parametrization='sfho', # sfho, sfhoy, sfhoy_star, 2fam, 2fam_phi or create_custom_parametrization(Uln,UsN,UxN,yl,yS,yx,xsd)
    particle_content='nucleons', # nucleons, nucleons_hyperons, nucleons_hyperons_deltas, nucleons_hyperons_deltas_mesons
    include_muons=False,
    
    # ===================== EQUILIBRIUM =====================
    # Options: 'beta_eq', 'fixed_YQ', 'fixed_YC_hadrons_only', 
    #          'fixed_YC_neutral', 'fixed_YC_YS', etc.
    equilibrium='beta_eq',
    
    # ===================== GRID =====================
    n_B_values=np.linspace(0.1, 10, 300) * 0.1583,
    T_values=np.concatenate((np.array([0.1]), np.linspace(10, 100, 10))),
    
    # ===================== CONSTRAINTS =====================
    # Y_C_values=0.5,    # For fixed_YC modes
    # Y_S_values=0.0,    # For fixed_YC_YS or fixed_YQ_YS
    # Y_Q_values=0.3,    # For fixed_YQ modes
    #Y_C_values=np.linspace(0.5, 0, 6),  # Reversed: 0.5→0 for better convergence at low Y_C
    # ===================== OUTPUT =====================
    print_results=False, 
    print_first_n=1,
    print_errors=True,
    print_timing=True,
    save_to_file=True,
    output_filename=None,  # Auto-generate: eos_{param}_{particles}_{eq}_{muon_status}.dat
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
