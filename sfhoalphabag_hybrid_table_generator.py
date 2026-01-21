#!/usr/bin/env python3
"""
sfhoalphabag_hybrid_table_generator.py
======================================
Hybrid EOS Table Generator Script for SFHo+AlphaBag mixed phase calculations.

This script orchestrates the entire computation workflow:
1. Initialize parameters
2. Compute pure H and Q tables
3. Compute phase boundaries for each η
4. Compute mixed phase in transition region for each η
5. Assemble unified EOS tables (H → Mixed → Q)
6. Save results with comprehensive thermodynamic quantities

Usage:
    python sfhoalphabag_hybrid_table_generator.py
"""

import numpy as np
import os
import time
from datetime import datetime
from typing import List, Dict

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# ===================== SFHo MODEL =====================
# Parametrization: 'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi', '2fam'
SFHO_PARAMETRIZATION = '2fam_phi'

# Particle content: 'nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas'
PARTICLE_CONTENT = 'nucleons_hyperons_deltas'

# ===================== ALPHABAG MODEL =====================
B4 = 180.0   # Bag constant B^{1/4} in MeV
alpha = 0.1 * np.pi / 2  # QCD coupling α_s

# ===================== DENSITY GRID =====================
n0 = 0.1583  # Nuclear saturation density
n_B_min = 0.1 * n0
n_B_max = 10.0 * n0
n_B_steps = 300

# ===================== TEMPERATURE GRID =====================
T_values = np.concatenate([[0.1], np.arange(2.5, 101, 2.5)])

# ===================== EQUILIBRIUM =====================
# Options: 'beta', 'fixed_yc'
EQUILIBRIUM_MODE = "fixed_yc"

# Surface tension parameter η: 0 = Gibbs, 1 = Maxwell
eta_values = [0.,1.0]

# Charge fractions (for fixed_yc mode)
Y_C_values = [0.,0.1,0.2,0.3,0.4,0.5]

# ===================== PHYSICS OPTIONS =====================
INCLUDE_PHOTONS = True
INCLUDE_THERMAL_NEUTRINOS = True
INCLUDE_GLUONS = True          # Gluons in Q phase 
INCLUDE_PSEUDOSCALAR_MESONS = True
INCLUDE_MUONS = False          # Muons in hadronic phase
INCLUDE_ELECTRONS = True       # Electrons (always needed for beta-eq)

# ===================== OUTPUT =====================
OUTPUT_DIR = "/Users/mircoguerrini/Desktop/Research/Python_codes/output/sfhoalphabag_hybrid_outputs"
BOUNDARY_DIR = os.path.join(OUTPUT_DIR, "boundaries")
VERBOSE = True

# ===================== RECOMPUTE OPTIONS =====================
FORCE_RECOMPUTE_PURE_PHASES = True
FORCE_RECOMPUTE_BOUNDARIES = True


# =============================================================================
# IMPORTS
# =============================================================================
from sfho_parameters import (
    get_sfho_nucleonic, 
    get_sfhoy_fortin,
    get_sfhoy_star_fortin,
    get_sfho_2fam_phi,
    get_sfho_2fam
)
from sfho_eos import (
    solve_sfho_beta_eq, solve_sfho_fixed_yc, SFHoEOSResult,
    BARYONS_N, BARYONS_NY, BARYONS_NYD
)
from sfho_compute_tables import TableSettings as SFHoTableSettings, compute_table as compute_sfho_table
from alphabag_parameters import AlphaBagParams
from alphabag_eos import solve_alphabag_beta_eq, solve_alphabag_fixed_yc, AlphaBagEOSResult
from alphabag_compute_tables import AlphaBagTableSettings, compute_alphabag_table
from sfhoalphabag_mixed_phase_eos import (
    solve_mixed_phase, MixedPhaseResult, PhaseBoundaryResult,
    find_phase_boundaries, find_all_boundaries, generate_unified_table, 
    solve_fixed_chi, result_to_guess
)


# =============================================================================
# PARAMETER HELPERS
# =============================================================================

def get_sfho_params():
    """Get SFHo parameters based on SFHO_PARAMETRIZATION."""
    param_map = {
        'sfho': get_sfho_nucleonic,
        'sfhoy': get_sfhoy_fortin,
        'sfhoy_star': get_sfhoy_star_fortin,
        '2fam_phi': get_sfho_2fam_phi,
        '2fam': get_sfho_2fam,
    }
    if SFHO_PARAMETRIZATION.lower() in param_map:
        return param_map[SFHO_PARAMETRIZATION.lower()]()
    raise ValueError(f"Unknown parametrization: {SFHO_PARAMETRIZATION}")


def get_particles():
    """Get particle list based on PARTICLE_CONTENT."""
    particle_map = {
        'nucleons': BARYONS_N,
        'nucleons_hyperons': BARYONS_NY,
        'nucleons_hyperons_deltas': BARYONS_NYD,
    }
    if PARTICLE_CONTENT.lower() in particle_map:
        return particle_map[PARTICLE_CONTENT.lower()]
    raise ValueError(f"Unknown particle content: {PARTICLE_CONTENT}")


# =============================================================================
# PURE PHASE TABLE CACHING
# =============================================================================

def get_pure_table_filename(phase: str, eq_mode: str, alphabag_params, output_dir: str) -> str:
    """Get filename for pure phase table cache file."""
    B4 = int(alphabag_params.B4)
    a = alphabag_params.alpha
    return os.path.join(output_dir, f"pure_{phase}_table_{eq_mode}_B{B4}_a{a:.4f}.pkl")


def save_pure_table(table: dict, phase: str, eq_mode: str, alphabag_params, output_dir: str, n_B_values: np.ndarray):
    """Save pure phase table to pickle file."""
    import pickle
    
    filename = get_pure_table_filename(phase, eq_mode, alphabag_params, output_dir)
    
    # Store table along with metadata for verification
    data = {
        'table': table,
        'n_B_values': n_B_values,
        'B4': alphabag_params.B4,
        'alpha': alphabag_params.alpha,
        'eq_mode': eq_mode
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"    Saved pure {phase} table: {os.path.basename(filename)}")


def load_pure_table(phase: str, eq_mode: str, alphabag_params, output_dir: str, n_B_values: np.ndarray) -> dict:
    """
    Load pure phase table from pickle file if exists and matches parameters.
    
    Returns None if file doesn't exist or parameters don't match.
    """
    import pickle
    
    filename = get_pure_table_filename(phase, eq_mode, alphabag_params, output_dir)
    
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Verify parameters match
        if (data['B4'] != alphabag_params.B4 or 
            data['alpha'] != alphabag_params.alpha or 
            data['eq_mode'] != eq_mode or
            len(data['n_B_values']) != len(n_B_values)):
            print(f"    Pure {phase} table parameters don't match, recomputing...")
            return None
        
        print(f"    Loaded pure {phase} table: {os.path.basename(filename)} ({len(data['table'])} points)")
        return data['table']
    
    except Exception as e:
        print(f"    Failed to load pure {phase} table: {e}")
        return None


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def results_to_dict(results: list, eta_override: float = None) -> dict:
    """Convert list of results to dict for saving."""
    if not results:
        return {}
    
    data = {
        'n_B': [], 'T': [], 'eta': [], 'converged': [], 'chi': [],
        'mu_B_H': [], 'mu_C_H': [], 'mu_B_Q': [], 'mu_C_Q': [],
        'P_H': [], 'P_Q': [], 'P_total': [],
        'e_total': [], 's_total': [], 'f_total': [],
        'n_B_H': [], 'n_C_H': [], 'n_B_Q': [], 'n_C_Q': [],
        'Y_C_tot': [],
    }
    
    for r in results:
        data['n_B'].append(r.n_B)
        data['T'].append(r.T)
        data['eta'].append(eta_override if eta_override is not None else getattr(r, 'eta', 0.0))
        data['converged'].append(1 if getattr(r, 'converged', True) else 0)
        
        # Determine result type
        is_mixed = hasattr(r, 'chi')
        is_pure_Q = hasattr(r, 'mu_u') and not hasattr(r, 'sigma_H')
        is_pure_H = hasattr(r, 'sigma') and not hasattr(r, 'mu_u_Q')
        
        if is_mixed:
            data['chi'].append(r.chi)
            data['mu_B_H'].append(r.mu_B_H)
            data['mu_C_H'].append(r.mu_C_H)
            data['mu_B_Q'].append(r.mu_B_Q)
            data['mu_C_Q'].append(r.mu_C_Q)
            data['P_H'].append(r.P_H)
            data['P_Q'].append(r.P_Q)
            data['P_total'].append(r.P_total)
            data['e_total'].append(r.e_total)
            data['s_total'].append(r.s_total)
            data['f_total'].append(r.f_total)
            data['n_B_H'].append(r.n_B_H)
            data['n_C_H'].append(r.n_C_H)
            data['n_B_Q'].append(r.n_B_Q)
            data['n_C_Q'].append(r.n_C_Q)
            data['Y_C_tot'].append(r.Y_C_tot)
        elif is_pure_H:
            data['chi'].append(0.0)
            data['mu_B_H'].append(getattr(r, 'mu_B', 0.0))
            data['mu_C_H'].append(getattr(r, 'mu_C', 0.0))
            data['mu_B_Q'].append(0.0)
            data['mu_C_Q'].append(0.0)
            data['P_H'].append(r.P_total)
            data['P_Q'].append(0.0)
            data['P_total'].append(r.P_total)
            data['e_total'].append(r.e_total)
            data['s_total'].append(r.s_total)
            f = r.e_total - r.T * r.s_total if r.T > 0 else r.e_total
            data['f_total'].append(f)
            data['n_B_H'].append(r.n_B)
            data['n_C_H'].append(getattr(r, 'n_C', 0.0))
            data['n_B_Q'].append(0.0)
            data['n_C_Q'].append(0.0)
            Y_C = getattr(r, 'Y_C', 0.0)
            data['Y_C_tot'].append(Y_C)
        elif is_pure_Q:
            data['chi'].append(1.0)
            data['mu_B_H'].append(0.0)
            data['mu_C_H'].append(0.0)
            data['mu_B_Q'].append(getattr(r, 'mu_B', 0.0))
            data['mu_C_Q'].append(getattr(r, 'mu_C', 0.0))
            data['P_H'].append(0.0)
            data['P_Q'].append(r.P_total)
            data['P_total'].append(r.P_total)
            data['e_total'].append(r.e_total)
            data['s_total'].append(r.s_total)
            f = r.e_total - r.T * r.s_total if r.T > 0 else r.e_total
            data['f_total'].append(f)
            data['n_B_H'].append(0.0)
            data['n_C_H'].append(0.0)
            data['n_B_Q'].append(r.n_B)
            Y_C = getattr(r, 'Y_C', 0.0)
            data['n_C_Q'].append(r.n_B * Y_C if r.n_B > 0 else 0.0)
            data['Y_C_tot'].append(Y_C)
    
    return {k: np.array(v) for k, v in data.items()}


def save_table(data: dict, filename: str, header: str = ""):
    """Save table data to .dat file."""
    if not data:
        return
    
    keys = list(data.keys())
    n_rows = len(data[keys[0]])
    
    with open(filename, 'w') as f:
        if header:
            for line in header.split('\n'):
                f.write(f"# {line}\n")
        f.write(f"# Columns: {len(keys)}, Rows: {n_rows}\n")
        f.write("#" + "".join(f"{k:>15}" for k in keys) + "\n")
        
        for i in range(n_rows):
            row = ""
            for k in keys:
                val = data[k][i]
                if isinstance(val, (int, np.integer)):
                    row += f"{val:>15d}"
                else:
                    row += f"{val:>15.6e}"
            f.write(row + "\n")
    
    print(f"  Saved: {filename} ({n_rows} rows)")


# =============================================================================
# PURE PHASE TABLE GENERATION
# =============================================================================

def compute_pure_H_table(n_B_values: np.ndarray, T_values: np.ndarray,
                         sfho_params, particles: list,
                         eq_mode: str = "beta", Y_C: float = None,
                         include_photons: bool = True,
                         include_muons: bool = False,
                         include_electrons: bool = False,
                         include_thermal_neutrinos: bool = False,
                         include_pseudoscalar_mesons: bool = False) -> Dict:
    """Compute pure hadronic phase table."""
    table = {}
    
    for T in T_values:
        for n_B in n_B_values:
            key = (n_B, T)
            
            if eq_mode == "beta":
                result = solve_sfho_beta_eq(
                    n_B, T, sfho_params, particles,
                    include_photons=include_photons,
                    include_muons=include_muons,
                    include_pseudoscalar_mesons=include_pseudoscalar_mesons
                )
            else:
                result = solve_sfho_fixed_yc(
                    n_B, Y_C, T, sfho_params, particles,
                    include_electrons=include_electrons,
                    include_photons=include_photons,
                    include_thermal_neutrinos=include_thermal_neutrinos,
                    include_pseudoscalar_mesons=include_pseudoscalar_mesons
                )
            
            table[key] = result
    
    return table


def compute_pure_Q_table(n_B_values: np.ndarray, T_values: np.ndarray,
                         alphabag_params, eq_mode: str = "beta",
                         Y_C: float = None,
                         include_photons: bool = True,
                         include_gluons: bool = True,
                         include_thermal_neutrinos: bool = True,
                         include_electrons: bool = False) -> Dict:
    """Compute pure quark phase table."""
    table = {}
    
    for T in T_values:
        for n_B in n_B_values:
            key = (n_B, T)
            
            if eq_mode == "beta":
                result = solve_alphabag_beta_eq(
                    n_B, T, alphabag_params,
                    include_photons=include_photons,
                    include_gluons=include_gluons,
                    include_thermal_neutrinos=include_thermal_neutrinos
                )
            else:
                result = solve_alphabag_fixed_yc(
                    n_B, Y_C, T, alphabag_params,
                    include_photons=include_photons,
                    include_gluons=include_gluons,
                    include_electrons=include_electrons
                )
            
            table[key] = result
    
    return table


# =============================================================================
# HELPER FUNCTIONS FOR SAVING
# =============================================================================


def _save_boundaries_file(all_boundaries: dict, Y_C: float, eta_values: list, alphabag_params):
    """Save phase boundaries to file."""
    filename = os.path.join(BOUNDARY_DIR, f"boundaries_YC{Y_C:.2f}_B{int(alphabag_params.B4)}_a{alphabag_params.alpha:.4f}.dat")
    
    with open(filename, 'w') as f:
        f.write(f"# Phase boundaries for Y_C={Y_C}, B^1/4={alphabag_params.B4}, α={alphabag_params.alpha}\n")
        f.write(f"# Columns: eta, T, n_B_onset, n_B_offset, converged\n")
        f.write("# " + "".join(f"{col:>15}" for col in ['eta', 'T', 'n_B_onset', 'n_B_offset', 'converged']) + "\n")
        
        for eta in eta_values:
            boundaries = all_boundaries.get(eta, [])
            for b in boundaries:
                f.write(f"{eta:>15.2f}{b.T:>15.2f}{b.n_B_onset:>15.6e}{b.n_B_offset:>15.6e}{1 if b.converged else 0:>15d}\n")
    
    print(f"    Saved: {os.path.basename(filename)}")


def _save_eos_tables(results: list, eta: float, output_dir: str, alphabag_params, Y_C: float = None):
    """Save EOS table to file."""
    if Y_C is not None:
        filename = os.path.join(output_dir, f"table_hybrid_eta{eta:.2f}_YC{Y_C:.2f}_B{int(alphabag_params.B4)}_a{alphabag_params.alpha:.4f}.dat")
        header = f"SFHo+AlphaBag hybrid EOS, eta={eta}, Y_C={Y_C}, B^1/4={alphabag_params.B4}, α={alphabag_params.alpha}"
    else:
        filename = os.path.join(output_dir, f"table_hybrid_eta{eta:.2f}_beta_B{int(alphabag_params.B4)}_a{alphabag_params.alpha:.4f}.dat")
        header = f"SFHo+AlphaBag hybrid EOS, eta={eta}, beta eq, B^1/4={alphabag_params.B4}, α={alphabag_params.alpha}"
    
    data = results_to_dict(results, eta_override=eta)
    save_table(data, filename, header)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_orchestrator():
    """Main orchestrator function."""
    
    print("=" * 70)
    print("SFHo + AlphaBag Hybrid EOS Orchestrator")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_start = time.time()
    
    # Initialize parameters
    print("\n" + "=" * 70)
    print("STEP 1: Initializing Parameters")
    print("=" * 70)
    
    sfho_params = get_sfho_params()
    particles = get_particles()
    alphabag_params = AlphaBagParams(
        name=f"AlphaBag_B{int(B4)}_a{alpha:.3f}",
        B4=B4,
        alpha=alpha
    )
    
    n_B_values = np.linspace(n_B_min, n_B_max, n_B_steps)
    T_array = np.array(T_values)
    
    print(f"  SFHo params: {sfho_params.name}")
    print(f"  Particles: {PARTICLE_CONTENT}")
    print(f"  AlphaBag params: B^1/4={B4} MeV, α={alpha:.4f}")
    print(f"  Density grid: [{n_B_min:.4f}, {n_B_max:.4f}] fm⁻³ ({n_B_steps} points)")
    print(f"  Temperature grid: {len(T_array)} values")
    print(f"  Equilibrium mode: {EQUILIBRIUM_MODE}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run based on mode
    if EQUILIBRIUM_MODE == "fixed_yc":
        _run_fixed_yc_mode(Y_C_values, T_array, n_B_values, eta_values,
                           sfho_params, alphabag_params, particles)
    else:
        _run_beta_mode(T_array, n_B_values, eta_values,
                       sfho_params, alphabag_params, particles)
    
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"COMPLETE - Total time: {total_elapsed:.1f}s")
    print("=" * 70)


def _run_fixed_yc_mode(Y_C_values, T_array, n_B_values, eta_values,
                       sfho_params, alphabag_params, particles):
    """
    Run orchestrator in fixed Y_C mode.
    
    For each Y_C value:
        1. Compute pure H and Q phase tables (with caching)
        2. Find phase boundaries for each η
        3. Generate unified tables (pure + mixed phases)
        4. Save results
    """
    all_results = {}
    
    for i_yc, Y_C in enumerate(Y_C_values):
        yc_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Processing Y_C = {Y_C} [{i_yc+1}/{len(Y_C_values)}]")
        print(f"{'='*70}")
        
        # ---------------------------------------------------------------------
        # STEP 2: COMPUTE PURE PHASE TABLES
        # ---------------------------------------------------------------------
        print(f"\n  [Step 2] Computing pure phase tables...")
        
        t_start = time.time()
        eq_mode_str = f"fixed_yc_YC{Y_C}"
        
        # --- Pure H Phase ---
        H_table = None
        if not FORCE_RECOMPUTE_PURE_PHASES:
            H_table = load_pure_table("H", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
            
        if H_table is None:
            # Compute pure H table using existing infrastructure
            print("    Computing pure H table...")
            sfho_settings = SFHoTableSettings(
                n_B_values=n_B_values,
                T_values=T_array,
                equilibrium='fixed_yc',
                Y_C_values=[Y_C],
                parametrization=SFHO_PARAMETRIZATION,
                particle_content=PARTICLE_CONTENT,
                include_photons=INCLUDE_PHOTONS,
                include_muons=INCLUDE_MUONS,
                include_electrons=INCLUDE_ELECTRONS,
                include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS,
                include_pseudoscalar_mesons=INCLUDE_PSEUDOSCALAR_MESONS,
                print_results=False,
                print_timing=VERBOSE,
                save_to_file=False
            )
            H_results_by_T = compute_sfho_table(sfho_settings)
            
            # Convert to (n_B, T) lookup format
            H_table = {}
            for grid_key, results in H_results_by_T.items():
                T = grid_key[1]  # (Y_C, T) tuple structure - T is at index 1
                for i, r in enumerate(results):
                    if r.converged:
                        H_table[(n_B_values[i], T)] = r
            
            # Save to cache
            save_pure_table(H_table, "H", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
        
        # --- Pure Q Phase ---
        Q_table = None
        if not FORCE_RECOMPUTE_PURE_PHASES:
            Q_table = load_pure_table("Q", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
            
        if Q_table is None:
            # Compute pure Q table using existing infrastructure
            print("    Computing pure Q table...")
            alphabag_settings = AlphaBagTableSettings(
                params=alphabag_params,
                phase='unpaired',
                equilibrium='fixed_yc',
                n_B_values=n_B_values,
                T_values=T_array,
                Y_C_values=[Y_C],
                include_photons=INCLUDE_PHOTONS,
                include_gluons=True,
                include_electrons=INCLUDE_ELECTRONS,
                include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS,
                print_results=False,
                print_timing=VERBOSE,
                save_to_file=False
            )
            Q_results_by_T = compute_alphabag_table(alphabag_settings)
            
            # Convert to (n_B, T) lookup format
            Q_table = {}
            for grid_key, results in Q_results_by_T.items():
                T = grid_key[0]  # (T, Y_C) tuple
                for i, r in enumerate(results):
                    if r.converged:
                        Q_table[(n_B_values[i], T)] = r
            
            # Save to cache
            save_pure_table(Q_table, "Q", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
        
        t_elapsed = time.time() - t_start
        n_total = len(n_B_values) * len(T_array)
        print(f"    Pure H: {len(H_table)}/{n_total} points")
        print(f"    Pure Q: {len(Q_table)}/{n_total} points")
        print(f"    Time: {t_elapsed:.1f}s")
        
        # ---------------------------------------------------------------------
        # STEP 3: COMPUTE PHASE BOUNDARIES
        # ---------------------------------------------------------------------
        print(f"\n  [Step 3] Computing phase boundaries...")
        
        os.makedirs(BOUNDARY_DIR, exist_ok=True)
        all_boundaries_yc = {}
        
        for eta in eta_values:
            print(f"\n    --- η = {eta:.2f} ---")
            t_start = time.time()
            
            # Use T-marching for robust boundary finding
            boundaries = find_all_boundaries(
                T_array, eta, sfho_params, alphabag_params, particles,
                H_table=H_table, Q_table=Q_table, n_B_values=n_B_values,
                eq_mode="fixed_yc", Y_C=Y_C, verbose=VERBOSE,
                include_pseudoscalar_mesons=INCLUDE_PSEUDOSCALAR_MESONS,
                include_gluons=INCLUDE_GLUONS,
                include_photons=INCLUDE_PHOTONS,
                include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS
            )
            
            # Filter converged only for storage
            converged_boundaries = [b for b in boundaries if b.converged]
            all_boundaries_yc[eta] = boundaries  # Keep all for generate_unified_table
            
            t_elapsed = time.time() - t_start
            print(f"    Converged: {len(converged_boundaries)}/{len(T_array)} ({t_elapsed:.1f}s)")
        
        # Save boundaries to file
        _save_boundaries_file(all_boundaries_yc, Y_C, eta_values, alphabag_params)
        
        # ---------------------------------------------------------------------
        # STEP 4: GENERATE UNIFIED TABLES
        # ---------------------------------------------------------------------
        print(f"\n  [Step 4] Generating unified tables...")
        
        for eta in eta_values:
            boundaries = all_boundaries_yc.get(eta, [])
            if not boundaries:
                print(f"    η = {eta:.2f}: No boundaries, skipping...")
                continue
            
            print(f"    η = {eta:.2f}: ", end="", flush=True)
            t_start = time.time()
            
            # Convert boundaries to dict by T - include full boundary for onset solution
            boundaries_by_T = {b.T: {'n_onset': b.n_B_onset, 'n_offset': b.n_B_offset, 'boundary': b}
                              for b in boundaries if b.converged}
            
            results = generate_unified_table(
                n_B_values, T_array, eta,
                sfho_params, alphabag_params, particles,
                H_table, Q_table, boundaries_by_T,
                eq_mode="fixed_yc", Y_C=Y_C,
                include_pseudoscalar_mesons=INCLUDE_PSEUDOSCALAR_MESONS,
                include_gluons=INCLUDE_GLUONS,
                include_photons=INCLUDE_PHOTONS,
                include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS,
                verbose=True
            )
            
            t_elapsed = time.time() - t_start
            all_results[(eta, Y_C)] = results
            
            n_mixed = sum(1 for r in results if hasattr(r, 'chi') and 0 < r.chi < 1)
            print(f"{len(results)} points, {n_mixed} mixed ({t_elapsed:.1f}s)")
        
        # ---------------------------------------------------------------------
        # STEP 5: SAVE RESULTS
        # ---------------------------------------------------------------------
        print(f"\n  [Step 5] Saving tables...")
        
        for (eta, yc), results in all_results.items():
            if yc != Y_C or not results:
                continue
            _save_eos_tables(results, eta, OUTPUT_DIR, alphabag_params, Y_C=yc)
        
        yc_elapsed = time.time() - yc_start_time
        print(f"\n  Total time for Y_C = {Y_C}: {yc_elapsed:.1f}s")


def _run_beta_mode(T_array, n_B_values, eta_values,
                   sfho_params, alphabag_params, particles):
    """
    Run orchestrator in beta equilibrium mode.
    
    Steps:
        1. Compute pure H and Q phase tables (with caching)
        2. Find phase boundaries for each η
        3. Generate unified tables
        4. Save results
    """
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"Processing Beta Equilibrium Mode")
    print(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # STEP 2: COMPUTE PURE PHASE TABLES
    # -------------------------------------------------------------------------
    print(f"\n  [Step 2] Computing pure phase tables...")
    
    t_start = time.time()
    eq_mode_str = "beta"
    
    # --- Pure H Phase ---
    H_table = None
    if not FORCE_RECOMPUTE_PURE_PHASES:
        H_table = load_pure_table("H", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
        
    if H_table is None:
        print("    Computing pure H table...")
        sfho_settings = SFHoTableSettings(
            n_B_values=n_B_values,
            T_values=T_array,
            equilibrium='beta_eq',
            parametrization=SFHO_PARAMETRIZATION,
            particle_content=PARTICLE_CONTENT,
            include_photons=INCLUDE_PHOTONS,
            include_muons=INCLUDE_MUONS,
            include_pseudoscalar_mesons=INCLUDE_PSEUDOSCALAR_MESONS,
            print_results=False,
            print_timing=VERBOSE,
            save_to_file=False
        )
        H_results_by_T = compute_sfho_table(sfho_settings)
        
        # Convert to (n_B, T) lookup format
        H_table = {}
        for grid_key, results in H_results_by_T.items():
            T = grid_key[0]  # (T,) tuple
            for i, r in enumerate(results):
                if r.converged:
                    H_table[(n_B_values[i], T)] = r
        
        save_pure_table(H_table, "H", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
    
    # --- Pure Q Phase ---
    Q_table = None
    if not FORCE_RECOMPUTE_PURE_PHASES:
        Q_table = load_pure_table("Q", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
        
    if Q_table is None:
        print("    Computing pure Q table...")
        alphabag_settings = AlphaBagTableSettings(
            params=alphabag_params,
            phase='unpaired',
            equilibrium='beta_eq',
            n_B_values=n_B_values,
            T_values=T_array,
            include_photons=INCLUDE_PHOTONS,
            include_gluons=True,
            include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS,
            print_results=False,
            print_timing=VERBOSE,
            save_to_file=False
        )
        Q_results_by_T = compute_alphabag_table(alphabag_settings)
        
        # Convert to (n_B, T) lookup format
        Q_table = {}
        for grid_key, results in Q_results_by_T.items():
            T = grid_key[0]  # (T,) tuple
            for i, r in enumerate(results):
                if r.converged:
                    Q_table[(n_B_values[i], T)] = r
        
        save_pure_table(Q_table, "Q", eq_mode_str, alphabag_params, OUTPUT_DIR, n_B_values)
    
    t_elapsed = time.time() - t_start
    n_total = len(n_B_values) * len(T_array)
    print(f"    Pure H: {len(H_table)}/{n_total} points")
    print(f"    Pure Q: {len(Q_table)}/{n_total} points")
    print(f"    Time: {t_elapsed:.1f}s")
    
    # -------------------------------------------------------------------------
    # STEP 3: COMPUTE PHASE BOUNDARIES
    # -------------------------------------------------------------------------
    print(f"\n  [Step 3] Computing phase boundaries...")
    
    os.makedirs(BOUNDARY_DIR, exist_ok=True)
    all_boundaries = {}
    
    for eta in eta_values:
        print(f"\n    --- η = {eta:.2f} ---")
        t_start = time.time()
        
        # Use T-marching for robust boundary finding
        boundaries = find_all_boundaries(
            T_array, eta, sfho_params, alphabag_params, particles,
            H_table=H_table, Q_table=Q_table, n_B_values=n_B_values,
            eq_mode="beta", verbose=VERBOSE,
            include_pseudoscalar_mesons=INCLUDE_PSEUDOSCALAR_MESONS,
            include_gluons=INCLUDE_GLUONS,
            include_photons=INCLUDE_PHOTONS,
            include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS
        )
        
        # Filter converged only for storage
        converged_boundaries = [b for b in boundaries if b.converged]
        all_boundaries[eta] = boundaries  # Keep all for generate_unified_table
        
        t_elapsed = time.time() - t_start
        print(f"    Converged: {len(converged_boundaries)}/{len(T_array)} ({t_elapsed:.1f}s)")
    
    # -------------------------------------------------------------------------
    # STEP 4: GENERATE UNIFIED TABLES
    # -------------------------------------------------------------------------
    print(f"\n  [Step 4] Generating unified tables...")
    
    for eta in eta_values:
        boundaries = all_boundaries.get(eta, [])
        if not boundaries:
            print(f"    η = {eta:.2f}: No boundaries, skipping...")
            continue
        
        print(f"    η = {eta:.2f}: ", end="", flush=True)
        t_start = time.time()
        
        # Convert boundaries to dict by T - include full boundary for onset solution
        boundaries_by_T = {b.T: {'n_onset': b.n_B_onset, 'n_offset': b.n_B_offset, 'boundary': b}
                          for b in boundaries if b.converged}
        
        results = generate_unified_table(
            n_B_values, T_array, eta,
            sfho_params, alphabag_params, particles,
            H_table, Q_table, boundaries_by_T, eq_mode="beta",
            include_pseudoscalar_mesons=INCLUDE_PSEUDOSCALAR_MESONS,
            include_gluons=INCLUDE_GLUONS,
            include_photons=INCLUDE_PHOTONS,
            include_thermal_neutrinos=INCLUDE_THERMAL_NEUTRINOS,
            verbose=True
        )
        
        t_elapsed = time.time() - t_start
        all_results[eta] = results
        
        n_mixed = sum(1 for r in results if hasattr(r, 'chi') and 0 < r.chi < 1)
        print(f"{len(results)} points, {n_mixed} mixed ({t_elapsed:.1f}s)")
    
    # -------------------------------------------------------------------------
    # STEP 5: SAVE RESULTS
    # -------------------------------------------------------------------------
    print(f"\n  [Step 5] Saving tables...")
    
    for eta, results in all_results.items():
        if results:
            _save_eos_tables(results, eta, OUTPUT_DIR, alphabag_params)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    run_orchestrator()
