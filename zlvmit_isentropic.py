#!/usr/bin/env python3
"""
zlvmit_isentropic.py
====================
Isentropic calculations for ZL+vMIT EOS.

This module provides:
    - Entropy interpolation from computed tables
    - Isentropic temperature finder (T at given S = s/n_B)
    - Batch isentrope computation

Usage:
    from zlvmit_isentropic import compute_isentropes, create_entropy_interpolator
    
    # After loading table data:
    interpolator = create_entropy_interpolator(data, n_B_values, T_values)
    T_isentrope = compute_isentropic_temperature(n_B=0.32, S_target=2.0, interpolator=interpolator)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import brentq, root_scalar
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# ENTROPY INTERPOLATOR
# =============================================================================

def create_entropy_interpolator(
    data: Dict[str, np.ndarray],
    n_B_values: np.ndarray = None,
    T_values: np.ndarray = None
) -> RegularGridInterpolator:
    """
    Create a 2D interpolator for s/n_B(n_B, T) from table data.
    
    Args:
        data: Dictionary with 'n_B', 'T', 's_total' arrays (from table output)
        n_B_values: Unique n_B grid values (if None, extracted from data)
        T_values: Unique T grid values (if None, extracted from data)
        
    Returns:
        RegularGridInterpolator for s/n_B(n_B, T)
    """
    # Extract unique grid values if not provided
    if n_B_values is None:
        n_B_values = np.unique(data['n_B'])
    if T_values is None:
        T_values = np.unique(data['T'])
    
    n_nB = len(n_B_values)
    n_T = len(T_values)
    
    # Compute entropy per baryon: S = s_total / n_B
    s_per_nB = data['s_total'] / data['n_B']
    
    # Reshape to 2D grid (n_B, T)
    # Assuming data is ordered by (n_B varies fastest) or (T varies fastest)
    # Check ordering by looking at first few entries
    if len(data['T']) >= 2 and data['T'][0] == data['T'][1]:
        # n_B varies fastest, T is outer loop
        S_grid = s_per_nB.reshape(n_T, n_nB).T  # Transpose to (n_B, T)
    else:
        # T varies fastest, n_B is outer loop
        S_grid = s_per_nB.reshape(n_nB, n_T)
    
    # Create interpolator
    interpolator = RegularGridInterpolator(
        (n_B_values, T_values), 
        S_grid,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolate
    )
    
    return interpolator


def create_entropy_interpolator_from_file(
    filepath: Union[str, Path],
    n_B_column: str = 'n_B',
    T_column: str = 'T',
    s_column: str = 's_total'
) -> Tuple[RegularGridInterpolator, np.ndarray, np.ndarray]:
    """
    Create entropy interpolator from a table file.
    
    Args:
        filepath: Path to the .dat file
        n_B_column, T_column, s_column: Column names
        
    Returns:
        (interpolator, n_B_values, T_values)
    """
    # Load data from file
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        # Find header line
        for line in f:
            if line.startswith('#') and n_B_column in line:
                headers = line[1:].split()
                break
        else:
            raise ValueError(f"Could not find column headers in {filepath}")
    
    # Load data
    raw_data = np.loadtxt(filepath, comments='#')
    
    # Create dictionary
    data = {}
    for i, h in enumerate(headers):
        data[h.strip()] = raw_data[:, i]
    
    n_B_values = np.unique(data[n_B_column])
    T_values = np.unique(data[T_column])
    
    interpolator = create_entropy_interpolator(data, n_B_values, T_values)
    
    return interpolator, n_B_values, T_values


# =============================================================================
# ISENTROPIC TEMPERATURE FINDER
# =============================================================================

def compute_isentropic_temperature(
    n_B: float,
    S_target: float,
    interpolator: RegularGridInterpolator,
    T_bounds: Tuple[float, float] = (0.1, 150.0),
    tol: float = 1e-6
) -> Optional[float]:
    """
    Find temperature T such that s(n_B, T)/n_B = S_target.
    
    Uses Brent's method root finding on the interpolated entropy function.
    
    Args:
        n_B: Baryon density (fm⁻³)
        S_target: Target entropy per baryon (dimensionless, in units of k_B)
        interpolator: RegularGridInterpolator for s/n_B(n_B, T)
        T_bounds: (T_min, T_max) search range in MeV
        tol: Tolerance for root finding
        
    Returns:
        Temperature T (MeV) or None if no solution found
    """
    T_min, T_max = T_bounds
    
    def objective(T):
        """S(n_B, T)/n_B - S_target"""
        S = interpolator((n_B, T))
        return S - S_target
    
    # Check if solution exists within bounds
    try:
        f_min = objective(T_min)
        f_max = objective(T_max)
    except Exception:
        return None
    
    # Check for sign change (necessary for Brent's method)
    if f_min * f_max > 0:
        # No sign change - check if target is outside range
        # Try to find a valid interval by searching
        T_test = np.linspace(T_min, T_max, 50)
        f_test = [objective(T) for T in T_test]
        
        # Look for sign changes
        for i in range(len(f_test) - 1):
            if f_test[i] * f_test[i+1] < 0:
                T_min, T_max = T_test[i], T_test[i+1]
                break
        else:
            # No valid interval found
            return None
    
    try:
        result = brentq(objective, T_min, T_max, xtol=tol)
        return result
    except ValueError:
        return None


def compute_isentropes(
    n_B_values: np.ndarray,
    S_values: np.ndarray = None,
    interpolator: RegularGridInterpolator = None,
    T_bounds: Tuple[float, float] = (0.1, 150.0)
) -> Dict[float, np.ndarray]:
    """
    Compute T(n_B) curves for multiple fixed entropy values.
    
    Args:
        n_B_values: Array of baryon densities (fm⁻³)
        S_values: Array of target S = s/n_B values 
                  (default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] in k_B)
        interpolator: RegularGridInterpolator for s/n_B(n_B, T)
        T_bounds: (T_min, T_max) search range in MeV
        
    Returns:
        Dictionary {S: T_array} where T_array[i] is T for n_B_values[i]
        NaN values indicate no solution found at that point.
    """
    if S_values is None:
        S_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    result = {}
    
    for S in S_values:
        T_array = np.full_like(n_B_values, np.nan)
        
        for i, n_B in enumerate(n_B_values):
            T = compute_isentropic_temperature(
                n_B, S, interpolator, T_bounds
            )
            if T is not None:
                T_array[i] = T
        
        result[S] = T_array
    
    return result


# =============================================================================
# ISENTROPE TABLE GENERATION
# =============================================================================

@dataclass
class IsentropeResult:
    """Result from isentrope calculation."""
    S: float              # Entropy per baryon
    n_B_values: np.ndarray  # Density grid
    T_values: np.ndarray    # Temperature at each density
    valid_mask: np.ndarray  # Boolean mask for valid points


def compute_isentrope_table(
    S_values: np.ndarray,
    n_B_values: np.ndarray,
    interpolator: RegularGridInterpolator,
    T_bounds: Tuple[float, float] = (0.1, 150.0)
) -> List[IsentropeResult]:
    """
    Compute isentrope data for tabulation.
    
    Args:
        S_values: Array of entropy per baryon values
        n_B_values: Array of baryon densities
        interpolator: Entropy interpolator
        T_bounds: Temperature search bounds
        
    Returns:
        List of IsentropeResult for each S value
    """
    results = []
    
    for S in S_values:
        T_array = np.full_like(n_B_values, np.nan)
        
        for i, n_B in enumerate(n_B_values):
            T = compute_isentropic_temperature(n_B, S, interpolator, T_bounds)
            if T is not None:
                T_array[i] = T
        
        valid_mask = ~np.isnan(T_array)
        
        results.append(IsentropeResult(
            S=S,
            n_B_values=n_B_values,
            T_values=T_array,
            valid_mask=valid_mask
        ))
    
    return results


def save_isentropes_to_file(
    isentropes: Dict[float, np.ndarray],
    n_B_values: np.ndarray,
    filepath: Union[str, Path],
    header: str = ""
):
    """
    Save isentrope data to file.
    
    Format: columns are n_B, T_S0.5, T_S1.0, T_S1.5, ...
    
    Args:
        isentropes: Dict {S: T_array} from compute_isentropes
        n_B_values: Density grid
        filepath: Output file path
        header: Optional header comment
    """
    S_values = sorted(isentropes.keys())
    
    with open(filepath, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        
        # Column header
        col_names = ['n_B'] + [f'T_S{S:.1f}' for S in S_values]
        f.write("# " + " ".join(f"{name:>12}" for name in col_names) + "\n")
        
        # Data rows
        for i, n_B in enumerate(n_B_values):
            row = [n_B] + [isentropes[S][i] for S in S_values]
            f.write(" ".join(f"{val:>12.6e}" for val in row) + "\n")
    
    print(f"Saved isentropes to {filepath}")


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("Isentropic Calculations Test")
    print("=" * 50)
    
    # Create synthetic test data
    n_B_grid = np.linspace(0.1, 1.0, 20)  # fm⁻³
    T_grid = np.linspace(1.0, 100.0, 30)  # MeV
    
    # Generate synthetic entropy: s ∝ T² / n_B (rough approximation)
    n_B_mesh, T_mesh = np.meshgrid(n_B_grid, T_grid, indexing='ij')
    s_mesh = 0.01 * T_mesh**1.5 / n_B_mesh**0.3  # Synthetic s/n_B
    
    # Flatten for table format
    data = {
        'n_B': n_B_mesh.flatten(),
        'T': T_mesh.flatten(),
        's_total': (s_mesh * n_B_mesh).flatten()  # s = S * n_B
    }
    
    # Create interpolator
    interp = create_entropy_interpolator(data, n_B_grid, T_grid)
    
    # Test single point
    n_B_test = 0.32
    S_test = 2.0
    T_result = compute_isentropic_temperature(n_B_test, S_test, interp)
    
    if T_result is not None:
        # Verify
        S_check = interp((n_B_test, T_result))
        print(f"Test point: n_B={n_B_test} fm⁻³, S_target={S_test}")
        print(f"  Found T = {T_result:.2f} MeV")
        print(f"  Check: S(n_B, T) = {S_check:.4f} (target: {S_test})")
    else:
        print("No solution found for test point")
    
    # Compute isentropes
    print("\nComputing isentropes...")
    isentropes = compute_isentropes(n_B_grid, S_values=np.array([1.0, 2.0, 3.0]), interpolator=interp)
    
    for S, T_arr in isentropes.items():
        valid = ~np.isnan(T_arr)
        print(f"  S={S}: {np.sum(valid)}/{len(T_arr)} valid points, T range: {np.nanmin(T_arr):.1f} - {np.nanmax(T_arr):.1f} MeV")
    
    print("\nOK!")
