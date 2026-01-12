"""
general_bose_integrals.py
Efficient evaluation of relativistic Bose-Einstein integrals for EOS calculations.

Methods:
    1. JEL (Johns, Ellis, Lattimer 1996) - Fast rational approximation
    2. Numerical integration (scipy.quad) for validation

Reference:
    Johns, Ellis & Lattimer, ApJ 473, 1020 (1996), Table 11
    
Notes:
    - For bosons, μ ≤ m always (to avoid Bose-Einstein condensation singularity)
    - The JEL parameter h is defined via ψ(h) = (μ-m)/T, with ψ ≤ 0

Units: All quantities in MeV (energy/mass) and fm (length)
Returns: n (fm⁻³), P (MeV/fm³), e (MeV/fm³), s (fm⁻³), ns (fm⁻³)
"""
import numpy as np
import scipy.integrate as integrate
from general_physics_constants import hc, hc3, PI2

# Numba JIT decorator - use identity if numba not available
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# =============================================================================
# JEL APPROXIMATION PARAMETERS (Table 11 of JEL 1996, Bose case)
# =============================================================================
aBJEL = 1.040       # 'a' parameter for bosons
MBJEL = 3           # M (h power max)
NBJEL = 4           # N (t power max)

# Coefficient matrix p_{mn} from Table 11
pmnB = np.array([
    [1.68130, 6.85060, 10.8539, 7.81762, 2.16465],
    [6.7252, 27.4024, 43.4156, 31.27048, 8.6586],
    [8.51373, 35.6576, 57.7975, 42.4049, 11.8321],
    [3.47433, 15.1995, 25.6536, 19.3811, 5.54423]
], dtype=np.float64)

# =============================================================================
# PRE-COMPUTED LOOKUP TABLES
# =============================================================================
_h_grid = np.concatenate((np.array([0.0]), np.logspace(-8, 8, 10000)))
_sqrt_a = np.sqrt(aBJEL)

# ψ(h) = h/(√a + h) - ln((√a + h)/√a)
_psi_grid_b = _h_grid / (_sqrt_a + _h_grid) - np.log((_sqrt_a + _h_grid) / _sqrt_a)

# Sort for interpolation (ψ is monotonically decreasing)
_psi_grid_b_sorted = _psi_grid_b[::-1]
_h_grid_sorted = _h_grid[::-1]
_PSI_MIN_B = _psi_grid_b_sorted[0]

# Gauss-Laguerre nodes and weights
_GL_ORDER = 30
_GL_NODES, _GL_WEIGHTS = np.polynomial.laguerre.laggauss(_GL_ORDER)
_GL_NODES = _GL_NODES.astype(np.float64)
_GL_WEIGHTS = _GL_WEIGHTS.astype(np.float64)

# =============================================================================
# JEL CORE FUNCTIONS
# =============================================================================
@njit(fastmath=True, cache=True)
def _psi_of_h(h):
    """
    Compute ψ(h) for bosons:
    ψ(h) = h/(√a + h) - ln((√a + h)/√a)
    
    For bosons, ψ(h) ≤ 0 for all h ≥ 0.
    ψ(0) = 0 corresponds to μ = m (condensation threshold).
    """
    if h < 1e-4:
        # Small h expansion: ψ ≈ -h²/(2a) + h³/(3a^{3/2})
        return -(h**2) / (2.0 * aBJEL) + (h**3) / (3.0 * aBJEL**1.5)
    
    term = _sqrt_a + h
    return h / term - np.log(term / _sqrt_a)


@njit(fastmath=True, cache=True)
def _dpsi_dh(h):
    """Derivative dψ/dh for Newton-Raphson iteration."""
    term = _sqrt_a + h
    return -h / (term**2)


@njit(fastmath=True, cache=True)
def _find_h_jel(mu, T, m, psi_grid, h_grid):
    """
    Find the JEL parameter h such that ψ(h) = (μ-m)/T.
    
    For bosons: μ ≤ m, so ψ_target ≤ 0.
    
    Returns:
        h_val: The JEL parameter (h ≥ 0)
        err: Convergence indicator
    """
    psi_target = (mu - m) / T
    
    # Enforce μ ≤ m constraint
    if psi_target > 0:
        psi_target = 0.0
    
    # Initial guess based on regime
    if psi_target < -14.0:
        # Deep non-degenerate: h ≈ √a × exp(1 - ψ)
        # Clamp argument to prevent overflow (max exp ~ 700)
        arg = min(1.0 - psi_target, 700.0)
        h_guess = _sqrt_a * np.exp(arg)
    elif psi_target > -1e-6:
        # Near condensation: ψ ≈ -h²/(2a), so h ≈ √(-2a×ψ)
        h_guess = np.sqrt(-2.0 * aBJEL * psi_target)
    else:
        # Interpolate from table
        h_guess = np.interp(psi_target, psi_grid, h_grid)
    
    # Newton-Raphson refinement
    h_curr = h_guess
    tol = 1e-10
    max_iter = 50
    
    for _ in range(max_iter):
        if h_curr < 0:
            h_curr = 0.0
        
        val = _psi_of_h(h_curr)
        diff = val - psi_target
        
        if np.abs(diff) < tol:
            break
        
        deriv = _dpsi_dh(h_curr)
        if deriv == 0:
            break
        
        h_next = h_curr - diff / deriv
        if h_next < 0:
            h_next = h_curr * 0.5  # Damped step
        h_curr = h_next
    
    return h_curr, np.abs(h_curr)


@njit(fastmath=True, cache=True)
def _compute_bose_thermo_single(h_val, T, m, g):
    """
    Compute (n, P, ε) for bosons at given h, T, m.
    
    Uses the JEL rational approximation (Eqs. 25-29 of JEL 1996).
    Only computes particles (not antiparticles).
    """
    # Regularize h near condensation
    hEff = h_val if h_val >= 1e-12 else 1e-12
    
    # If h is extremely large, bosons are exponentially suppressed
    # Return zero to avoid numerical overflow
    # (h > 100 means μ-m << -T, so f_B ~ exp(-(m-μ)/T) ~ exp(-ψT/T) ~ 0)
    if hEff > 1e6:
        return 0.0, 0.0, 0.0
    
    # Dimensionless variables
    t = T / m
    t1 = 1.0 + t
    h1 = 1.0 + hEff
    sqrt_a = np.sqrt(aBJEL)
    
    # Pre-compute powers of t
    t_pow = np.ones(NBJEL + 1)
    for j in range(1, NBJEL + 1):
        t_pow[j] = t_pow[j-1] * t
    
    # Common prefactor
    const_pre = g / (2.0 * PI2 * hc3)
    
    # 1. Pressure (Eq. 26)
    sum_P = 0.0
    h_pow_i = 1.0  # h^0
    for i in range(MBJEL + 1):
        for j in range(NBJEL + 1):
            sum_P += pmnB[i, j] * h_pow_i * t_pow[j]
        h_pow_i *= hEff  # h^{i+1}
    
    denom_P = h1**(MBJEL + 1) * t1**NBJEL
    P_res = const_pre * m**4 * (t**2.5 * t1**1.5 / denom_P) * sum_P
    
    # 2. Number Density - exact Mathematica formula:
    # sum0 += pm[i,j] * h^{i-2} * t^j * (-i + h*(M+1-i))
    sum_n = 0.0
    for i in range(MBJEL + 1):
        h_pow_im2 = hEff**(i - 2)  # h^{i-2}
        coeff = -i + hEff * (MBJEL + 1 - i)  # (-i + h*(M+1-i))
        for j in range(NBJEL + 1):
            sum_n += pmnB[i, j] * h_pow_im2 * t_pow[j] * coeff
    
    denom_n = h1**(MBJEL + 2) * t1**NBJEL
    pre_n = t**1.5 * t1**1.5 * (sqrt_a + hEff)**2 / denom_n
    n_res = const_pre * m**3 * pre_n * sum_n
    
    # 3. Energy Density (kinetic part) - Eq. 28
    sum_e = 0.0
    h_pow_i = 1.0
    for i in range(MBJEL + 1):
        for j in range(NBJEL + 1):
            coeff_e = 1.5 + j + (1.5 - NBJEL) * t / t1
            sum_e += pmnB[i, j] * h_pow_i * t_pow[j] * coeff_e
        h_pow_i *= hEff
    
    u_res = const_pre * m**4 * (t**2.5 * t1**1.5 / denom_P) * sum_e
    e_res = u_res + m * n_res  # Total = kinetic + rest mass
    
    return n_res, P_res, e_res


# =============================================================================
# MAIN JEL SOLVER
# =============================================================================
@njit(fastmath=True, cache=True)
def calculate_jel_bose_fast(mu, T, m, g_deg, psi_grid, h_grid,
                            include_antiparticles, return_error):
    """
    Main JEL calculator for Bose integrals.
    
    Parameters:
        mu: Chemical potential (MeV), must satisfy μ ≤ m
        T: Temperature (MeV)
        m: Mass (MeV)
        g_deg: Degeneracy factor
        psi_grid, h_grid: Pre-computed lookup tables
        include_antiparticles: Include antiparticle contribution
        return_error: Return convergence error
    
    Returns:
        Array [n, P, e, s, ns] or [n, P, e, s, ns, err]
    """
    # Particles
    h_part, err_part = _find_h_jel(mu, T, m, psi_grid, h_grid)
    n_p, P_p, e_p = _compute_bose_thermo_single(h_part, T, m, g_deg)
    
    n_tot, P_tot, e_tot, max_err = n_p, P_p, e_p, err_part
    
    # Antiparticles (μ → -μ)
    if include_antiparticles:
        h_anti, err_anti = _find_h_jel(-mu, T, m, psi_grid, h_grid)
        n_a, P_a, e_a = _compute_bose_thermo_single(h_anti, T, m, g_deg)
        
        n_tot = n_p - n_a   # Net number
        P_tot = P_p + P_a   # Total pressure
        e_tot = e_p + e_a   # Total energy
        max_err = max(err_part, err_anti)
    
    # Derived quantities
    s_tot = (P_tot + e_tot - mu * n_tot) / T
    ns_tot = (e_tot - 3.0 * P_tot) / m if m > 1e-9 else 0.0
    
    if return_error:
        return np.array([n_tot, P_tot, e_tot, s_tot, ns_tot, max_err])
    return np.array([n_tot, P_tot, e_tot, s_tot, ns_tot])


# =============================================================================
# PUBLIC API
# =============================================================================
def solve_bose_jel(mu, T, m, g, include_antiparticles=True, return_error=False):
    """
    Solve Bose integrals using JEL approximation.
    
    Parameters:
        mu: Chemical potential (MeV), must satisfy |μ| ≤ m
        T: Temperature (MeV)
        m: Particle mass (MeV)
        g: Degeneracy factor
        include_antiparticles: Include antiparticle contribution (default True)
        return_error: Return convergence error (default False)
    
    Returns:
        tuple or array: (n, P, e, s, ns) [, err]
            n: Number density (fm⁻³)
            P: Pressure (MeV/fm³)
            e: Energy density (MeV/fm³)
            s: Entropy density (fm⁻³)
            ns: Scalar density (fm⁻³)
    
    Note:
        For bosons, |μ| must not exceed m to avoid the Bose-Einstein
        condensation singularity. Values of μ > m are automatically
        clamped to μ = m.
        
        At T=0, thermal bosons contribute nothing (only condensates exist),
        so we return zeros.
    """
    T = float(T)
    
    # At T=0 or very low T, thermal bosons don't exist
    # Return zeros to avoid numerical overflow
    if T < 1e-6:  # T < 1 eV effectively means T=0
        if return_error:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    
    return calculate_jel_bose_fast(
        float(mu), T, float(m), float(g),
        _psi_grid_b_sorted, _h_grid_sorted,
        include_antiparticles, return_error
    )


# =============================================================================
# GAUSS-LAGUERRE ALTERNATIVE
# =============================================================================
@njit(cache=True, fastmath=True)
def _bose_gauss_laguerre_kernel(mu, T, m, g, nodes, weights, include_antiparticles):
    """
    Gauss-Laguerre quadrature for relativistic Bose integrals.
    
    Uses change of variables x = (E - m)/T for exponential weighting.
    """
    inv_T = 1.0 / T
    
    # Prefactors
    const_n = g / (2.0 * PI2 * hc3)
    const_P = g / (6.0 * PI2 * hc3)
    const_e = g / (2.0 * PI2 * hc3)
    
    # Exponential factors (must be > 0 to avoid condensation singularity)
    # For bosons: require mu < m, so A_p = (m - mu)/T > 0
    A_p = (m - mu) * inv_T
    if A_p < 1e-10:
        A_p = 1e-10  # Clamp to avoid singularity
    expA_p = np.exp(A_p)
    
    A_a = (m + mu) * inv_T
    expA_a = np.exp(A_a)
    
    n_sum, P_sum, e_sum = 0.0, 0.0, 0.0
    
    for i in range(nodes.shape[0]):
        x = nodes[i]
        w = weights[i]
        
        E = m + T * x
        k2 = E * E - m * m
        k = np.sqrt(k2) if k2 > 0.0 else 0.0
        
        exp_minus_x = np.exp(-x)
        
        # Bose distribution (without e^{-x} weight, already in quadrature)
        # f_B(E) = 1/(exp((E-μ)/T) - 1) = 1/(exp(A) * exp(x/T) - 1)
        denom_p = expA_p - exp_minus_x
        if denom_p > 1e-12:
            f_p_eff = 1.0 / denom_p
        else:
            f_p_eff = 0.0  # Avoid singularity
        
        if include_antiparticles:
            denom_a = expA_a - exp_minus_x
            if denom_a > 1e-12:
                f_a_eff = 1.0 / denom_a
            else:
                f_a_eff = 0.0
        else:
            f_a_eff = 0.0
        
        diff = f_p_eff - f_a_eff
        sumfa = f_p_eff + f_a_eff
        
        # Integrand contributions
        Tk = T * k
        n_sum += w * Tk * E * diff
        P_sum += w * Tk * k * k * sumfa
        e_sum += w * Tk * E * E * sumfa
    
    # Physical quantities
    n_res = const_n * n_sum
    P_res = const_P * P_sum
    e_res = const_e * e_sum
    
    s_res = (P_res + e_res - mu * n_res) / T
    ns_res = (e_res - 3.0 * P_res) / m if m > 0.0 else 0.0
    
    return n_res, P_res, e_res, s_res, ns_res


def solve_bose_gl(mu, T, m, g, include_antiparticles=True):
    """
    Solve Bose integrals using Gauss-Laguerre quadrature.
    
    Higher accuracy than JEL, but slower.
    
    Parameters:
        mu: Chemical potential (MeV), must satisfy |μ| < m
        T: Temperature (MeV)
        m: Particle mass (MeV)
        g: Degeneracy factor
        include_antiparticles: Include antiparticle contribution (default True)
    
    Returns:
        tuple: (n, P, e, s, ns)
            n: Number density (fm⁻³)
            P: Pressure (MeV/fm³)
            e: Energy density (MeV/fm³)
            s: Entropy density (fm⁻³)
            ns: Scalar density (fm⁻³)
    
    Note:
        For bosons, |μ| must be less than m to avoid the Bose-Einstein
        condensation singularity.
    """
    mu, T, m, g = float(mu), float(T), float(m), float(g)
    
    # Clamp μ to avoid condensation singularity
    if mu >= m:
        mu = m * (1.0 - 1e-10)
    if mu <= -m:
        mu = -m * (1.0 - 1e-10)
    
    return _bose_gauss_laguerre_kernel(
        mu, T, m, g, _GL_NODES, _GL_WEIGHTS, include_antiparticles
    )

def Bose_Numerical(mu, T, m, g, include_antiparticles=True):
    """
    Direct numerical integration for Bose integrals (validation).
    
    Note: This will fail if μ approaches m (condensation singularity).
    """
    prefactor = g / (2.0 * PI2 * hc3)
    
    # Clamp μ to avoid singularity
    if mu >= m:
        mu = m * (1.0 - 1e-12)
    
    def distrib(E, chem_pot):
        arg = (E - chem_pot) / T
        if arg < 1e-4:
            return 1.0 / arg  # Approximate for small arg
        if arg > 100:
            return 0.0
        return 1.0 / (np.exp(arg) - 1.0)
    
    def integrands(k):
        E = np.sqrt(k**2 + m**2)
        f_p = distrib(E, mu)
        f_a = distrib(E, -mu) if include_antiparticles else 0.0
        
        dn = (f_p - f_a) * k**2
        dP = (f_p + f_a) * k**4 / (3.0 * E)
        de = (f_p + f_a) * k**2 * E
        return dn, dP, de
    
    upper_limit = max(1000.0, abs(mu) + 20*T)
    
    n_res = prefactor * integrate.quad(lambda k: integrands(k)[0], 0, upper_limit)[0]
    P_res = prefactor * integrate.quad(lambda k: integrands(k)[1], 0, upper_limit)[0]
    e_res = prefactor * integrate.quad(lambda k: integrands(k)[2], 0, upper_limit)[0]
    
    s_res = (P_res + e_res - mu * n_res) / T
    ns_res = (e_res - 3*P_res) / m if m > 0 else 0
    
    return n_res, P_res, e_res, s_res, ns_res