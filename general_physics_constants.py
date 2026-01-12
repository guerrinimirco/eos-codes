"""
general_physics_constants.py
====================
Central repository for fundamental physical constants and unit conversions.
All quantities in natural units: MeV for energy/mass, fm for length.

Values from Particle Data Group (PDG) compilation.
"""
import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS (PDG values)
# =============================================================================
# Speed of light
c_SI = 299792458              # m/s (exact)
c_cgs = 2.99792458e10         # cm/s

# Planck's constant (reduced)
hbar_SI = 1.054571817e-34     # J·s
hbar_MeV_s = 6.582119569e-22  # MeV·s
hc = 197.3269804              # MeV·fm (ℏc)
hc2 = hc**2                   # (MeV·fm)²
hc3 = hc**3                   # (MeV·fm)³

# Boltzmann constant
k_B_SI = 1.380649e-23         # J/K (exact, SI definition)
k_B_MeV = 8.617333262e-11     # MeV/K

# Gravitational constant
G_SI = 6.67430e-11            # m³/kg/s²
G_newton = 1.3271244e11       # km³/s²/M_sun (for TOV equations)
G_MeV = 6.70883e-39           # GeV⁻²c⁴ = MeV⁻² (fm/MeV)² in natural units

# Fine structure constant & elementary charge
alpha_EM = 1.0 / 137.035999084   # dimensionless
e2_MeV_fm = 1.439964547       # e² in MeV·fm (= α·ℏc)

# =============================================================================
# PARTICLE MASSES (PDG values, MeV/c²)
# =============================================================================
m_neutron = 939.56542052      # MeV/c²
m_proton = 938.27208816       # MeV/c²
m_electron = 0.51099895000    # MeV/c²
m_muon = 105.6583745          # MeV/c²
m_nucleon = (m_neutron + m_proton) / 2.0  # Average nucleon mass

# Unified atomic mass unit
u_MeV = 931.49410242          # MeV/c²

# Solar mass
M_sun_kg = 1.98841e30         # kg
M_sun_g = 1.98841e33          # g
M_sun_MeV = 1.11542e60        # MeV/c²

# =============================================================================
# NUCLEAR SATURATION PROPERTIES (typical/reference values)
# =============================================================================
n0_default = 0.16             # fm⁻³ (saturation density, typical value)
n0_sfho = 0.1583                 # fm⁻³ (commonly used for SFHo)

# =============================================================================
# UNIT CONVERSIONS
# =============================================================================
# Pressure: MeV/fm³ → dyne/cm² (CGS)
MEV_FM3_TO_DYNE_CM2 = 1.602176634e33 

# Mass density: MeV/fm³ → g/cm³ (via E=mc²)
MEV_FM3_TO_G_CM3 = MEV_FM3_TO_DYNE_CM2 / (c_cgs**2)

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================
PI = np.pi
PI2 = PI**2
TWO_PI2 = 2.0 * PI2
SQRT2 = np.sqrt(2.0)

# =============================================================================
# DERIVED CONSTANTS (commonly used in integrals)
# =============================================================================
# Prefactor for fermion/boson integrals: 1/(2π²ℏ³c³)
PHASE_SPACE_PREFACTOR = 1.0 / (TWO_PI2 * hc3)


# =============================================================================
# UNIT CONVERSION FUNCTIONS
# =============================================================================
def convert_pressure_to_cgs(P_mev_fm3: float) -> float:
    """Convert pressure from MeV/fm³ to dyne/cm²."""
    return P_mev_fm3 * MEV_FM3_TO_DYNE_CM2


def convert_energy_density_to_cgs(e_mev_fm3: float) -> float:
    """Convert energy density from MeV/fm³ to g/cm³."""
    return e_mev_fm3 * MEV_FM3_TO_G_CM3


def convert_density_to_mev_fm3(rho_g_cm3: float) -> float:
    """Convert mass density from g/cm³ to MeV/fm³."""
    return rho_g_cm3 / MEV_FM3_TO_G_CM3


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("Physical Constants Module (PDG values)")
    print("=" * 50)
    print(f"ℏc = {hc:.6f} MeV·fm")
    print(f"α = 1/{1/alpha_EM:.6f}")
    print(f"e² = {e2_MeV_fm:.6f} MeV·fm")
    print()
    print("Particle masses:")
    print(f"  m_n = {m_neutron:.8f} MeV")
    print(f"  m_p = {m_proton:.8f} MeV")
    print(f"  m_e = {m_electron:.11f} MeV")
    print(f"  m_μ = {m_muon:.7f} MeV")
    print()
    print(f"n_sat = {n_sat} fm⁻³")