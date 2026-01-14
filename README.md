# Nuclear Equation of State (EOS) Table Generator

A comprehensive Python library for calculating Equations of State (EOS) for astrophysical simulations, including neutron star matter, supernovae, and compact object mergers.

## Overview

This repository implements several nuclear and quark matter EOS models:

| Model | Description | Particles | 
|-------|-------------|-----------|
| **SFHo** | Relativistic Mean Field model | Nucleons, Hyperons, Deltas |
| **AlphaBag** | α_s-corrected MIT bag model | u, d, s quarks |
| **ZL** | Zhao-Lattimer nucleonic model | Nucleons |
| **vMIT** | Vector-MIT bag model | u, d, s quarks |
| **ZLvMIT** | Hybrid hadron-quark construction | Mixed phase |

---

## File Structure

### General Modules (`general_*.py`)
Shared infrastructure used by all models.

| File | Description |
|------|-------------|
| `general_physics_constants.py` | Physical constants (ħc, m_n, m_p, etc.) |
| `general_particles.py` | Particle definitions (baryons, quarks, leptons) |
| `general_fermi_integrals.py` | Fermi-Dirac integrals for fermions |
| `general_bose_integrals.py` | Bose-Einstein integrals for mesons |
| `general_thermodynamics_leptons.py` | Electron, muon, neutrino thermodynamics |
| `general_eos_solver.py` | Generic EOS solver framework |
| `general_plotting_info.py` | Plotting utilities and styles |

### SFHo Model (`sfho_*.py`)
Relativistic Mean Field (RMF) hadronic EOS with σ-ω-ρ-φ meson exchange.

| File | Description |
|------|-------------|
| `sfho_parameters.py` | Model parameters (sfho, sfhoy, 2fam variants) |
| `sfho_thermodynamics_hadrons.py` | Hadron thermodynamics with meson fields |
| `sfho_eos.py` | EOS solvers (beta_eq, fixed_yc, trapped_neutrinos) |
| `sfho_compute_tables.py` | **Main script to generate SFHo tables** |
| `sfho_nuclear_saturation_properties.py` | Nuclear matter properties at saturation |
| `sfho_compare_with_compose.py` | Validation against CompOSE tables |

### AlphaBag Model (`alphabag_*.py`)
α_s-corrected MIT bag model for quark matter with perturbative QCD corrections.

| File | Description |
|------|-------------|
| `alphabag_parameters.py` | Bag constant B, α_s coupling, quark masses |
| `alphabag_thermodynamics_quarks.py` | Quark thermodynamics with α_s corrections |
| `alphabag_eos.py` | EOS solvers (beta_eq, fixed_yc, CFL phase) |
| `alphabag_compute_tables.py` | **Main script to generate AlphaBag tables** |

### vMIT Model (`vmit_*.py`)
Vector-MIT bag model with vector interactions for quark matter.

| File | Description |
|------|-------------|
| `vmit_parameters.py` | Bag constant, vector coupling G_V |
| `vmit_thermodynamics_quarks.py` | Quark thermodynamics with vector fields |
| `vmit_eos.py` | EOS solvers |
| `vmit_compute_tables.py` | Table generation script |

### ZL Model (`zl_*.py`)
Zhao-Lattimer nucleonic model.

| File | Description |
|------|-------------|
| `zl_parameters.py` | Model coupling constants |
| `zl_thermodynamics_nucleons.py` | Nucleon thermodynamics |
| `zl_eos.py` | EOS solvers |
| `zl_compute_tables.py` | Table generation script |

### ZLvMIT Hybrid Model (`zlvmit_*.py`)
Hadron-quark phase transition using Gibbs, Maxwell, or intermediate constructions.

| File | Description |
|------|-------------|
| `zlvmit_mixed_phase_eos.py` | Mixed phase solver with phase boundaries |
| `zlvmit_hybrid_table_generator.py` | **Main script for hybrid tables** |
| `zlvmit_isentropic.py` | Isentropic trajectory calculations |
| `zlvmit_thermodynamic_derivatives.py` | χ, c_s², susceptibilities |
| `zlvmit_trapped_solvers.py` | Trapped neutrino mode solvers |
| `zlvmit_plot_results.py` | Plotting utilities for hybrid EOS |

---

## Quick Start

### 1. Generate SFHo Tables

```python
from sfho_compute_tables import TableSettings, compute_table
import numpy as np

settings = TableSettings(
    parametrization='sfho',           # 'sfho', 'sfhoy', 'sfhoy_star', '2fam_phi'
    particle_content='nucleons',       # 'nucleons', 'nucleons_hyperons', 'nucleons_hyperons_deltas'
    equilibrium='beta_eq',             # 'beta_eq', 'fixed_yc', 'fixed_yc_ys', 'trapped_neutrinos'
    n_B_values=np.linspace(0.1, 10, 300) * 0.1583,  # Baryon density grid (fm⁻³)
    T_values=np.linspace(0.1, 100, 41),             # Temperature grid (MeV)
    include_photons=True,
    include_electrons=True,
    save_to_file=True,
)

results = compute_table(settings)
```

### 2. Generate AlphaBag Tables

```python
from alphabag_compute_tables import AlphaBagTableSettings, compute_alphabag_table
import numpy as np

settings = AlphaBagTableSettings(
    phase='unpaired',                  # 'unpaired' or 'cfl'
    equilibrium='beta_eq',             # 'beta_eq', 'fixed_yc'
    B_eff=180.0,                       # Bag constant (MeV)
    alpha_s=0.2,                       # Strong coupling
    n_B_values=np.linspace(0.1, 2.0, 100),
    T_values=np.linspace(1, 100, 20),
    save_to_file=True,
)

results = compute_alphabag_table(settings)
```

### 3. Generate Hybrid ZLvMIT Tables

```bash
python zlvmit_hybrid_table_generator.py
```

---

## Equilibrium Modes

| Mode | Description | Constraints |
|------|-------------|-------------|
| `beta_eq` | Beta equilibrium | Weak equilibrium: μ_n = μ_p + μ_e |
| `fixed_yc` | Fixed charge fraction | Y_C = n_C/n_B fixed (+ charge neutrality) |
| `fixed_yc_ys` | Fixed charge & strangeness | Y_C and Y_S fixed |
| `trapped_neutrinos` | Supernova conditions | Y_L = (n_e + n_ν)/n_B fixed |

---

## Output Format

Tables are saved as whitespace-separated `.dat` files with headers:

```
# SFHo EOS Table: sfho, nucleons
# Equilibrium: beta_eq
# Components: photons, electrons
#     n_B    T    sigma  omega  rho  phi  mu_B  mu_C  mu_S  mu_e  mu_nu  P_total  e_total  s_total  Y_C  Y_S  Y_L  converged
  ...
```

Key columns:
- `n_B`: Baryon density (fm⁻³)
- `T`: Temperature (MeV)
- `sigma, omega, rho, phi`: Meson fields (MeV)
- `mu_B, mu_C, mu_S`: Baryon, charge, strangeness chemical potentials (MeV)
- `mu_e, mu_nu`: Electron and neutrino chemical potentials (MeV)
- `P_total, e_total, s_total`: Pressure (MeV/fm³), energy density (MeV/fm³), entropy density (fm⁻³)
- `Y_C, Y_S, Y_L`: Charge, strangeness, lepton fractions

---

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (for plotting)

```bash
pip install numpy scipy matplotlib
```

---

## Sign Conventions

| Quantity | Convention |
|----------|------------|
| μ_C (charge chemical potential) | **Negative** for neutron-rich matter |
| μ_e (electron chemical potential) | Positive |
| Y_C (charge fraction) | n_C/n_B, typically 0 to 0.5 |

---

## Parametrizations

### SFHo Variants

| Name | Description |
|------|-------------|
| `sfho` | Standard SFHo (nucleons only) |
| `sfhoy` | SFHo extended with hyperon couplings (Fortin et al.) |
| `sfhoy_star` | Alternative hyperon couplings |
| `2fam_phi` | Two-families model with φ meson |

### AlphaBag Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| B_eff | 150-200 MeV | Effective bag constant |
| α_s | 0.1-0.4 | Strong coupling constant |
| m_s | 100-150 MeV | Strange quark mass |

---

## References

- Steiner, A. W., Hempel, M., & Fischer, T. (2013). Core-collapse supernova equations of state based on neutron star observations. ApJ, 774, 17.
- Fortin, M., et al. (2018). Neutron star radii and crusts: Uncertainties and unified equations of state. Phys. Rev. C, 97, 035802.

---

## Author

Mirco Guerrini

## License

MIT License
