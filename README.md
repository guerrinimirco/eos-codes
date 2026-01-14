# Nuclear Equation of State (EOS) Table Generator

A Python library for calculating finite temperature Equations of State (EOS) for astrophysical applications (including neutron star matter, supernovae, and compact object mergers) and heavy ion collisions.

The details of the models are reported in M. Guerrini PhD Thesis Chapter 2 (soon available) and references therein.

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
| `general_bose_integrals.py` | Bose-Einstein integrals for bosons |
| `general_thermodynamics_leptons.py` | Electron, muon, neutrino thermodynamics |
| `general_eos_solver.py` | Generic EOS solver framework (old approach, not used in new files) |
| `general_plotting_info.py` | Plotting utilities and styles (old file, will be updated in future) |

### SFHo Model (`sfho_*.py`)
Relativistic Mean Field (RMF) hadronic EOS with σ-ω-ρ-φ meson exchange. 
See Guerrini PhD Thesis for a summary of the model and Fortin et al. (2017), Steiner et al. (2013) for details.

| File | Description |
|------|-------------|
| `sfho_parameters.py` | Model parameterizations (some default ones: sfho, sfhoy, sfhoy_star, 2fam variants) |
| `sfho_thermodynamics_hadrons.py` | Hadron thermodynamics with meson mean fields and pseudoscalar mesons |
| `sfho_eos.py` | EOS solvers (beta_eq, fixed_yc, fixed_yc_ys, trapped_neutrinos) |
| `sfho_compute_tables.py` | **Main script to generate SFHo tables** |
| `sfho_nuclear_saturation_properties.py` | Nuclear matter properties at saturation |
| `sfho_compare_with_compose.py` | Validation against CompOSE tables (not updated) |

### AlphaBag Model (`alphabag_*.py`)
α_s-corrected MIT bag model for quark matter with perturbative QCD correct  ions.
See Guerrini PhD Thesis for a summary of the model and references therein.

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
| `vmit_compute_tables.py` | **Main script to generate vMIT tables**  |

### ZL Model (`zl_*.py`)
Zhao-Lattimer nucleonic model.
See Guerrini PhD Thesis for a summary of the model and Zhao and Lattimer (2020) for details.

| File | Description |
|------|-------------|
| `zl_parameters.py` | Model coupling constants |
| `zl_thermodynamics_nucleons.py` | Nucleon thermodynamics |
| `zl_eos.py` | EOS solvers |
| `zl_compute_tables.py` | **Main script to generate ZL tables** |

### ZLvMIT Hybrid Model (`zlvmit_*.py`)
Hadron-quark phase transition using Gibbs, Maxwell, or intermediate constructions.
See Guerrini PhD Thesis for a summary of the model and Constantinou et al. (2025) for details.

| File | Description |
|------|-------------|
| `zlvmit_mixed_phase_eos.py` | Mixed phase solver with phase boundaries |
| `zlvmit_hybrid_table_generator.py` | **Main script for hybrid tables** |
| `zlvmit_isentropic.py` | Isentropic trajectory calculations |
| `zlvmit_thermodynamic_derivatives.py` | χ, c_s², susceptibilities (still work in progress)|
| `zlvmit_trapped_solvers.py` | Trapped neutrino mode solvers (will be unified with mixed_phase_eos.py) |
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
modify the 

# =============================================================================
# USER CONFIGURATION - EDIT THESE VALUES
# =============================================================================

and run

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


---

## Conventions
Y_S is the strangeness fraction with S=number of strange quarks (S=1 for hyperons, note that sometimes in the literature S=-1 is used )
Y_C is the electric charge fraction of hadrons and quarks (B,C,S could be remapped in u,d,s or B,I,S where I=isospin)

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

## General informations

This is a recent (2026) Python rewrite of some of the code I wrote in Mathematica and Python for my Master's thesis (2022) and during my PhD (2023-2026). For the moment, the documentation is poor and the usability is not very user-friendly, but please contact me if you want to use them and need help.

mirco.guerrini@unife.it
https://inspirehep.net/authors/2775420

My main collaborators:
- A. Drago (U. Ferrara) - PhD supervisor
- G. Pagliara (U. Ferrara) - PhD supervisor
- C. Constantinou (ECT* Trento) 
- A. Lavagno (Politecnico di Torino)
- T. Zhao 
- S. Han


## Future plans
- general hadron-quark mixed phase for SFHo-alphaBag with hyperons and deltas
- DD2+vMIT mixed phase with hyperons
- rewrite in python my codes for adding crusts (subnuclear phases) and solve TOV equations
- response functions (heat capacities, susceptibilities, speed of sound, etc.)
- rewrite in python my codes for nucleation of quark matter (see my PhD Thesis)
- Hydrodynamic hadron-quark matter conversion