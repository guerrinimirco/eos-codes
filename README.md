# Research Python Codes

This repository contains Python codes for calculating Equations of State (EOS) for astrophysical simulations.

## Contents

The project includes solvers for different nuclear models:
*   **SFHo**: Relativistic Mean Field model (Nucleons, Hyperons, Deltas).
*   **ZL**: ZL nucleonic model.
*   **vMIT**: Vector-MIT bag model for quark matter.
*   **ZLvMIT**: Hybrid construction mixing ZL and vMIT phases (Gibbs/Maxwell/Intermediate constructions).

## Structure

*   `general_*.py`: Shared modules for thermodynamics, particle definitions, and constants.
*   `sfho_*.py`: SFHo model implementation.
*   `zl_*.py`: ZL model implementation.
*   `vmit_*.py`: vMIT model implementation.
*   `zlvmit_*.py`: Hybrid EOS implementation and phase boundary finders.

## Usage

Scripts are generally run directly. For example, to generate hybrid EOS tables:
```bash
python zlvmit_orchestrator.py
```
