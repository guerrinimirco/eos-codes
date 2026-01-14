"""
alphabag_parameters.py
======================
Parameter dataclasses for the αBag (AlphaBag) quark EOS model.

The αBag model includes:
- Current quark masses (u, d ~ massless, s ~ 150 MeV)
- Bag constant B (confining pressure)
- Perturbative QCD coupling constant α_s

This is different from vMIT which uses a vector field approach.
Here perturbative corrections are parametrized by α_s.

References:
- M. G. Alford et al. Phys. Rev. D 67.7 (2003)
- T. Fischer et al. Astrophys. J. Suppl. 194:39 (2011)
- M. Guerrini PhD Thesis (2026)
"""
from dataclasses import dataclass


@dataclass
class AlphaBagParams:
    """
    Parameters for the αBag quark EOS.
    
    Attributes:
        name: Parameter set identifier
        m_u: Up quark mass (MeV), typically ~0 (massless)
        m_d: Down quark mass (MeV), typically ~0 (massless)  
        m_s: Strange quark mass (MeV), typically ~150 MeV
        alpha: QCD coupling constant α_s (dimensionless), typical 0.2-0.5
        B4: Bag constant B^(1/4) (MeV), typical 145-180 MeV
        B: Bag constant B (MeV⁴) = B4⁴
    """
    name: str = "alphabag_default"
    m_u: float = 0.0         # MeV (up quark mass, treated as massless)
    m_d: float = 0.0         # MeV (down quark mass, treated as massless)
    m_s: float = 150.0       # MeV (strange quark mass)
    alpha: float = 0.3       # dimensionless (QCD coupling α_s)
    B4: float = 165.0        # MeV (bag constant B^1/4)
    
    @property
    def B(self) -> float:
        """Bag constant B = (B^1/4)^4 in MeV⁴."""
        return self.B4**4


def get_alphabag_default() -> AlphaBagParams:
    """Get default αBag parameter set."""
    return AlphaBagParams(name="alphabag_default")


def get_alphabag_custom(
    m_u: float = 0.0, m_d: float = 0.0, m_s: float = 150.0,
    alpha: float = 0.3, B4: float = 165.0, name: str = "alphabag_custom"
) -> AlphaBagParams:
    """
    Create custom αBag parameter set.
    
    Args:
        m_u, m_d, m_s: Quark masses (MeV)
        alpha: QCD coupling constant α_s (dimensionless)
        B4: Bag constant B^(1/4) (MeV)
        name: Parameter set name
        
    Returns:
        AlphaBagParams with specified values
    """
    return AlphaBagParams(
        name=name,
        m_u=m_u, m_d=m_d, m_s=m_s,
        alpha=alpha, B4=B4
    )


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    from general_physics_constants import hc3
    
    print("αBag Parameters Test")
    print("=" * 50)
    
    params = get_alphabag_default()
    print(f"\nDefault parameters: {params.name}")
    print(f"  m_u   = {params.m_u} MeV (massless)")
    print(f"  m_d   = {params.m_d} MeV (massless)")
    print(f"  m_s   = {params.m_s} MeV")
    print(f"  α_s   = {params.alpha}")
    print(f"  B^1/4 = {params.B4} MeV")
    print(f"  B     = {params.B:.4e} MeV⁴")
    print(f"  B     = {params.B/hc3:.4f} MeV/fm³")
    
    print("\n✓ All OK")
