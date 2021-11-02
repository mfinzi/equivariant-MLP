import jax.numpy as jnp
from emlp.groups import SO,S
import emlp
from emlp.reps.representation import Base,ScalarRep,Rep
from emlp.groups import Z,S,SO,Group,DirectProduct
import numpy as np
from emlp.utils import export

__all__ = ["V","Vector", "Scalar"]

V=Vector= Base()  #: Alias V or Vector for an instance of the Base representation of a group

Scalar = ScalarRep()#: An instance of the Scalar representation, equivalent to V**0

@export
def T(p,q=0,G=None):
    """ A convenience function for creating rank (p,q) tensors."""
    return (V**p*V.T**q)(G)

@export
class Restricted(Rep):
    """ Restricted Representation: a base representation of one
         of the subgroups of G = G1 x G2 x ... x Gn

        Args:
            G (Group): Product group G = G1 x G2 x ... x Gn
            subgroup_id (int): specifying G1,..., Gn"""

    def __init__(self,G,subgroup_id):
        assert isinstance(G,DirectProduct), "Restricted representation is only for direct product groups"
        self.G = G
        self.index = subgroup_id
        
    def __str__(self):
        return "V_"+str(self.G).split('×')[self.index]
    def size(self):
        return self.G._Gs[self.index].d
    def rho(self,M): 
        # Given that M is a LazyKron object, we can just get the argument
        return M.Ms[self.index]
    def drho(self,A):
        return A.Ms[self.index]
    def __call__(self,G):
        # adding this will probably not be necessary in a future release,
        # necessary now because rep is __call__ed in nn.EMLP constructor
        assert self.G==G
        return self

@export
class PseudoScalar(Rep):
    """ Pseudo-Scalar representation. Transforms by sign(det(g)).
        Can be used to construct Pseudo-Vectors, Pseudo-Tensors etc
        by multiplying by the PseudoScalar """
    def __init__(self,G=None):
        self.G=G
    def __str__(self):
        return "P"
    def rho(self,M):
        sign = jnp.linalg.slogdet(M@jnp.eye(M.shape[0]))[0]
        return sign*jnp.eye(1)
    def __call__(self,G):
        return PseudoScalar(G)

@export
class SO2Irreps(Rep):
    """ (Real) Irreducible representations of SO2 """
    def __init__(self,order):
        assert order>0, "Use Scalar for 𝜓₀"
        self.G=SO(2)
        self.order = order
    def rho(self,M):
        return jnp.linalg.matrix_power(M,self.order)
    def __str__(self):
        number2sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return f"𝜓{self.order}".translate(number2sub)
