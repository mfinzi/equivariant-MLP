import numpy as np
#import torch
import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax import device_put
import collections,itertools
from functools import lru_cache as cache
from emlp_jax.utils import disk_cache
import scipy as sp
import scipy.linalg
import functools
import random
from emlp_jax.equivariant_subspaces import TensorRep,Rep,SumRep
from emlp_jax.linear_operator_jax import LinearOperator
import logging
import copy
import math

class ProductGroupTensorRep(Rep): # Eventually will need to handle reordering to canonical G1,G2, etc (from hash?)
    # TO be used like (T(0) + T(1))(SO(3))*T(1)(S(5)) -> T(2)(SO(3))
    def __init__(self,rep_dict):
        assert len(rep_dict)>1, "Not product rep?"
        self.reps = rep_dict
    def rho(self,Ms): 
        rhos = [rep.rho(Ms[G]) for (G,rep) in self.reps.items()]
        return functools.reduce(jnp.kron,rhos,1)
    def drho(self,As):
        drhos = [rep.drho(As[G]) for (G,rep) in self.reps.items()]
        raise functools.reduce(kronsum,drhos,0)
    @property
    def T(self):
        return ProductGroupTensorRep({G:rep.T for G,rep in self.reps.items()})
    def __eq__(self, other): 
        if not isinstance(other,ProductGroupTensorRep): return False
        return len(self.reps)==len(other.reps) \
            and all(Ga==Gb for Ga,Gb in zip(self.reps,other.reps)) \
            and all(ra==rb for ra,rb in zip(self.reps.values(),other.reps.values())) \
            

    def __hash__(self):
        return hash(tuple(self.reps.items()))
    def size(self):
        return math.prod([rep.size() for rep in self.reps.values()])

    def __mul__(self, other): 
        #TODO: worry about ordering of representation differing from dict order when new elems are added
        out = copy.copy(self) #TODO: check deepcopy not requried
        if isinstance(other,ProductGroupTensorRep):
            for Gb,rep in other.reps.items():
                if Gb in copy.reps: copy.reps[Gb]=copy.reps[Gb]*rep
                else: copy.reps[Gb] = rep
            return copy
        elif isinstance(other,TensorRep):
            copy.reps[other.G] = copy.reps[other.G]*other
            return copy
        else: return NotImplemented

    def __rmul__(self, other):
        out = copy.copy(self)
        if isinstance(other,ProductGroupTensorRep):
            for Gb,rep in other.reps.items():
                if Gb in copy.reps: copy.reps[Gb]=rep*copy.reps[Gb]
                else: copy.reps[Gb] = rep
            return copy
        elif isinstance(other,TensorRep):
            copy.reps[other.G] = copy.reps[other.G]*other
            return copy
        else: return NotImplemented

    def __str__(self):
        return "⊗".join([str(rep) for rep in self.reps.values()])
    def symmetric_basis(self): 
        raise NotImplementedError

    def symmetric_projector(self):
        projectors = [rep.symmetric_projector() for rep in self.reps.values()]
        #print([p.shape for p in projectors])
        #P1 = self.rep1.symmetric_projector()
        #P2 = self.rep2.symmetric_projector()
        return lazy_kron(projectors)
@jit
def kronsum(A,B):
    return jnp.kron(A,jnp.eye(B.shape[-1])) + jnp.kron(jnp.eye(A.shape[-1]),B)

class lazy_kron(LinearOperator):
    def __init__(self,Ms):
        self.Ms = Ms
        self.shape = math.prod([Mi.shape[0] for Mi in Ms]), math.prod([Mi.shape[1] for Mi in Ms])
        #self.dtype=Ms[0].dtype

    def _matvec(self,v):
        #print(v.shape)
        ev = v.reshape(tuple(Mi.shape[-1] for Mi in self.Ms))
        for i,M in enumerate(self.Ms):
            ev_front = jnp.moveaxis(ev,i,0)
            Mev_front = (M@ev_front.reshape((M.shape[-1],-1))).reshape(ev_front.shape)
            ev = jnp.moveaxis(Mev_front,0,i)
        return ev.reshape(-1)
    def _adjoint(self):
        return lazy_kron([Mi.T for Mi in self.Ms])
        # for i in range(p,p+q):
        #     eV = np.moveaxis(np.dot(MinvT,np.moveaxis(eV,i,0)),0,i)
        # return eV.reshape(*V.shape)
        # eV = V.reshape(self.M1.shape[-1],self.M2.shape[-1])
        # PV = (self.M2@(self.M1@eV).T).T
        # return PV.reshape(-1)
    # def _adjoint(self):
    #     return rho_lazy(self.M1.T,self.M2.T)