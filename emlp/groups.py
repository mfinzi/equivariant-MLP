
import numpy as np
from scipy.linalg import expm

class Group(object):
    lie_algebra = None
    discrete_generators = None

    def exp(self,A):
        return expm(A)

    @property
    def d(self):
        if self.lie_algebra is not None:
            return self.lie_algebra.shape[-1]
        if self.discrete_generators is not None:
            return self.discrete_generators.shape[-1]
        return self._d

    def sample(self):
        g = np.eye(self.d)
        if self.lie_algebra is not None:
            z = np.random.randn(self.lie_algebra.shape[0])
            A = (z[:,None,None]*self.lie_algebra).sum(0)
            g = g@self.exp(A)
        if self.discrete_generators is not None:
            for h in self.discrete_generators:
                k = np.random.randint(-5,5)
                g = g@np.linalg.matrix_power(h,k)
        return g

    def __mul__(self,G2):
        return CrossProductGroup(self,G2)

class CrossProductGroup(Group):
    def __init__(self,G1,G2):
        if G1.lie_algebra is not None and G2.lie_algebra is not None:
            self.lie_algebra = np.concatenate([G1.lie_algebra,G2.lie_algebra])
        else: 
            self.lie_algebra=G1.lie_algebra if G1.lie_algebra is not None else G2.lie_algebra
        if G1.discrete_generators is not None and G2.discrete_generators is not None:
            self.discrete_generators = np.concatenate([G1.discrete_generators,G2.discrete_generators])
        else: 
            self.discrete_generators=G1.discrete_generators if G1.discrete_generators is not None\
                else G2.discrete_generators

class Trivial(Group):
    def __init__(self,N):
        self._d = N

class SO(Group):
    def __init__(self,N):
        self.lie_algebra = np.zeros(((N*(N-1))//2,N,N))
        k=0
        for i in range(N):
            for j in range(i):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,j,i] = -1
                k+=1

class O(SO):
    def __init__(self,N):
        super().__init__(N)
        self.discrete_generators = -np.eye(N)[None]

class Scaling(Group):
    def __init__(self,N):
        self.lie_algebra = np.eye(N)[None]

class Parity(Group):
    discrete_generators = -np.eye(4)[None]
    discrete_generators[0,0,0] = 1

class TimeReversal(Group):
    discrete_generators = np.eye(4)[None]
    discrete_generators[0,0,0] = -1

class SO13p(Group):
    lie_algebra = np.zeros((6,4,4))
    lie_algebra[3:,1:,1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i,1+i,0] = lie_algebra[i,0,1+i] = 1

SO13 = SO13p()*TimeReversal()

Lorentz = O13 = SO13p()*Parity()*TimeReversal()
