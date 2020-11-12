
import numpy as np
from scipy.linalg import expm

class Group(object):
    lie_algebra = NotImplemented
    discrete_generators = NotImplemented
    def __init__(self):
        if self.lie_algebra is NotImplemented:
            self.lie_algebra = np.zeros((0,self.d,self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = np.zeros((0,self.d,self.d))
    def exp(self,A):
        return expm(A)

    @property
    def d(self):
        if self.lie_algebra is not NotImplemented:
            return self.lie_algebra.shape[-1]
        if self.discrete_generators is not NotImplemented:
            return self.discrete_generators.shape[-1]
        return self._d

    def sample(self):
        g = np.eye(self.d)
        if self.lie_algebra.shape[0]:
            z = np.random.randn(self.lie_algebra.shape[0])
            A = (z[:,None,None]*self.lie_algebra).sum(0)
            g = g@self.exp(A)
        for h in self.discrete_generators:
            k = np.random.randint(-5,5)
            g = g@np.linalg.matrix_power(h,k)
        return g

    def __mul__(self,G2):
        return CrossProductGroup(self,G2)

class CrossProductGroup(Group):
    def __init__(self,G1,G2):
        self.lie_algebra = np.concatenate((G1.lie_algebra,G2.lie_algebra))
        self.discrete_generators = np.concatenate((G1.discrete_generators,G2.discrete_generators))

class Trivial(Group): #""" The trivial group G={I} in N dimensions """
    def __init__(self,N):
        self._d = N
        super().__init__()

class SO(Group): #""" The special orthogonal group SO(N) in N dimensions"""
    def __init__(self,N):
        self.lie_algebra = np.zeros(((N*(N-1))//2,N,N))
        k=0
        for i in range(N):
            for j in range(i):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,j,i] = -1
                k+=1
        super().__init__()

class O(SO): #""" The Orthogonal group O(N) in N dimensions"""
    def __init__(self,N):
        self.discrete_generators = np.eye(N)[None]
        self.discrete_generators[0,0,0]=-1
        super().__init__(N)

class C(Group): #""" The Cyclic group Ck in 2 dimensions"""
    def __init__(self,k):
        theta = 2*np.pi/k
        self.discrete_generators = np.zeros((1,2,2))
        self.discrete_generators[0,:,:] = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        super().__init__()

class D(C): #""" The Dihedral group Dk in 2 dimensions"""
    def __init__(self,k):
        super().__init__(k)
        self.discrete_generators = np.concatenate((self.discrete_generators,np.array([[[-1,0],[0,1]]])))

class Scaling(Group):
    def __init__(self,N):
        self.lie_algebra = np.eye(N)[None]
        super().__init__()

class Parity(Group): #""" The spacial parity group in 1+3 dimensions"""
    discrete_generators = -np.eye(4)[None]
    discrete_generators[0,0,0] = 1

class TimeReversal(Group): #""" The time reversal group in 1+3 dimensions"""
    discrete_generators = np.eye(4)[None]
    discrete_generators[0,0,0] = -1

class SO13p(Group): #""" The component of Lorentz group connected to identity"""
    lie_algebra = np.zeros((6,4,4))
    lie_algebra[3:,1:,1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i,1+i,0] = lie_algebra[i,0,1+i] = 1

SO13 = SO13p()*TimeReversal() # The parity preserving Lorentz group

Lorentz = O13 = SO13p()*Parity()*TimeReversal() # The full lorentz group with P,T transformations
