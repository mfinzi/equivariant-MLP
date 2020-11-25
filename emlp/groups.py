
import numpy as np
from scipy.linalg import expm
from oil.utils.utils import Named

def rel_err(A,B):
    return np.mean(np.abs(A-B))/(np.mean(np.abs(A)) + np.mean(np.abs(B))+1e-7)

class Group(object,metaclass=Named):
    lie_algebra = NotImplemented
    discrete_generators = NotImplemented
    def __init__(self,*args,**kwargs):
        if self.lie_algebra is NotImplemented:
            self.lie_algebra = np.zeros((0,self.d,self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = np.zeros((0,self.d,self.d))
        self.args = args
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
    def samples(self,N):
        return np.stack(self.sample() for _ in range(N))

    def is_unimodular(self):
        unimodular = True
        if self.lie_algebra.shape[0]!=0:
            unimodular &= rel_err(-self.lie_algebra.transpose((0,2,1)),self.lie_algebra)<1e-6
        h = self.discrete_generators
        if h.shape[0]!=0:
            unimodular &= rel_err(h.transpose((0,2,1))@h,np.eye(self.d))<1e-6
        return unimodular

    def __mul__(self,G2):
        return CrossProductGroup(self,G2)
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f"{self.__class__}{list(self.args) if self.args else ''}"
    def __eq__(self,G2): #TODO: check that spans are equal?
        return (self.lie_algebra==G2.lie_algebra).all() and (self.discrete_generators==G2.discrete_generators).all()
    def __hash__(self):
        return hash((self.lie_algebra.tostring(),self.discrete_generators.tostring()))

class CrossProductGroup(Group):
    def __init__(self,G1,G2):
        self.lie_algebra = np.concatenate((G1.lie_algebra,G2.lie_algebra))
        self.discrete_generators = np.concatenate((G1.discrete_generators,G2.discrete_generators))
        self.names = (repr(G1),repr(G2))
        super().__init__()
        
    def __repr__(self):
        return f"{self.names[0]}x{self.names[1]}"

class Trivial(Group): #""" The trivial group G={I} in N dimensions """
    def __init__(self,N):
        self._d = N
        super().__init__(N)

class SO(Group): #""" The special orthogonal group SO(N) in N dimensions"""
    def __init__(self,N):
        self.lie_algebra = np.zeros(((N*(N-1))//2,N,N))
        k=0
        for i in range(N):
            for j in range(i):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,j,i] = -1
                k+=1
        super().__init__(N)

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
        super().__init__(k)

class D(C): #""" The Dihedral group Dk in 2 dimensions"""
    def __init__(self,k):
        super().__init__(k)
        self.discrete_generators = np.concatenate((self.discrete_generators,np.array([[[-1,0],[0,1]]])))

class Scaling(Group):
    def __init__(self,N):
        self.lie_algebra = np.eye(N)[None]
        super().__init__(N)

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

class Symplectic(Group):
    def __init__(self,m):
        self.lie_algebra = np.zeros((m*(2*m+1),2*m,2*m))
        k=0
        for i in range(m): # block diagonal elements
            for j in range(m):
                self.lie_algebra[k,i,j] = 1
                self.lie_algebra[k,m+j,m+i] = -1
                k+=1
        for i in range(m):
            for j in range(i+1):
                self.lie_algebra[k,m+i,j] = 1
                self.lie_algebra[k,m+j,i] = 1
                k+=1
                self.lie_algebra[k,i,m+j] = 1
                self.lie_algebra[k,j,m+i] = 1
                k+=1
        super().__init__(m)
