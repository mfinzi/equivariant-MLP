
import numpy as np
from scipy.linalg import expm
from oil.utils.utils import Named,export
import jax
import jax.numpy as jnp
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
import emlp_jax.equivariant_subspaces as equivariant_subspaces
from jax import jit,vmap
from functools import partial
import logging

def rel_err(A,B):
    return jnp.mean(jnp.abs(A-B))/(jnp.mean(jnp.abs(A)) + jnp.mean(jnp.abs(B))+1e-6)

class Group(object,metaclass=Named):
    lie_algebra = NotImplemented
    discrete_generators = NotImplemented
    z_scale=None # For scale noise for sampling elements
    def __init__(self,*args,**kwargs):
        if self.lie_algebra is NotImplemented:
            self.lie_algebra = np.zeros((0,self.d,self.d))
        if self.discrete_generators is NotImplemented:
            self.discrete_generators = np.zeros((0,self.d,self.d))
        self.args = args
        self.lie_algebra = jax.device_put(self.lie_algebra)
        self.discrete_generators = jax.device_put(self.discrete_generators)

        # Set orthogonal flag
        self.is_orthogonal = True
        if self.lie_algebra.shape[0]!=0:
            self.is_orthogonal &= rel_err(-self.lie_algebra.transpose((0,2,1)),self.lie_algebra)<1e-6
        h = self.discrete_generators
        if h.shape[0]!=0:
            self.is_orthogonal &= rel_err(h.transpose((0,2,1))@h,jnp.eye(self.d))<1e-6
    def exp(self,A):
        return expm(A)
    def num_constraints(self):
        return self.lie_algebra.shape[0]+self.discrete_generators.shape[0]
    @property
    def d(self):
        if self.lie_algebra is not NotImplemented:
            return self.lie_algebra.shape[-1]
        if self.discrete_generators is not NotImplemented:
            return self.discrete_generators.shape[-1]
        return self._d

    def sample(self):
        return self.samples(1)[0]

    def samples(self,N):
        z = np.random.randn(N,self.lie_algebra.shape[0])
        if self.z_scale is not None:
            z*= self.z_scale
        k = np.random.randint(-5,5,size=(N,self.discrete_generators.shape[0]))
        return noise2samples(z,k,self.lie_algebra,self.discrete_generators)

    def check_valid_group_elems(self,g):
        return True

    def __and__(self,G2):
        return CombinedGenerators(self,G2)
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f"{self.__class__}{list(self.args) if self.args else ''}"
    def __eq__(self,G2): #TODO: check that spans are equal?
        return repr(self)==repr(G2)
        # if self.lie_algebra.shape!=G2.lie_algebra.shape or \
        #     self.discrete_generators.shape!=G2.discrete_generators.shape:
        #     return False
        # return  (self.lie_algebra==G2.lie_algebra).all() and (self.discrete_generators==G2.discrete_generators).all()
    def __hash__(self):
        return hash(repr(self))
        # algebra = jax.device_get(self.lie_algebra).tobytes()
        # gens = jax.device_get(self.discrete_generators).tobytes()
        # return hash((algebra,gens,self.lie_algebra.shape,self.discrete_generators.shape))
    def __lt__(self, other):
        return hash(self) < hash(other) #For sorting purposes only

@jit
def matrix_power_simple(M,n):
    out = jnp.eye(M.shape[-1])
    body = lambda Mn: jax.lax.fori_loop(0,Mn[1],lambda i,g: Mn[0]@g,out)
    out = jax.lax.cond(n<0,(jnp.linalg.inv(M),-n),body,(M,n),body)
    return out
    
@jit
def noise2sample(z,ks,lie_algebra,discrete_generators):
    """ [zs (D,)] [ks (M,)] [lie_algebra (D,d,d)] [discrete_generators (M,d,d)] """
    g = jnp.eye(lie_algebra.shape[-1])
    if lie_algebra.shape[0]:
        A = (z[:,None,None]*lie_algebra).sum(0)
        g = g@jax.scipy.linalg.expm(A)
    for i in range(discrete_generators.shape[0]):
        g = g@matrix_power_simple(discrete_generators[i],ks[i])#jnp.linalg.matrix_power(discrete_generators[i],ks[i])
    return g

@jit
def noise2samples(zs,ks,lie_algebra,discrete_generators):
    return vmap(noise2sample,(0,0,None,None),0)(zs,ks,lie_algebra,discrete_generators)


class CombinedGenerators(Group):
    def __init__(self,G1,G2):
        self.lie_algebra = np.concatenate((G1.lie_algebra,G2.lie_algebra))
        self.discrete_generators = np.concatenate((G1.discrete_generators,G2.discrete_generators))
        self.names = (repr(G1),repr(G2))
        super().__init__()
        
    def __repr__(self):
        return f"{self.names[0]}&{self.names[1]}"

class WreathProduct(Group):
    def __init__(self,G1,G2):
        raise NotImplementedError

class SemiDirectProduct(Group):
    def __init__(self,G1,G2):
        raise NotImplementedError

@export
class Trivial(Group): #""" The trivial group G={I} in N dimensions """
    def __init__(self,N):
        self._d = N
        super().__init__(N)
@export
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
@export
class O(SO): #""" The Orthogonal group O(N) in N dimensions"""
    def __init__(self,N):
        self.discrete_generators = np.eye(N)[None]
        self.discrete_generators[0,0,0]=-1
        super().__init__(N)
@export
class C(Group): #""" The Cyclic group Ck in 2 dimensions"""
    def __init__(self,k):
        theta = 2*np.pi/k
        self.discrete_generators = np.zeros((1,2,2))
        self.discrete_generators[0,:,:] = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
        super().__init__(k)
@export
class D(C): #""" The Dihedral group Dk in 2 dimensions"""
    def __init__(self,k):
        super().__init__(k)
        self.discrete_generators = np.concatenate((self.discrete_generators,np.array([[[-1,0],[0,1]]])))
@export
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

@export
class SO13p(Group): #""" The component of Lorentz group connected to identity"""
    lie_algebra = np.zeros((6,4,4))
    lie_algebra[3:,1:,1:] = SO(3).lie_algebra
    for i in range(3):
        lie_algebra[i,1+i,0] = lie_algebra[i,0,1+i] = 1.

    # Adjust variance for samples along boost generators. For equivariance checks
    # the exps for high order tensors can get very large numbers
    z_scale = np.array([.3,.3,.3,1,1,1])
@export
class SO13(SO13p):
    discrete_generators = -np.eye(4)[None]

@export
class O13(SO13p):
    discrete_generators = np.eye(4)[None] +np.zeros((2,1,1))
    discrete_generators[0] *= -1
    discrete_generators[1,0,0] = -1
@export
class Lorentz(O13): pass

@export
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
@export
class Permutation(Group):
    def __init__(self,n):
        self.discrete_generators = np.zeros((n-1,n,n))
        self.discrete_generators += np.eye(n)
        self.discrete_generators[:,0,0]=0
        for i in range(n-1):
            self.discrete_generators[i,0,i+1]=1
            self.discrete_generators[i,i+1,0]=1
            self.discrete_generators[i,i+1,i+1]=0
        super().__init__(n)

@export
class S(Permutation): pass #Alias permutation group with Sn.

@export
class DiscreteTranslation(Group):
    def __init__(self,n):
        self.discrete_generators = np.roll(np.eye(n),1,axis=1)[None]
        super().__init__(n)

@export
class U(Group): # Of dimension n^2
    def __init__(self,n):
        lie_algebra_real = np.zeros((n**2,n,n))
        lie_algebra_imag = np.zeros((n**2,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k,i,j] = 1
                lie_algebra_real[k,j,i] = -1
                k+=1
                # symmetric imaginary generators
                lie_algebra_imag[k,i,j] = 1
                lie_algebra_imag[k,j,i] = 1
                k+=1
        for i in range(n):
            # diagonal imaginary generators
            lie_algebra_imag[k,i,i] = 1
            k+=1
        I,J = np.eye(2),np.array([[0,1],[-1,0]])
        self.lie_algebra = np.kron(lie_algebra_real,I)+np.kron(lie_algebra_imag,J)
        super().__init__(n)
@export
class SU(Group): # Of dimension n^2-1
    def __init__(self,n):
        if n==1: return Trivial(1)
        lie_algebra_real = np.zeros((n**2-1,n,n))
        lie_algebra_imag = np.zeros((n**2-1,n,n))
        k=0
        for i in range(n):
            for j in range(i):
                # Antisymmetric real generators
                lie_algebra_real[k,i,j] = 1
                lie_algebra_real[k,j,i] = -1
                k+=1
                # symmetric imaginary generators
                lie_algebra_imag[k,i,j] = 1
                lie_algebra_imag[k,j,i] = 1
                k+=1
        for i in range(n-1):
            # diagonal traceless imaginary generators
            lie_algebra_imag[k,i,i] = 1
            for j in range(n):
                if i==j: continue
                lie_algebra_imag[k,j,j] = -1/(n-1)
            k+=1
        I,J = np.eye(2),np.array([[0,1],[-1,0]])
        self.lie_algebra = np.kron(lie_algebra_real,I)+np.kron(lie_algebra_imag,J)
        super().__init__(n)

