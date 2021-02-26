from .linear_operator_jax import LinearOperator
import jax.numpy as jnp
import numpy as np
from jax import jit
import jax
from functools import reduce
from .utils import prod as product

class Lazy(LinearOperator):
    def __init__(self,dense_matrix):
        self.A = dense_matrix
        self.shape = self.A.shape
        self.dtype = self.A.dtype
    def _matmat(self,V):
        return self.A@V
    def _matvec(self,v):
        return self.A@v
    def _rmatmat(self,V):
        return self.A.T@V
    def _rmatvec(self,v):
        return self.A.T@v
    def to_dense(self):
        return self.A

class I(LinearOperator):
    def __init__(self,d):
        self.shape = (d,d)
    def _matmat(self,V): #(c,k)
        return V
    def _matvec(self,V):
        return V
    def _adjoint(self):
        return self
    def invT(self):
        return self

class LazyKron(LinearOperator):

    def __init__(self,Ms):
        self.Ms = Ms
        self.shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        #self.dtype=Ms[0].dtype
        self.dtype=jnp.dtype('float32')

    def _matvec(self,v):
        return self._matmat(v).reshape(-1)
    def _matmat(self,v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms],-1)
        for i,M in enumerate(self.Ms):
            ev_front = jnp.moveaxis(ev,i,0)
            Mev_front = (M@ev_front.reshape(M.shape[-1],-1)).reshape(M.shape[0],*ev_front.shape[1:])
            ev = jnp.moveaxis(Mev_front,0,i)
        return ev.reshape(self.shape[0],ev.shape[-1])
    def _adjoint(self):
        return LazyKron([Mi.T for Mi in self.Ms])
    def invT(self):
        return LazyKron([M.invT() for M in self.Ms])
    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M,LinearOperator) else M for M in self.Ms]
        return reduce(jnp.kron,Ms)
    def __new__(cls,Ms):
        if len(Ms)==1: return Ms[0]
        return super().__new__(cls)

@jit
def kronsum(A,B):
    return jnp.kron(A,jnp.eye(B.shape[-1])) + jnp.kron(jnp.eye(A.shape[-1]),B)

class LazyKronsum(LinearOperator):
    
    def __init__(self,Ms):
        self.Ms = Ms
        self.shape = product([Mi.shape[0] for Mi in Ms]), product([Mi.shape[1] for Mi in Ms])
        #self.dtype=Ms[0].dtype
        self.dtype=jnp.dtype('float32')

    def _matvec(self,v):
        return self._matmat(v).reshape(-1)

    def _matmat(self,v):
        ev = v.reshape(*[Mi.shape[-1] for Mi in self.Ms],-1)
        out = 0*ev
        for i,M in enumerate(self.Ms):
            ev_front = jnp.moveaxis(ev,i,0)
            Mev_front = (M@ev_front.reshape(M.shape[-1],-1)).reshape(M.shape[0],*ev_front.shape[1:])
            out += jnp.moveaxis(Mev_front,0,i)
        return out.reshape(self.shape[0],ev.shape[-1])
        
    def _adjoint(self):
        return LazyKronsum([Mi.T for Mi in self.Ms])
    def to_dense(self):
        Ms = [M.to_dense() if isinstance(M,LinearOperator) else M for M in self.Ms]
        return reduce(kronsum,Ms)
    def __new__(cls,Ms):
        if len(Ms)==1: return Ms[0]
        return super().__new__(cls)
## could also be implemented as follows, but the fusing the sum into a single linearOperator is faster
# def lazy_kronsum(Ms):
#     n = len(Ms)
#     lprod = np.cumprod([1]+[mi.shape[-1] for mi in Ms])
#     rprod = np.cumprod([1]+[mi.shape[-1] for mi in reversed(Ms)])[::-1]
#     return reduce(lambda a,b: a+b,[lazy_kron([I(lprod[i]),Mi,I(rprod[i+1])]) for i,Mi in enumerate(Ms)])

class ConcatLazy(LinearOperator):
    """ Produces a linear operator equivalent to concatenating
        a collection of matrices Ms along axis=0 """
    def __init__(self,Ms):
        self.Ms = Ms
        assert all(M.shape[0]==Ms[0].shape[0] for M in Ms),\
             f"Trying to concatenate matrices of different sizes {[M.shape for M in Ms]}"
        self.shape = (sum(M.shape[0] for M in Ms),Ms[0].shape[1])

    def _matmat(self,V):
        return jnp.concatenate([M@V for M in self.Ms],axis=0)
    def _rmatmat(self,V):
        Vs = jnp.split(V,len(self.Ms))
        return sum([self.Ms[i].T@Vs[i] for i in range(len(self.Ms))])
    def to_dense(self):
        dense_Ms = [M.to_dense() if isinstance(M,LinearOperator) else M for M in self.Ms]
        return jnp.concatenate(dense_Ms,axis=0)
    
class LazyDirectSum(LinearOperator):
    def __init__(self,Ms,multiplicities=None):
        self.Ms = [jax.device_put(M.astype(np.float32)) if isinstance(M,(np.ndarray,jnp.ndarray)) else M for M in Ms]
        self.multiplicities = [1 for M in Ms] if multiplicities is None else multiplicities
        self.shape = (sum(Mi.shape[0]*c for Mi,c in zip(Ms,multiplicities)),
                      sum(Mi.shape[0]*c for Mi,c in zip(Ms,multiplicities)))
        #self.dtype=Ms[0].dtype
        self.dtype=jnp.dtype('float32')

    def _matvec(self,v):
        return self._matmat(v.reshape(v.shape[0],-1)).reshape(-1)

    def _matmat(self,v): # (n,k)
        return lazy_direct_matmat(v,self.Ms,self.multiplicities)
    def _adjoint(self):
        return LazyDirectSum([Mi.T for Mi in self.Ms])
    def invT(self):
        return LazyDirectSum([M.invT() for M in self.Ms])
    def to_dense(self):
        Ms_all = [self.Ms[i] for i in range(len(self.Ms)) for _ in range(self.multiplicities[i])]
        Ms_all = [Mi.to_dense() if isinstance(Mi,LinearOperator) else Mi for Mi in Ms_all]
        return jax.scipy.linalg.block_diag(*Ms_all)
    def __new__(cls,Ms,multiplicities=None):
        if len(Ms)==1 and multiplicities is None: return Ms[0]
        return super().__new__(cls)
        
def lazy_direct_matmat(v,Ms,mults):
    n,k = v.shape
    i=0
    y = []
    for M, multiplicity in zip(Ms,mults):
        if not M.shape[-1]: continue
        i_end = i+multiplicity*M.shape[-1]
        elems = M@v[i:i_end].T.reshape(-1,M.shape[-1]).T
        y.append(elems.T.reshape(k,multiplicity*M.shape[0]).T)
        i = i_end
    y = jnp.concatenate(y,axis=0) #concatenate over rep axis
    return  y



class LazyPerm(LinearOperator):
    def __init__(self,perm):
        self.perm=perm
        self.shape = (len(perm),len(perm))

    def _matmat(self,V):
        return V[self.perm]
    def _matvec(self,V):
        return V[self.perm]
    def _adjoint(self):
        return LazyPerm(np.argsort(self.perm))
    def invT(self):
        return self

class LazyShift(LinearOperator):
    def __init__(self,n,k=1):
        self.k=k
        self.shape = (n,n)

    def _matmat(self,V): #(c,k) #Still needs to be tested??
        return jnp.roll(V,self.k,axis=0)
    def _matvec(self,V):
        return jnp.roll(V,self.k,axis=0)
    def _adjoint(self):
        return LazyShift(self.shape[0],-self.k)
    def invT(self):
        return self

class SwapMatrix(LinearOperator):
    def __init__(self,swaprows,n):
        self.swaprows=swaprows
        self.shape = (n,n)
    def _matmat(self,V): #(c,k)
        V = jax.ops.index_update(V, jax.ops.index[self.swaprows], V[self.swaprows[::-1]])
        return V
    def _matvec(self,V):
        return self._matmat(V)
    def _adjoint(self):
        return self
    def invT(self):
        return self

class Rot90(LinearOperator):
    def __init__(self,n,k):
        self.shape = (n*n,n*n)
        self.n=n
        self.k = k
    def _matmat(self,V): #(c,k)
        return jnp.rot90(V.reshape((self.n,self.n,-1)),self.k).reshape(V.shape)
    def _matvec(self,V):
        return jnp.rot90(V.reshape((self.n,self.n,-1)),self.k).reshape(V.shape)
    def invT(self):
        return self