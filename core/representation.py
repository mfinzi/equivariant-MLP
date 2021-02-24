import numpy as np
#import torch
import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax import device_put
import optax
import collections,itertools
from functools import lru_cache as cache
from core.utils import disk_cache,ltqdm
from core.linear_operator_jax import LinearOperator
import scipy as sp
import scipy.linalg
import functools
import random
import logging
import core
import math
from jax.ops import index, index_add, index_update
import matplotlib.pyplot as plt
from collections import Counter
from functools import reduce
#TODO: add rep,v = flatten({'Scalar':..., 'Vector':...,}), to_dict(rep,vector) returns {'Scalar':..., 'Vector':...,}
#TODO and simpler rep = flatten({Scalar:2,Vector:10,...}),
# Do we even want + operator to implement non canonical orderings?
class OrderedCounter(collections.Counter,collections.OrderedDict): pass

class Rep(object):
    concrete=True
    def __eq__(self, other): raise NotImplementedError
    def size(self): raise NotImplementedError # dim(V) dimension of the representation
    
    @property
    def T(self): raise NotImplementedError # dual representation V*, rho*, drho*
    def __repr__(self): return str(self)#raise NotImplementedError
    def __str__(self): raise NotImplementedError 
    def __call__(self,G): 
        print(type(self),self)
        raise NotImplementedError # set the symmetry group
    def canonicalize(self): return self, np.arange(self.size()) # return canonicalized rep
    def size(self):
        print(self,type(self))
        raise NotImplementedError # The dimension of the representation
    def rho_dense(self,M):
        return self.rho(M)@jnp.eye(self.size())
    def drho_dense(self,A):
        return self.rho(A)@jnp.eye(self.size())
    # def rho(self,M): # Group representation of matrix M (n,n)
    #     if hasattr(self,'_rho'): return self._rho(M)
    #     elif hasattr(self,'_rho_lazy'): return self._rho_lazy(M)@jnp.eye(self.size())
    #     else: raise NotImplementedError
    # def drho(self,A):# Lie Algebra representation of matrix A (n,n)
    #     if hasattr(self,'_drho'): return self._drho(M)
    #     elif hasattr(self,'_drho_lazy'): return self._drho_lazy(M)@jnp.eye(self.size())
    #     else: raise NotImplementedError
    # def rho_lazy(self,M): # Group representation of matrix M (n,n)
    #     if hasattr(self,'_rho_lazy'): return self._rho_lazy(M)
    #     elif hasattr(self,'_rho'): return self._rho(M)
    #     else: raise NotImplementedError
    # def drho_lazy(self,A):# Lie Algebra representation of matrix A (n,n)
    #     if hasattr(self,'_drho_lazy'): return self._drho_lazy(M)
    #     elif hasattr(self,'_drho'): return self._drho
    #     else: raise NotImplementedError
    # def rho_lazy(self,M): raise NotImplementedError # Lazy version of rho
    # def drho_lazy(self,M): raise NotImplementedError # Lazy version of drho
    # def constraint_matrix(self):
    #     """ Given a sequence of exponential generators [A1,A2,...]
    #     and a tensor rank (p,q), the function concatenates the representations
    #     [drho(A1), drho(A2), ...] into a single large projection matrix.
    #     Input: [generators seq(tensor(d,d))], [rank tuple(p,q)], [d int] """
    #     constraints = [] # Multiply by identity to convert lazy to dense matrices
    #     constraints.extend([self.drho(A)@jnp.eye(self.size()) for A in self.G.lie_algebra])
    #     constraints.extend([self.rho(h)@jnp.eye(self.size())-jnp.eye(self.size()) for h in self.G.discrete_generators])
    #     P = jnp.concatenate(constraints,axis=0) if constraints else jnp.zeros((1,self.size()))
    #     return P
    
    def constraint_matrix(self):
        """ A lazy version of constraint_matrix"""
        return ConstraintMatrixLazy(self.G,self.rho,self.drho,self.size())

    #@disk_cache('./_subspace_cache_jax.dat')
    solcache = {}
    def symmetric_basis(self):  
        """ Given an array of generators [M1,M2,...] and tensor rank (p,q)
            this function computes the orthogonal complement to the projection
            matrix formed by stacking the rows of drho(Mi) together.
            Output [Q (d^(p+q),r)] """
        if self==Scalar: return jnp.ones((1,1))
        canon_rep,perm = self.canonicalize()
        invperm = np.argsort(perm)
        if canon_rep not in self.solcache:
            logging.info(f"{canon_rep} cache miss")
            logging.info(f"Solving basis for {self}"+(f", for G={self.G}" if hasattr(self,"G") else ""))
            #if isinstance(group,Trivial): return np.eye(size(rank,group.d))
            C_lazy = canon_rep.constraint_matrix()
            if math.prod(C_lazy.shape)>3e7: #Too large to use SVD
                result = krylov_constraint_solve(C_lazy)
            else:
                C_dense = C_lazy@jnp.eye(C_lazy.shape[-1])
                result = orthogonal_complement(C_dense)
            self.solcache[canon_rep]=result
        return self.solcache[canon_rep][invperm]
    
    def symmetric_projector(self):
        Q = self.symmetric_basis()
        Q_lazy = Lazy(Q)
        P = Q_lazy@Q_lazy.H
        return P

    def visualize(self,*shape):
        #TODO: add support for non square
        rep,perm = self.canonicalize()
        Q = rep.symmetric_basis()
        A = (sparsify_basis(Q)[np.argsort(perm)]@np.arange(1,Q.shape[-1]+1))
        # Q= self.symmetric_basis() #THIS:
        # A = sparsify_basis(Q)
        if hasattr(self,"viz_shape_hint") and not shape:  shape = self.viz_shape_hint
        plt.imshow(A.reshape(shape))
        plt.axis('off')

    def __add__(self, other): # Tensor sum representation R1 + R2
        if isinstance(other,int):
            if other==0: return self
        elif all(rep.concrete for rep in (self,other) if hasattr(rep,'concrete')):
            return core.product_sum_reps.SumRep(self,other)
        else:
            return core.product_sum_reps.DeferredSumRep(self,other)
    def __radd__(self,other):
        if isinstance(other,int): 
            if other==0: return self
        else: return NotImplemented
        
    def __mul__(self,other):
        if isinstance(other,(int,ScalarRep)):
            if other==1 or other==Scalar: return self
            if other==0: return 0
            return sum(other*[self])
        elif all(rep.concrete for rep in (self,other) if hasattr(rep,'concrete')):
            if any(isinstance(rep,core.product_sum_reps.SumRep) for rep in [self,other]):
                return core.product_sum_reps.distribute_product([self,other])
            elif (hasattr(self,'G') and hasattr(other,'G') and self.G!=other.G) or not (hasattr(self,'G') and hasattr(other,'G')):
                return core.product_sum_reps.DirectProduct(self,other)
            else: 
                return core.product_sum_reps.ProductRep(self,other)
        else:
            return core.product_sum_reps.DeferredProductRep(self,other)
    def __rmul__(self,other):
        if isinstance(other,(int,ScalarRep)): 
            if other==1 or other==Scalar: return self
            if other==0: return 0
            return sum(other*[self])
        else: return NotImplemented

    def __pow__(self,other):
        assert isinstance(other,int), f"Power only supported for integers, not {type(other)}"
        assert other>=0, f"Negative powers {other} not supported"
        return reduce(lambda a,b:a*b,other*[self],Scalar)
    def __rshift__(self,other):
        return other*self.T
    def __lshift__(self,other):
        return self*other.T
    def __lt__(self, other):
        #Canonical ordering is determined 1st by Group, then by size, then by hash
        if other==Scalar: return False
        try: 
            if self.G<other.G: return True
            if self.G>other.G: return False
        except (AttributeError,TypeError): pass
        if self.size()<other.size(): return True
        if self.size()>other.size(): return False
        return hash(self) < hash(other) #For sorting purposes only
    def __mod__(self,other): # Wreath product
        raise NotImplementedError

# A possible
class ScalarRep(Rep):
    def __init__(self,G=None):
        self.G=G
        self.concrete = True#(G is not None)
        self.is_regular = True
    def __call__(self,G):
        self.G=G
        return self
    def size(self):
        return 1
    def __repr__(self): return str(self)#f"T{self.rank+(self.G,)}"
    def __str__(self):
        return "V⁰"
    @property
    def T(self):
        return self
    def rho(self,M):
        return jnp.eye(1)
    def drho(self,M):
        return 0*jnp.eye(1)
    def __hash__(self):
        return 0
    def __eq__(self,other):
        return isinstance(other,ScalarRep)
    def __mul__(self,other):
        if isinstance(other,int): return super().__mul__(other)
        return other
    def __rmul__(self,other):
        if isinstance(other,int): return super().__rmul__(other)
        return other

class Base(Rep):
    def __init__(self,G=None):
        self.G=G
        self.concrete = (G is not None)
        if G is not None: self.is_regular = G.is_regular
    def __call__(self,G):
        return self.__class__(G)
    def rho(self,M):
        if hasattr(self,'G') and isinstance(M,dict): M=M[self.G]
        return M
    def drho(self,A):
        if hasattr(self,'G') and isinstance(A,dict): A=A[self.G]
        return A
    def size(self):
        assert self.G is not None, f"must know G to find size for rep={self}"
        return self.G.d
    def __repr__(self): return str(self)#f"T{self.rank+(self.G,)}"
    def __str__(self):
        return "V"# +(f"_{self.G}" if self.G is not None else "")
    @property
    def T(self):
        return Dual(self.G)
    def __hash__(self):
        return hash((type(self),self.G))
    def __eq__(self,other):
        return type(other)==type(self) and self.G==other.G
    def __lt__(self,other):
        if isinstance(other,Dual): return True
        return super().__lt__(other)

class Dual(Base):
    def __new__(cls,G=None):
        if G is not None and G.is_orthogonal: return Base(G)
        else: return super(Dual,cls).__new__(cls)
    def rho(self,M):
        MinvT = M.invT() if hasattr(M,'invT') else jnp.linalg.inv(M).T
        return MinvT
    def drho(self,A):
        return -A.T
    def __str__(self):
        return "V*"#+(f"_{self.G}" if self.G is not None else "")
    @property
    def T(self):
        return Base(self.G)
    def __lt__(self,other):
        if isinstance(other,Base): return False
        return super().__lt__(other)

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

V=Vector= Base()
Scalar = ScalarRep()#V**0
def T(p,q=0,G=None):
    return (V**p*V.T**q)(G)

class ConstraintMatrixLazy(LinearOperator):
    def __init__(self,group,rho_lazy,drho_lazy,size):
        self.d = group.d
        self.rho_lazy=rho_lazy
        self.drho_lazy=drho_lazy
        # if group.discrete_generators_lazy is not NotImplemented:
        #     self.hi = group.discrete_generators_lazy
        # else: 
        #     self.hi=group.discrete_generators
        #     logging.debug(f"no discrete lazy found for {group}")
        # if group.lie_algebra_lazy is not NotImplemented:
        #     self.Ai = group.lie_algebra_lazy
        # else:
        #     self.Ai = group.lie_algebra
        #     logging.debug(f"no Lie Algebra lazy found for {group}")
        self.hi = group.discrete_generators
        self.Ai = group.lie_algebra
        self.G=group
        self.n_constraints= len(self.hi)+len(self.Ai)
        if not self.n_constraints: raise NotImplementedError
        self.c = size
        self.dtype=np.float32
    @property
    def shape(self):
        return (self.c*self.n_constraints,self.c)
    def _matmat(self,V): #(c,k)
        constraints = []
        constraints.extend([self.drho_lazy(A)@V for A in self.Ai])
        constraints.extend([self.rho_lazy(h)@V-V for h in self.hi])
        CV = jnp.concatenate(constraints,axis=0)
        return CV
    def _rmatmat(self,V):
        n_constraints = len(self.hi)+len(self.Ai)
        Vi = jnp.split(V,self.n_constraints)
        out = 0
        out += sum([self.drho_lazy(A).T@Vi[i] for i,A in enumerate(self.Ai)])
        out += sum([self.rho_lazy(h).T@Vi[i+len(self.Ai)] for i,h in enumerate(self.hi)])
        return out

def orthogonal_complement(proj):
    """ Computes the orthogonal complement to a given matrix proj"""
    U,S,VH = jnp.linalg.svd(proj,full_matrices=True) # Changed from full_matrices=True
    rank = (S>1e-5).sum()
    return VH[rank:].conj().T

def krylov_constraint_solve(C,tol=1e-5):
    r = 5
    if C.shape[0]*r*2>2e9: raise Exception(f"Solns for contraints {C.shape} too large to fit in memory")
    found_rank=5
    while found_rank==r:
        r *= 2
        if C.shape[0]*r>2e9:
            logging.error(f"Hit memory limits, switching to sample equivariant subspace of size {found_rank}")
            break
        Q = krylov_constraint_solve_upto_r(C,r,tol)
        found_rank = Q.shape[-1]
    return Q

def krylov_constraint_solve_upto_r(C,r,tol=1e-5,lr=1e-2):#,W0=None):
    W = np.random.randn(C.shape[-1],r)/np.sqrt(C.shape[-1])# if W0 is None else W0

    opt_init,opt_update = optax.sgd(lr,.9)
    opt_state = opt_init(W)  # init stats

    def loss(W):
        return (jnp.absolute(C@W)**2).sum()/2 # added absolute for complex support
    loss_and_grad = jit(jax.value_and_grad(loss))

    for i in ltqdm(range(20000),desc=f'Krylov Solving for Equivariant Subspace r<={r}',level='info'):
        lossval, grad = loss_and_grad(W)
        updates, opt_state = opt_update(grad, opt_state, W)
        W = optax.apply_updates(W, updates)
        #W /= jnp.sqrt((W**2).sum(0,keepdims=True))
        if jnp.sqrt(lossval) <tol: break # has converged
        if lossval>2e3 and i>100: # Solve diverged due to too high learning rate
            logging.warning(f"Constraint solving diverged, trying lower learning rate {lr/3:.2e}")
            if lr < 1e-4: raise ConvergenceError(f"Failed to converge even with smaller learning rate {lr:.2e}")
            return krylov_constraint_solve_upto_r(C,r,tol,lr=lr/3)
    else: raise ConvergenceError("Failed to converge.")
    # Orthogonalize solution at the end
    U,S,VT = np.linalg.svd(np.array(W),full_matrices=False)
    
    rank = (S>10*tol).sum()
    Q = U[:,:rank]
    # final_L
    final_L = loss_and_grad(Q)[0]
    assert final_L <tol, f"Normalized basis has too high error {final_L:.2e} for tol {tol:.2e}"
    scutoff = (S[rank] if r>rank else 0)
    assert rank==0 or scutoff < S[rank-1]/100, f"Singular value gap too small: {S[rank-1]:.2e} \
        above cutoff {scutoff:.2e} below cutoff. Final L {final_L:.2e}, earlier {S[rank-5:rank]}"
    #logging.debug(f"found Rank {r}, above cutoff {S[rank-1]:.3e} after {S[rank] if r>rank else np.inf:.3e}. Loss {final_L:.1e}")
    return device_put(Q)

class ConvergenceError(Exception): pass

def sparsify_basis(Q,lr=1e-2): #(n,r)
    W = np.random.randn(Q.shape[-1],Q.shape[-1])
    W,_ = np.linalg.qr(W)
    opt_init,opt_update = optax.adam(lr)#optax.sgd(1e2,.9)#optax.adam(lr)#optax.sgd(3e-3,.9)#optax.adam(lr)
    opt_update = jit(opt_update)
    opt_state = opt_init(W)  # init stats

    def loss(W):
        return jnp.abs(Q@W.T).mean() + .1*(jnp.abs(W.T@W-jnp.eye(W.shape[0]))).mean()+.01*jax.numpy.linalg.slogdet(W)[1]**2

    loss_and_grad = jit(jax.value_and_grad(loss))

    for i in ltqdm(range(3000),desc=f'sparsifying basis',level='info'):
        lossval, grad = loss_and_grad(W)
        updates, opt_state = opt_update(grad, opt_state, W)
        W = optax.apply_updates(W, updates)
        #W,_ = np.linalg.qr(W)
        if lossval>1e2 and i>100: # Solve diverged due to too high learning rate
            logging.warning(f"basis sparsification diverged, trying lower learning rate {lr/3:.2e}")
            return sparsify_basis(Q,lr=lr/3)
    Q = np.copy(Q@W.T)
    Q[np.abs(Q)<1e-2]=0
    Q[np.abs(Q)>1e-2] /= np.abs(Q[np.abs(Q)>1e-2])
    A = Q@(1+np.arange(Q.shape[-1]))
    if len(np.unique(np.abs(A)))!=Q.shape[-1]+1 and len(np.unique(np.abs(A)))!=Q.shape[-1]:
        logging.error(f"Basis elems did not separate: found only {len(np.unique(np.abs(A)))}/{Q.shape[-1]}")
        #raise ConvergenceError(f"Basis elems did not separate: found only {len(np.unique(A))}/{Q.shape[-1]}")
    return Q

#@partial(jit,static_argnums=(0,1))
def bilinear_weights(out_rep,in_rep):
    W_rep,W_perm = (in_rep>>out_rep).canonicalize()
    inv_perm = np.argsort(W_perm)
    mat_shape = out_rep.size(),in_rep.size()
    x_rep=in_rep
    W_multiplicities = W_rep.reps
    x_multiplicities = x_rep.reps
    x_multiplicities = {rep:n for rep,n in x_multiplicities.items() if rep!=Scalar}
    nelems = lambda nx,rep: min(nx,rep.size())
    active_dims = sum([W_multiplicities.get(rep,0)*nelems(n,rep) for rep,n in x_multiplicities.items()])
    reduced_indices_dict = {rep:ids[np.random.choice(len(ids),nelems(len(ids),rep))].reshape(-1)\
                                for rep,ids in x_rep.as_dict(np.arange(x_rep.size())).items()}
    # Apply the projections for each rank, concatenate, and permute back to orig rank order
    @jit
    def lazy_projection(params,x): # (r,), (*c) #TODO: find out why backwards of this function is so slow
        bshape = x.shape[:-1]
        x = x.reshape(-1,x.shape[-1])
        bs = x.shape[0]
        i=0
        Ws = []
        for rep, W_mult in W_multiplicities.items():
            if rep not in x_multiplicities:
                Ws.append(jnp.zeros((bs,W_mult*rep.size())))
                continue
            x_mult = x_multiplicities[rep]
            n = nelems(x_mult,rep)
            i_end = i+W_mult*n
            bids =  reduced_indices_dict[rep]
            bilinear_params = params[i:i_end].reshape(W_mult,n) # bs,nK-> (nK,bs)
            i = i_end  # (bs,W_mult,d^r) = (W_mult,n)@(n,d^r,bs)
            bilinear_elems = bilinear_params@x[...,bids].T.reshape(n,rep.size()*bs)
            bilinear_elems = bilinear_elems.reshape(W_mult*rep.size(),bs).T
            Ws.append(bilinear_elems)
        Ws = jnp.concatenate(Ws,axis=-1) #concatenate over rep axis
        return Ws[...,inv_perm].reshape(*bshape,*mat_shape) # reorder to original rank ordering
    return active_dims,lazy_projection
        
@jit
def mul_part(bparams,x,bids):
    b = math.prod(x.shape[:-1])
    return (bparams@x[...,bids].T.reshape(bparams.shape[-1],-1)).reshape(-1,b).T
