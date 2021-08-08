import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, device_put,vmap
import optax
from sklearn.cluster import KMeans
from tqdm.auto import tqdm
from .linear_operator_base import LinearOperator, Lazy
from .linear_operators import ConcatLazy, I, lazify, densify, LazyJVP
import logging
import matplotlib.pyplot as plt
from functools import reduce
from emlp.utils import export

from plum import dispatch
import emlp.reps
#TODO: add rep,v = flatten({'Scalar':..., 'Vector':...,}), to_dict(rep,vector) returns {'Scalar':..., 'Vector':...,}
#TODO and simpler rep = flatten({Scalar:2,Vector:10,...}),
# Do we even want + operator to implement non canonical orderings?

__all__ = ["V","Vector", "Scalar"]

@export
class Rep(object):
    r""" The base Representation class. Representation objects formalize the vector space V
        on which the group acts, the group representation matrix ρ(g), and the Lie Algebra
        representation dρ(A) in a single object. Representations act as types for vectors coming
        from V. These types can be manipulated and transformed with the built in operators
        ⊕,⊗,dual, as well as incorporating custom representations. Rep objects should
        be immutable.

        At minimum, new representations need to implement ``rho``, ``__str__``."""
        
    is_permutation=False

    def rho(self,M):
        """ Group representation of the matrix M of shape (d,d)"""
        raise NotImplementedError
        
    def drho(self,A): 
        """ Lie Algebra representation of the matrix A of shape (d,d)"""
        In = jnp.eye(A.shape[0])
        return LazyJVP(self.rho,In,A)

    def __call__(self,G):
        """ Instantiate (non concrete) representation with a given symmetry group"""
        raise NotImplementedError

    def __str__(self): raise NotImplementedError 
    #TODO: separate __repr__ and __str__?
    def __repr__(self): return str(self)
    
    
    def __eq__(self,other):
        if type(self)!=type(other): return False
        d1 = tuple([(k,v) for k,v in self.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        d2 = tuple([(k,v) for k,v in other.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        return d1==d2
    def __hash__(self):
        d1 = tuple([(k,v) for k,v in self.__dict__.items() if (k not in ['_size','is_permutation','is_orthogonal'])])
        return hash((type(self),d1))

    def size(self): 
        """ Dimension dim(V) of the representation """
        if hasattr(self,'_size'):
            return self._size
        elif self.concrete and hasattr(self,"G"):
            self._size = self.rho(self.G.sample()).shape[-1]
            return self._size
        else: raise NotImplementedError

    def canonicalize(self): 
        """ An optional method to convert the representation into a canonical form
            in order to reuse equivalent solutions in the solver. Should return
            both the canonically ordered representation, along with a permutation
            which can be applied to vectors of the current representation to achieve
            that ordering. """
        return self, np.arange(self.size()) # return canonicalized rep
    
    def rho_dense(self,M):
        """ A convenience function which returns rho(M) as a dense matrix."""
        return densify(self.rho(M))
    def drho_dense(self,A):
        """ A convenience function which returns drho(A) as a dense matrix."""
        return densify(self.drho(A))
    
    def constraint_matrix(self):
        """ Constructs the equivariance constrant matrix (lazily) by concatenating
        the constraints (ρ(hᵢ)-I) for i=1,...M and dρ(Aₖ) for k=1,..,D from the generators
        of the symmetry group. """
        n = self.size()
        constraints = []
        constraints.extend([lazify(self.rho(h))-I(n) for h in self.G.discrete_generators])
        constraints.extend([lazify(self.drho(A)) for A in self.G.lie_algebra])
        return ConcatLazy(constraints) if constraints else lazify(jnp.zeros((1,n)))

    solcache = {}
    def equivariant_basis(self):  
        """ Computes the equivariant solution basis for the given representation of size N.
            Canonicalizes problems and caches solutions for reuse. Output [Q (N,r)] """
        if self==Scalar: return jnp.ones((1,1))
        canon_rep,perm = self.canonicalize()
        invperm = np.argsort(perm)
        if canon_rep not in self.solcache:
            logging.info(f"{canon_rep} cache miss")
            logging.info(f"Solving basis for {self}"+(f", for G={self.G}" if hasattr(self,"G") else ""))
            #if isinstance(group,Trivial): return np.eye(size(rank,group.d))
            C_lazy = canon_rep.constraint_matrix()
            if C_lazy.shape[0]*C_lazy.shape[1]>3e7: #Too large to use SVD
                result = krylov_constraint_solve(C_lazy)
            else:
                C_dense = C_lazy.to_dense()
                result = orthogonal_complement(C_dense)
            self.solcache[canon_rep]=result
        return self.solcache[canon_rep][invperm]
    
    def equivariant_projector(self):
        """ Computes the (lazy) projection matrix P=QQᵀ that projects to the equivariant basis."""
        Q = self.equivariant_basis()
        Q_lazy = lazify(Q)
        P = Q_lazy@Q_lazy.H
        return P

    @property
    def concrete(self):
        return hasattr(self,"G") and self.G is not None
        # if hasattr(self,"_concrete"): return self._concrete
        # else:
        #     return hasattr(self,"G") and self.G is not None

    def __add__(self, other):
        """ Direct sum (⊕) of representations. """
        if isinstance(other,int):
            if other==0: return self
            else: return self+other*Scalar
        elif emlp.reps.product_sum_reps.both_concrete(self,other):
            return emlp.reps.product_sum_reps.SumRep(self,other)
        else:
            return emlp.reps.product_sum_reps.DeferredSumRep(self,other)

    def __radd__(self,other):
        if isinstance(other,int): 
            if other==0: return self
            else: return other*Scalar+self
        else: return NotImplemented
        
    def __mul__(self,other):
        """ Tensor sum (⊗) of representations. """
        return mul_reps(self,other)
            
    def __rmul__(self,other):
        return mul_reps(other,self)

    def __pow__(self,other):
        """ Iterated tensor product. """
        assert isinstance(other,int), f"Power only supported for integers, not {type(other)}"
        assert other>=0, f"Negative powers {other} not supported"
        return reduce(lambda a,b:a*b,other*[self],Scalar)
    def __rshift__(self,other):
        """ Linear maps from self -> other """
        return other*self.T
    def __lshift__(self,other):
        """ Linear maps from other -> self """
        return self*other.T
    def __lt__(self, other):
        """ less than defined to disambiguate ordering multiple different representations.
            Canonical ordering is determined first by Group, then by size, then by hash"""
        if other==Scalar: return False
        try: 
            if self.G<other.G: return True
            if self.G>other.G: return False
        except (AttributeError,TypeError): pass
        if self.size()<other.size(): return True
        if self.size()>other.size(): return False
        return hash(self) < hash(other) #For sorting purposes only
    def __mod__(self,other): # Wreath product
        """ Wreath product of representations (Not yet implemented)"""
        raise NotImplementedError
    @property
    def T(self):
        """ Dual representation V*, rho*, drho*."""
        if hasattr(self,"G") and (self.G is not None) and self.G.is_orthogonal: return self
        return Dual(self)


@dispatch
def mul_reps(ra,rb:int):
    if rb==1: return ra
    if rb==0: return 0
    if (not hasattr(ra,'concrete')) or ra.concrete:
        return emlp.reps.product_sum_reps.SumRep(*(rb*[ra]))
    else:
        return emlp.reps.product_sum_reps.DeferredSumRep(*(rb*[ra]))

@dispatch
def mul_reps(ra:int,rb):
    return mul_reps(rb,ra)

# Continued with non int cases in product_sum_reps.py

# A possible
class ScalarRep(Rep):
    def __init__(self,G=None):
        self.G=G
        self.is_permutation = True
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
    @property
    def concrete(self):
        return True

class Base(Rep):
    """ Base representation V of a group."""
    def __init__(self,G=None):
        self.G=G
        if G is not None: self.is_permutation = G.is_permutation
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
    
    def __hash__(self):
        return hash((type(self),self.G))
    def __eq__(self,other):
        return type(other)==type(self) and self.G==other.G
    def __lt__(self,other):
        if isinstance(other,Dual): return True
        return super().__lt__(other)
    # @property
    # def T(self):
    #     return Dual(self.G)

class Dual(Rep):
    def __init__(self,rep):
        self.rep = rep
        self.G=rep.G
        if hasattr(rep,"is_permutation"): self.is_permutation = rep.is_permutation
    def __call__(self,G):
        return self.rep(G).T
    def rho(self,M):
        rho = self.rep.rho(M)
        rhoinvT = rho.invT() if isinstance(rho,LinearOperator) else jnp.linalg.inv(rho).T
        return rhoinvT
    def drho(self,A):
        return -self.rep.drho(A).T
    def __str__(self):
        return str(self.rep)+"*"
    def __repr__(self): return str(self)
    @property
    def T(self):
        return self.rep
    def __eq__(self,other):
        return type(other)==type(self) and self.rep==other.rep
    def __hash__(self):
        return hash((type(self),self.rep))
    def __lt__(self,other):
        if other==self.rep: return False
        return super().__lt__(other)
    def size(self):
        return self.rep.size()
        
V=Vector= Base()  #: Alias V or Vector for an instance of the Base representation of a group

Scalar = ScalarRep()#: An instance of the Scalar representation, equivalent to V**0

@export
def T(p,q=0,G=None):
    """ A convenience function for creating rank (p,q) tensors."""
    return (V**p*V.T**q)(G)

def orthogonal_complement(proj):
    """ Computes the orthogonal complement to a given matrix proj"""
    U,S,VH = jnp.linalg.svd(proj,full_matrices=True)
    rank = (S>1e-5).sum()
    return VH[rank:].conj().T

def krylov_constraint_solve(C,tol=1e-5):
    """ Computes the solution basis Q for the linear constraint CQ=0  and QᵀQ=I
        up to specified tolerance with C expressed as a LinearOperator. """
    r = 5
    if C.shape[0]*r*2>2e9: raise Exception(f"Solns for contraints {C.shape} too large to fit in memory")
    found_rank=5
    while found_rank==r:
        r *= 2 # Iterative doubling of rank until large enough to include the full solution space
        if C.shape[0]*r>2e9:
            logging.error(f"Hit memory limits, switching to sample equivariant subspace of size {found_rank}")
            break
        Q = krylov_constraint_solve_upto_r(C,r,tol)
        found_rank = Q.shape[-1]
    return Q

def krylov_constraint_solve_upto_r(C,r,tol=1e-5,lr=1e-2):#,W0=None):
    """ Iterative routine to compute the solution basis to the constraint CQ=0 and QᵀQ=I
        up to the rank r, with given tolerance. Uses gradient descent (+ momentum) on the
        objective |CQ|^2, which provably converges at an exponential rate."""
    W = np.random.randn(C.shape[-1],r)/np.sqrt(C.shape[-1])# if W0 is None else W0
    W = device_put(W)
    opt_init,opt_update = optax.sgd(lr,.9)
    opt_state = opt_init(W)  # init stats

    def loss(W):
        return (jnp.absolute(C@W)**2).sum()/2 # added absolute for complex support

    loss_and_grad = jit(jax.value_and_grad(loss))
    # setup progress bar
    pbar = tqdm(total=100,desc=f'Krylov Solving for Equivariant Subspace r<={r}',
    bar_format="{l_bar}{bar}| {n:.3g}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")
    prog_val = 0
    lstart, _ = loss_and_grad(W)
    
    for i in range(20000):
        
        lossval, grad = loss_and_grad(W)
        updates, opt_state = opt_update(grad, opt_state, W)
        W = optax.apply_updates(W, updates)
        # update progress bar
        progress = max(100*np.log(lossval/lstart)/np.log(tol**2/lstart)-prog_val,0)
        progress = min(100-prog_val,progress)
        if progress>0:
            prog_val += progress
            pbar.update(progress)

        if jnp.sqrt(lossval) <tol: # check convergence condition
            pbar.close()
            break # has converged
        if lossval>2e3 and i>100: # Solve diverged due to too high learning rate
            logging.warning(f"Constraint solving diverged, trying lower learning rate {lr/3:.2e}")
            if lr < 1e-4: raise ConvergenceError(f"Failed to converge even with smaller learning rate {lr:.2e}")
            return krylov_constraint_solve_upto_r(C,r,tol,lr=lr/3)
    else: raise ConvergenceError("Failed to converge.")
    # Orthogonalize solution at the end
    U,S,VT = np.linalg.svd(np.array(W),full_matrices=False) 
    # Would like to do economy SVD here (to not have the unecessary O(n^2) memory cost) 
    # but this is not supported in numpy (or Jax) unfortunately.
    rank = (S>10*tol).sum()
    Q = device_put(U[:,:rank])
    # final_L
    final_L = loss_and_grad(Q)[0]
    if final_L >tol: logging.warning(f"Normalized basis has too high error {final_L:.2e} for tol {tol:.2e}")
    scutoff = (S[rank] if r>rank else 0)
    assert rank==0 or scutoff < S[rank-1]/100, f"Singular value gap too small: {S[rank-1]:.2e} \
        above cutoff {scutoff:.2e} below cutoff. Final L {final_L:.2e}, earlier {S[rank-5:rank]}"
    #logging.debug(f"found Rank {r}, above cutoff {S[rank-1]:.3e} after {S[rank] if r>rank else np.inf:.3e}. Loss {final_L:.1e}")
    return Q

class ConvergenceError(Exception): pass

@export
def sparsify_basis(Q,lr=1e-2): #(n,r)
    """ Convenience function to attempt to sparsify a given basis by applying an orthogonal transformation
        W, Q' = QW where Q' has only 1s, 0s and -1s. Notably this method does not have the same convergence
        gauruntees of krylov_constraint_solve and can fail (even silently). Intended to be used only for
        visualization purposes, use at your own risk. """
    W = np.random.randn(Q.shape[-1],Q.shape[-1])
    W,_ = np.linalg.qr(W)
    W = device_put(W.astype(jnp.float32))
    opt_init,opt_update = optax.adam(lr)#optax.sgd(1e2,.9)#optax.adam(lr)#optax.sgd(3e-3,.9)#optax.adam(lr)
    opt_update = jit(opt_update)
    opt_state = opt_init(W)  # init stats

    def loss(W):
        return jnp.abs(Q@W.T).mean() + .1*(jnp.abs(W.T@W-jnp.eye(W.shape[0]))).mean()+.01*jax.numpy.linalg.slogdet(W)[1]**2

    loss_and_grad = jit(jax.value_and_grad(loss))

    for i in tqdm(range(3000),desc=f'sparsifying basis'):
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
@export
def bilinear_weights(out_rep,in_rep):
    #TODO: replace lazy_projection function with LazyDirectSum LinearOperator
    W_rep,W_perm = (in_rep>>out_rep).canonicalize()
    # possible bug when in_rep and out_rep are both non sumreps? #TODO: investigate
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
        
# @jit
# def mul_part(bparams,x,bids):
#     b = prod(x.shape[:-1])
#     return (bparams@x[...,bids].T.reshape(bparams.shape[-1],-1)).reshape(-1,b).T

@export
def vis(repin,repout,cluster=True):
    """ A function to visualize the basis of equivariant maps repin>>repout
        as an image. Only use cluster=True if you know Pv will only have
        r distinct values (true for G<S(n) but not true for many continuous groups)."""
    rep = (repin>>repout)
    P = rep.equivariant_projector() # compute the equivariant basis
    Q = rep.equivariant_basis()
    v = np.random.randn(P.shape[1])  # sample random vector
    v = np.round(P@v,decimals=4)  # project onto equivariant subspace (and round)
    if cluster: # cluster nearby values for better color separation in plot
        v = KMeans(n_clusters=Q.shape[-1]).fit(v.reshape(-1,1)).labels_
    plt.imshow(v.reshape(repout.size(),repin.size()))
    plt.axis('off')


def scale_adjusted_rel_error(t1,t2,g):
    error = jnp.sqrt(jnp.mean(jnp.abs(t1-t2)**2))
    tscale = jnp.sqrt(jnp.mean(jnp.abs(t1)**2)) + jnp.sqrt(jnp.mean(jnp.abs(t2)**2))
    gscale = jnp.sqrt(jnp.mean(jnp.abs(g-jnp.eye(g.shape[-1]))**2))
    scale = jnp.maximum(tscale,gscale)
    return error/jnp.maximum(scale,1e-7)

@export
def equivariance_error(W,repin,repout,G):
    """ Computes the equivariance relative error rel_err(Wρ₁(g),ρ₂(g)W)
        of the matrix W (dim(repout),dim(repin)) [or basis Q: (dim(repout)xdim(repin), r)]
        according to the input and output representations and group G. """
    W = W.reshape(repout.size(),repin.size(),-1).transpose((2,0,1))[None]

    # Sample 5 group elements and verify the equivariance for each
    gs = G.samples(5)
    ring = vmap(repin.rho_dense)(gs)[:,None]
    routg = vmap(repout.rho_dense)(gs)[:,None]
    equiv_err = scale_adjusted_rel_error(W@ring,routg@W,gs)
    return equiv_err

import emlp.groups # Why is this necessary to avoid circular import?