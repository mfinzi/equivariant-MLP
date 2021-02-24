import numpy as np
#import torch
import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax import device_put
import collections,itertools
from functools import lru_cache as cache
from core.utils import disk_cache
import scipy as sp
import scipy.linalg
import functools
import random
from core.representation import Rep
from core.linear_operator_jax import LinearOperator
import logging
import copy
import math
from functools import reduce
from collections import defaultdict
import objax


class SumRep(Rep):
    concrete=True
    atomic=False
    def __init__(self,*reps,extra_perm=None,viz_shape_hint=None):#repcounter,repperm=None):
        """ Constructs a tensor type based on a list of tensor ranks
            and possibly the symmetry generators gen."""
        # Integers can be used as shorthand for scalars.
        reps = [SumRepFromCollection({Scalar:rep}) if isinstance(rep,int) else rep for rep in reps]
        # Get reps and permutations
        #print(f"received following {reps}, {len(reps)}")
        reps,perms = zip(*[rep.canonicalize() for rep in reps])
        rep_counters = [rep.reps if isinstance(rep,SumRep) else {rep:1} for rep in reps]
        # Combine reps and permutations: ∑_a + ∑_b = ∑_{a∪b}
        #self.reps = sum(rep_counters,Counter({}))
        self.reps,perm = self.compute_canonical(rep_counters,perms)
        self.perm = extra_perm[perm] if extra_perm is not None else perm
        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==self.invperm).all()
        self.is_regular = all(rep.is_regular for rep in self.reps.keys())
        if viz_shape_hint is not None: self.viz_shape_hint = viz_shape_hint
        # if not self.canonical:
        #     print(self,self.perm,self.invperm)

    def canonicalize(self):
        """Returns a canonically ordered rep with order np.arange(self.size()) and the
            permutation which achieves that ordering"""
        return SumRepFromCollection(self.reps),self.perm
    def __eq__(self, other):
        assert self.canonical
        return self.reps==other.reps# and self.perm == other.perm
    def __len__(self):
        return sum(multiplicity for multiplicity in self.reps.values())
    def size(self):
        return sum(rep.size()*count for rep,count in self.reps.items())

    @property
    def T(self):
        """ only swaps to adjoint representation, does not reorder elems"""
        return SumRep(*[rep.T for rep,c in self.reps.items() for _ in range(c)],extra_perm=self.perm)
        # not necessarily still canonical ordered
        #return SumRepFromCollection({rep.T:c for rep,c in self.reps.items()},self.perm)

    def __repr__(self):
        return "+".join(f"{count if count > 1 else ''}{repr(rep)}" for rep, count in self.reps.items())
    def __str__(self):
        tensors = "+".join(f"{count if count > 1 else ''}{rep}" for rep, count in self.reps.items())
        return tensors#+f" @ d={self.d}" if self.d is not None else tensors

    def __hash__(self):
        assert self.canonical
        return hash(tuple(self.reps.items()))

    @staticmethod
    def compute_canonical(rep_cnters,rep_perms):
        """ given that rep1_perm and rep2_perm are the canonical orderings for
            rep1 and rep2 (ie v[rep1_perm] is in canonical order) computes
            the canonical order for rep1 + rep2"""
        # First: merge counters
        unique_reps = sorted(reduce(lambda a,b: a|b,[cnter.keys() for cnter in rep_cnters]))
        merged_cnt = defaultdict(int)
        permlist = []
        ids = [0]*len(rep_cnters)
        shifted_perms = []
        n=0
        for perm in rep_perms:
            shifted_perms.append(n+perm)
            n+=len(perm)
        for rep in unique_reps:
            for i in range(len(ids)):
                c = rep_cnters[i].get(rep,0)
                permlist.append(shifted_perms[i][ids[i]:ids[i]+c*rep.size()])
                ids[i]+=+c*rep.size()
                merged_cnt[rep]+=c
        #print(permlist)
        return dict(merged_cnt),np.concatenate(permlist)

    def symmetric_basis(self):
        """ Given a representation which is a sequence of tensors
        with ranks (p_i,q_i), computes the orthogonal complement
        to the projection matrix drho(Mi). Function returns both the
        dimension of the active subspace (r) and also a function that
        maps an array of size (*,r) to a vector v with a representaiton
        given by the rnaks that satisfies drho(Mi)v=0 for each i.
        Inputs: [generators seq(tensor(d,d))] [ranks seq(tuple(p,q))]
        Outputs: [r int] [projection (tensor(r)->tensor(rep_dim))]"""
        Qs = {rep: rep.symmetric_basis() for rep in self.reps}
        Qs = {rep: (jax.device_put(Q.astype(np.float32)) if isinstance(Q,(np.ndarray)) else Q) for rep,Q in Qs.items()}
        active_dims = sum([self.reps[rep]*Qs[rep].shape[-1] for rep in Qs.keys()])
        def lazy_Q(array):
            array = array.T
            i=0
            Ws = []
            for rep, multiplicity in self.reps.items():
                Qr = Qs[rep]
                if not Qr.shape[-1]: continue
                i_end = i+multiplicity*Qr.shape[-1]
                elems = Qr@array[...,i:i_end].reshape(-1,Qr.shape[-1]).T
                Ws.append(elems.T.reshape(*array.shape[:-1],multiplicity*rep.size()))
                i = i_end
            Ws = jnp.concatenate(Ws,axis=-1) #concatenate over rep axis
            inp_ordered_Ws = Ws[...,self.invperm] #(should it be inverse?) reorder to original rep ordering 
            return  inp_ordered_Ws.T
        return LinearOperator(shape=(self.size(),active_dims),matvec=lazy_Q,matmat=lazy_Q)

    def symmetric_projector(self):
        Ps = {rep:rep.symmetric_projector() for rep in self.reps}
        
        # Apply the projections for each rep, concatenate, and permute back to orig rep order
        def lazy_QQT(W):
            ordered_W = W[self.perm]
            PWs = []
            i=0
            for rep, multiplicity in self.reps.items():
                P = Ps[rep]
                i_end = i+multiplicity*rep.size()
                PWs.append((P@ordered_W[i:i_end].reshape(multiplicity,rep.size()).T).T.reshape(-1))
                i = i_end
                #print(rep,multiplicity,i_end)
            PWs = jnp.concatenate(PWs,axis=-1) #concatenate over rep axis
            inp_ordered_PWs = PWs[self.invperm]
            #print(inp_ordered_PWs)
            return  inp_ordered_PWs # reorder to original rep ordering
        return LinearOperator(shape=(self.size(),self.size()),matvec=lazy_QQT)

    ##TODO: investigate why these more idiomatic definitions with Lazy Tensors end up slower
    # def symmetric_basis(self):
    #     Qs = [rep.symmetric_basis() for rep in self.reps]
    #     multiplicities  = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(Qs,multiplicities)
    # def symmetric_projector(self):
    #     Ps = [rep.symmetric_projector() for rep in self.reps]
    #     multiplicities  = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(Ps,multiplicities)@LazyPerm(self.perm)
    def rho(self,M):
        rhos = [rep.rho(M) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm)@LazyDirectSum(rhos,multiplicities)@LazyPerm(self.perm)
    def drho(self,A):
        drhos = [rep.drho(A) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm)@LazyDirectSum(drhos,multiplicities)@LazyPerm(self.perm)
    # def _rho_lazy(self,M):
    #     rhos = [rep.rho_lazy(M) for rep in self.reps]
    #     multiplicities = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(rhos,multiplicities)@LazyPerm(self.perm)

    # def _rho(self,M):
    #     rhos = [rep.rho(M) for rep,c in self.reps.items() for _ in range(c)]
    #     rho_blocks = jax.scipy.linalg.block_diag(*rhos)
    #     return rho_blocks[self.invperm,:][:,self.invperm]

    # def _drho_lazy(self,A):
    #     drhos = [rep.drho_lazy(A) for rep in self.reps]
    #     multiplicities = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(drhos,multiplicities)@LazyPerm(self.perm)

    # def _drho(self,A):
    #     drhos = [rep.drho(A) for rep,c in self.reps.items() for _ in range(c)]
    #     drhoA_blocks = jax.scipy.linalg.block_diag(*rhos)
    #     return drhoA_blocks[self.invperm,:][:,self.invperm]

    def as_dict(self,v):
        out_dict = {}
        i =0
        for rep,c in self.reps.items():
            chunk = c*rep.size()
            out_dict[rep] = v[...,self.perm[i:i+chunk]].reshape(v.shape[:-1]+(c,rep.size()))
            i+= chunk
        return out_dict

    def __call__(self,G):
        return SumRepFromCollection({rep.T:c for rep,c in self.reps.items()},perm=self.perm)

class SumRepFromCollection(SumRep): # a different constructor for SumRep
    def __init__(self,counter,perm=None):
        self.reps = counter
        self.perm = np.arange(self.size()) if perm is None else perm
        self.reps,self.perm = self.compute_canonical([counter],[self.perm])
        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==self.invperm).all()
        self.is_regular = all(rep.is_regular for rep in self.reps.keys())
        # if not self.canonical:
        #     print(self,self.perm,self.invperm)

def distribute_product(reps,extra_perm=None):

    reps,perms =zip(*[repsum.canonicalize() for repsum in reps])
    reps = [rep if isinstance(rep,SumRep) else SumRepFromCollection({rep:1}) for rep in reps]
    # compute axis_wise perm to canonical vector ordering along each axis
    
    axis_sizes = [len(perm) for perm in perms]

    order = np.arange(math.prod(axis_sizes)).reshape(tuple(len(perm) for perm in perms))
    for i,perm in enumerate(perms):
        order = np.swapaxes(np.swapaxes(order,0,i)[perm,...],0,i)
    order = order.reshape(-1)
    #logging.info(f"axiswise: {order}")
    # Compute permutation from multilinear map ordering -> vector ordering (decomposing the blocks)
    repsizes_all = []
    for rep in reps:
        this_rep_sizes = []
        for r,c in rep.reps.items():
            this_rep_sizes.extend([c*r.size()])
        repsizes_all.append(tuple(this_rep_sizes))
    block_perm = rep_permutation(tuple(repsizes_all))
    #logging.info(f"block perm {block_perm}")
    # must go from itertools product ordering to multiplicity grouped ordering
    ordered_reps = []
    each_perm = []
    i = 0
    for prod in itertools.product(*[rep.reps.items() for rep in reps]):
        rs,cs = zip(*prod)
        #import pdb; pdb.set_trace()
        prod_rep,canonicalizing_perm = (math.prod(cs)*reduce(lambda a,b: a*b,rs)).canonicalize()
        #print(f"{rs}:{cs} in distribute yield prod_rep {prod_rep}")
        ordered_reps.append(prod_rep)
        shape = []
        for r,c in prod:
            shape.extend([c,r.size()])
        axis_perm = np.concatenate([2*np.arange(len(prod)),2*np.arange(len(prod))+1])
        mul_perm = np.arange(len(canonicalizing_perm)).reshape(shape).transpose(axis_perm).reshape(-1)
        each_perm.append(mul_perm[canonicalizing_perm]+i)
        i+= len(canonicalizing_perm)
    each_perm = np.concatenate(each_perm)
    #logging.info(f"each perm {each_perm}")
    #
    total_perm = order[block_perm[each_perm]]
    if extra_perm is not None: total_perm = extra_perm[total_perm]
    #TODO: could achieve additional reduction by canonicalizing at this step, but unnecessary for now
    return SumRep(*ordered_reps,extra_perm=total_perm,viz_shape_hint=axis_sizes)


@cache()
def rep_permutation(repsizes_all):
    """Permutation from block ordering to flattened ordering"""
    size_cumsums = [np.cumsum([0] + [size for size in repsizes]) for repsizes in repsizes_all]
    permutation = np.zeros([cumsum[-1] for cumsum in size_cumsums]).astype(np.int)
    arange = np.arange(permutation.size)
    indices_iter = itertools.product(*[range(len(repsizes)) for repsizes in repsizes_all])
    i = 0
    for indices in indices_iter:
        slices = tuple([slice(cumsum[idx], cumsum[idx + 1]) for idx, cumsum in zip(indices, size_cumsums)])
        slice_lengths = [sl.stop - sl.start for sl in slices]
        chunk_size = np.prod(slice_lengths)
        permutation[slices] += arange[i:i + chunk_size].reshape(*slice_lengths)
        i += chunk_size
    return np.argsort(permutation.reshape(-1))

class ProductRep(Rep):
    concrete=True
    def __init__(self,*reps,extra_perm=None,counter=None):
        #Two variants of the constructor:
        if counter is not None: #one with counter specified directly
            self.reps=counter
            self.reps,self.perm = self.compute_canonical([counter],[np.arange(self.size()) if extra_perm is None else extra_perm])
            #assert (self.perm==np.arange(len(self.perm))).all()
        else: # other with list
            # Get reps and permutations
            reps,perms = zip(*[rep.canonicalize() for rep in reps])
            rep_counters = [rep.reps if type(rep)==ProductRep else {rep:1} for rep in reps]
            
            # Combine reps and permutations: Pi_a + Pi_b = Pi_{a x b}
            self.reps,perm = self.compute_canonical(rep_counters,perms)
            #logging.info(f"reps {self.reps}")
            self.perm = extra_perm[perm] if extra_perm is not None else perm
            #logging.info(f"prod perm={self.perm}")
            # l2 = int(np.log2(self.size()))
            # logging.info(f"prod perm={self.perm.reshape(l2*(2,))}")
            
        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==self.invperm).all()
        Gs = tuple(set(rep.G for rep in self.reps.keys()))
        assert len(Gs)==1, f"Multiple different groups {Gs} in product rep {self}"
        self.G= Gs[0]
        self.is_regular = all(rep.is_regular for rep in self.reps.keys())
        
        # if not self.canonical:
        #     print(self,self.perm,self.invperm)
    # def __new__(cls,*reps,extra_perm=None,counter=None):
        
    #     if counter is not None: reps = counter.keys()
    #     unique_groups = set(rep.G for rep in reps if hasattr(rep,'G'))
    #     if len(unique_groups)>1 and len(unique_groups)!=len(reps):
    #         assert counter is None
    #         # write as ProductRep of separate ProductReps each with only one Group
    #         reps,perms = zip(*[rep.canonicalize() for rep in reps])
    #         rep_counters = [rep.reps if isinstance(rep,ProductRep) else {rep:1} for rep in reps]
    #         reps,perm = cls.compute_canonical(rep_counters,perms) # so that reps is sorted by group
    #         perm = extra_perm[perm] if extra_perm is not None else perm
    #         group_dict = defaultdict(dict)
    #         for rep,c in reps.items():
    #             group_dict[rep.G][rep]=c
    #         sub_products = {ProductRep(counter=repdict):1 for G,repdict in group_dict.items()}
    #         print(f"calling with {sub_products}")
    #         return ProductRep(counter=sub_products,extra_perm=perm) 
    #         #init is being called twice because ProductRepFromCollection is a subclass
    #     else:
    #         return super().__new__(cls)

    def canonicalize(self):
        """Returns a canonically ordered rep with order np.arange(self.size()) and the
            permutation which achieves that ordering"""
        return self.__class__(counter=self.reps),self.perm

    # def rho(self,Ms): 
    #     rhos = [rep.rho(Ms) for rep,c in self.reps.items() for _ in range(c)]
    #     return functools.reduce(jnp.kron,rhos)[self.invperm,:][:,self.invperm]
    # def drho(self,As):
    #     drhos = [rep.drho(As) for rep,c in self.reps.items() for _ in range(c)]
    #     return functools.reduce(kronsum,drhos)[self.invperm,:][:,self.invperm]
    
    def rho(self,Ms,lazy=False):
        if hasattr(self,'G') and isinstance(Ms,dict): Ms=Ms[self.G]
        canonical_lazy = lazy_kron([rep.rho(Ms) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def drho(self,As):
        if hasattr(self,'G') and isinstance(As,dict): As=As[self.G]
        canonical_lazy = lazy_kronsum([rep.drho(As) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def __hash__(self):
        assert self.canonical, f"Not canonical {repr(self)}? perm {self.perm}"
        return hash(tuple(self.reps.items()))
    def __eq__(self, other): #TODO: worry about non canonical?
        return isinstance(other,ProductRep) and self.reps==other.reps# and self.perm == other.perm
    def size(self):
        return math.prod([rep.size()**count for rep,count in self.reps.items()])
    @property
    def T(self): #TODO: reavaluate if this needs to change the order (it does not)
        """ only swaps to adjoint representation, does not reorder elems"""
        return self.__class__(*[rep.T for rep,c in self.reps.items() for _ in range(c)],extra_perm=self.perm)
        #return self.__class__(counter={rep.T:c for rep,c in self.reps.items()},extra_perm=self.perm)
    def __str__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep)+(f"{c}".translate(superscript) if c>1 else "") for rep,c in self.reps.items()])

    @staticmethod
    def compute_canonical(rep_cnters,rep_perms):
        """ given that rep1_perm and rep2_perm are the canonical orderings for
            rep1 and rep2 (ie v[rep1_perm] is in canonical order) computes
            the canonical order for rep1 * rep2"""
        order = np.arange(math.prod(len(perm) for perm in rep_perms))
        # First: merge counters
        unique_reps = sorted(reduce(lambda a,b: a|b,[cnter.keys() for cnter in rep_cnters]))
        merged_cnt = defaultdict(int)
        # Reshape like the tensor it is
        order = order.reshape(tuple(len(perm) for perm in rep_perms))
        # apply the canonicalizing permutations along each axis
        for i,perm in enumerate(rep_perms):
            order = np.moveaxis(np.moveaxis(order,i,0)[perm,...],0,i)

        # sort the axes by canonical ordering
        
        # get original axis ids
        axis_ids = []
        n=0
        for cnter in rep_cnters:
            axis_idsi = {}
            for rep,c in cnter.items():
                axis_idsi[rep] = n+np.arange(c)
                n+= c
            axis_ids.append(axis_idsi)
        axes_perm = []
        for rep in unique_reps:
            for i in range(len(rep_perms)):
                c = rep_cnters[i].get(rep,0)
                if c!=0:
                    axes_perm.append(axis_ids[i][rep])
                    merged_cnt[rep]+=c
        axes_perm = np.concatenate(axes_perm)
        #print(f"prod_axes_perm {axes_perm} with rep_cnters {rep_cnters}, orig perms {rep_perms}")
        # reshaped but with inner axes within a collection explicitly expanded
        order = order.reshape(tuple(rep.size() for cnter in rep_cnters for rep,c in cnter.items() for _ in range(c)))
        final_order = np.transpose(order,axes_perm)
        return dict(merged_cnt),final_order.reshape(-1)


class DirectProduct(ProductRep):
    def __init__(self, *reps,counter=None,extra_perm=None):
        #Two variants of the constructor:
        if counter is not None: #one with counter specified directly
            self.reps=counter
            self.reps,perm = self.compute_canonical([counter],[np.arange(self.size())])
            self.perm = extra_perm[perm] if extra_perm is not None else perm
        else: # other with list
            reps,perms = zip(*[rep.canonicalize() for rep in reps])
            #print([type(rep) for rep in reps],type(rep1),type(rep2))
            rep_counters = [rep.reps if type(rep)==DirectProduct else {rep:1} for rep in reps]
            # Combine reps and permutations: Pi_a + Pi_b = Pi_{a x b}
            reps,perm = self.compute_canonical(rep_counters,perms)
            #print("dprod init",self.reps)
            group_dict = defaultdict(lambda: 1)
            for rep,c in reps.items():
                group_dict[rep.G]=group_dict[rep.G]*rep**c
            sub_products = {rep:1 for G,rep in group_dict.items()}
            self.reps = counter = sub_products
            self.reps,perm2 = self.compute_canonical([counter],[np.arange(self.size())])
            self.perm = extra_perm[perm[perm2]] if extra_perm is not None else perm[perm2]
        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==self.invperm).all()
        # self.G = tuple(set(rep.G for rep in self.reps.keys()))
        # if len(self.G)==1: self.G= self.G[0]
        self.is_regular = all(rep.is_regular for rep in self.reps.keys())
        assert all(count==1 for count in self.reps.values())
    def __str__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep)+f"_{rep.G}" for rep,c in self.reps.items()])

    def symmetric_basis(self):
        return lazy_kron([rep.symmetric_basis() for rep,c in self.reps.items()])

    def symmetric_projector(self):
        return lazy_kron([rep.symmetric_projector() for rep,c in self.reps.items()])

    def rho(self,Ms):
        canonical_lazy = lazy_kron([rep.rho(Ms) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def drho(self,As):
        canonical_lazy = lazy_kronsum([rep.drho(As) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)


class DeferredSumRep(Rep):
    concrete=False
    def __init__(self,*reps):
        self.to_sum=[]
        for rep in reps:
            #assert not isinstance(rep,SumRep),f"{rep} of type {type(rep)} tosum {self.to_sum}"
            self.to_sum.extend(rep.to_sum if isinstance(rep,DeferredSumRep) else [rep])
       
    def __call__(self,G):
        if G is None: return self
        return sum([rep(G) for rep in self.to_sum])
        #return SumRep(*[rep(G) for rep in self.to_sum])
    def __repr__(self):
        return '('+"+".join(f"{rep}" for rep in self.to_sum)+')'
    def __str__(self):
        return repr(self)
    @property
    def T(self):
        return DeferredSumRep(*[rep.T for rep in self.to_sum])

class DeferredProductRep(Rep):
    concrete=False
    def __init__(self,*reps):
        self.to_prod=[]
        for rep in reps:
            assert not isinstance(rep,ProductRep)
            self.to_prod.extend(rep.to_prod if isinstance(rep,DeferredProductRep) else [rep])
    def __call__(self,G):
        if G is None: return self
        return reduce(lambda a,b:a*b,[rep(G) for rep in self.to_prod])
        #return ProductRep(*[rep(G) for rep in self.to_prod])
    def __repr__(self):
        return "⊗".join(f"{rep}" for rep in self.to_prod)
    def __str__(self):
        return repr(self)
    @property
    def T(self):
        return DeferredProductRep(*[rep.T for rep in self.to_prod])#TODO: need to reverse the order?

# class ProductGroupTensorRep(Rep): # Eventually will need to handle reordering to canonical G1,G2, etc (from hash?)
#     atomic=False
#     # TO be used like (T(0) + T(1))(SO(3))*T(1)(S(5)) -> T(2)(SO(3))
#     def __init__(self,rep_dict):
#         assert len(rep_dict)>1, "Not product rep?"
#         self.reps = rep_dict
#         #for rep in rep_dict.values():
#         #self.ordering = 
#     def rho(self,Ms): 
#         rhos = [rep.rho(Ms[G]) for (G,rep) in self.reps.items()]
#         return functools.reduce(jnp.kron,rhos,1)
#     def drho(self,As):
#         drhos = [rep.drho(As[G]) for (G,rep) in self.reps.items()]
#         raise functools.reduce(kronsum,drhos,0)

#     def __eq__(self, other): 
#         if not isinstance(other,ProductGroupTensorRep): return False
#         return len(self.reps)==len(other.reps) \
#             and all(Ga==Gb for Ga,Gb in zip(self.reps,other.reps)) \
#             and all(ra==rb for ra,rb in zip(self.reps.values(),other.reps.values())) \
            
@jit
def kronsum(A,B):
    return jnp.kron(A,jnp.eye(B.shape[-1])) + jnp.kron(jnp.eye(A.shape[-1]),B)



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

class lazy_kron(LinearOperator):

    def __init__(self,Ms):
        self.Ms = Ms
        self.shape = math.prod([Mi.shape[0] for Mi in Ms]), math.prod([Mi.shape[1] for Mi in Ms])
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
        return lazy_kron([Mi.T for Mi in self.Ms])
    def invT(self):
        return lazy_kron([M.invT() for M in self.Ms])
    # def __new__(cls,Ms):
    #     if len(Ms)==1: return Ms[0]
    #     return super().__new__(cls)

class lazy_kronsum(LinearOperator):
    
    def __init__(self,Ms):
        self.Ms = Ms
        self.shape = math.prod([Mi.shape[0] for Mi in Ms]), math.prod([Mi.shape[1] for Mi in Ms])
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
        return lazy_kronsum([Mi.T for Mi in self.Ms])

    def __new__(cls,Ms):
        if len(Ms)==1: return Ms[0]
        return super().__new__(cls)
## could also be implemented as follows, but the fusing the sum into a single linearOperator is faster
# def lazy_kronsum(Ms):
#     n = len(Ms)
#     lprod = np.cumprod([1]+[mi.shape[-1] for mi in Ms])
#     rprod = np.cumprod([1]+[mi.shape[-1] for mi in reversed(Ms)])[::-1]
#     return reduce(lambda a,b: a+b,[lazy_kron([I(lprod[i]),Mi,I(rprod[i+1])]) for i,Mi in enumerate(Ms)])

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