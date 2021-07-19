import numpy as np
import jax
from jax import jit
import collections,itertools
from functools import lru_cache as cache
from .representation import Rep,ScalarRep,Scalar
from .linear_operator_base import LinearOperator
from .linear_operators import LazyPerm,LazyDirectSum,LazyKron,LazyKronsum,I,lazy_direct_matmat,lazify,product
from functools import reduce
from collections import defaultdict
from plum import dispatch

class SumRep(Rep):
    def __init__(self,*reps,extra_perm=None):#repcounter,repperm=None):
        """ Constructs a tensor type based on a list of tensor ranks
            and possibly the symmetry generators gen."""
        # Integers can be used as shorthand for scalars.
        reps = [SumRepFromCollection({Scalar:rep}) if isinstance(rep,int) else rep for rep in reps]
        # Get reps and permutations
        reps,perms = zip(*[rep.canonicalize() for rep in reps])
        rep_counters = [rep.reps if isinstance(rep,SumRep) else {rep:1} for rep in reps]
        # Combine reps and permutations: ∑_a + ∑_b = ∑_{a∪b}
        #self.reps = sum(rep_counters,Counter({}))
        self.reps,perm = self.compute_canonical(rep_counters,perms)
        self.perm = extra_perm[perm] if extra_perm is not None else perm
        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==np.arange(len(self.perm))).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())

    def size(self):
        return sum(rep.size()*count for rep,count in self.reps.items())

    def rho(self,M):
        rhos = [rep.rho(M) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm)@LazyDirectSum(rhos,multiplicities)@LazyPerm(self.perm)

    def drho(self,A):
        drhos = [rep.drho(A) for rep in self.reps]
        multiplicities = self.reps.values()
        return LazyPerm(self.invperm)@LazyDirectSum(drhos,multiplicities)@LazyPerm(self.perm)

    def __eq__(self, other):
        return self.reps==other.reps and (self.perm==other.perm).all()

    def __hash__(self):
        assert self.canonical
        return hash(tuple(self.reps.items()))

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

    def canonicalize(self):
        """Returns a canonically ordered rep with order np.arange(self.size()) and the
            permutation which achieves that ordering"""
        return SumRepFromCollection(self.reps),self.perm

    def __call__(self,G):
        return SumRepFromCollection({rep.T:c for rep,c in self.reps.items()},perm=self.perm)

    @property
    def concrete(self):
        return True

    def equivariant_basis(self):
        """ Overrides default implementation with a more efficient version which decomposes the constraints
            across the sum."""
        Qs = {rep: rep.equivariant_basis() for rep in self.reps}
        Qs = {rep: (jax.device_put(Q.astype(np.float32)) if isinstance(Q,(np.ndarray)) else Q) for rep,Q in Qs.items()}
        active_dims = sum([self.reps[rep]*Qs[rep].shape[-1] for rep in Qs.keys()])
        multiplicities = self.reps.values()
        def lazy_Q(array):
            return lazy_direct_matmat(array,Qs.values(),multiplicities)[self.invperm]
        return LinearOperator(shape=(self.size(),active_dims),matvec=lazy_Q,matmat=lazy_Q)

    def equivariant_projector(self):
        """ Overrides default implementation with a more efficient version which decomposes the constraints
            across the sum."""
        Ps = {rep:rep.equivariant_projector() for rep in self.reps}
        multiplicities = self.reps.values()
        def lazy_P(array):
            return lazy_direct_matmat(array[self.perm],Ps.values(),multiplicities)[self.invperm]#[:,self.invperm]
        return LinearOperator(shape=(self.size(),self.size()),matvec=lazy_P,matmat=lazy_P)

    # ##TODO: investigate why these more idiomatic definitions with Lazy Operators end up slower
    # def equivariant_basis(self):
    #     Qs = [rep.equivariant_basis() for rep in self.reps]
    #     Qs = [(jax.device_put(Q.astype(np.float32)) if isinstance(Q,(np.ndarray)) else Q) for Q in Qs]
    #     multiplicities  = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(Qs,multiplicities)
    # def equivariant_projector(self):
    #     Ps = [rep.equivariant_projector() for rep in self.reps]
    #     Ps = (jax.device_put(P.astype(np.float32)) if isinstance(P,(np.ndarray)) else P)
    #     multiplicities  = self.reps.values()
    #     return LazyPerm(self.invperm)@LazyDirectSum(Ps,multiplicities)@LazyPerm(self.perm)

    # Some additional SumRep specific methods to be used for internal purposes
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
        return dict(merged_cnt),np.concatenate(permlist)
    def __iter__(self):  # not a great idea to use this method (ignores permutation ordering)
        return (rep for rep,c in self.reps.items() for _ in range(c))
    def __len__(self):
        return sum(multiplicity for multiplicity in self.reps.values())
    def as_dict(self,v):
        out_dict = {}
        i =0
        for rep,c in self.reps.items():
            chunk = c*rep.size()
            out_dict[rep] = v[...,self.perm[i:i+chunk]].reshape(v.shape[:-1]+(c,rep.size()))
            i+= chunk
        return out_dict



def both_concrete(rep1,rep2):
    return all(rep.concrete for rep in (rep1,rep2) if hasattr(rep,'concrete'))

@dispatch.multi((SumRep,Rep),(Rep,SumRep),(SumRep,SumRep))
def mul_reps(ra,rb):
    if not both_concrete(ra,rb):
        return DeferredProductRep(ra,rb)
    return distribute_product([ra,rb])

@dispatch
def mul_reps(ra,rb):  # base case
    if type(ra) is ScalarRep: return rb
    if type(rb) is ScalarRep: return ra
    if not both_concrete(ra,rb):
        return DeferredProductRep(ra,rb)
    if hasattr(ra,"G") and hasattr(rb,"G") and ra.G==rb.G:
        return ProductRep(ra,rb)
    return DirectProduct(ra,rb)


    
#TODO: consolidate with the __init__ method of the basic SumRep
class SumRepFromCollection(SumRep): # a different constructor for SumRep
    def __init__(self,counter,perm=None):
        self.reps = counter
        self.perm = np.arange(self.size()) if perm is None else perm
        self.reps,self.perm = self.compute_canonical([counter],[self.perm])
        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==np.arange(len(self.perm))).all()
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())
        # if not self.canonical:
        #     print(self,self.perm,self.invperm)

def distribute_product(reps,extra_perm=None):
    """ For expanding products of sums into sums of products, (ρ₁⊕ρ₂)⊗ρ₃ = (ρ₁⊗ρ₃)⊕(ρ₂⊗ρ₃).
        takes in a sequence of reps=[ρ₁,ρ₂,ρ₃,...] which are to be multiplied together and at
        least one of the reps is a SumRep, and distributes out the terms."""
    reps,perms =zip(*[repsum.canonicalize() for repsum in reps])
    reps = [rep if isinstance(rep,SumRep) else SumRepFromCollection({rep:1}) for rep in reps]
    # compute axis_wise perm to canonical vector ordering along each axis
    
    axis_sizes = [len(perm) for perm in perms]

    order = np.arange(product(axis_sizes)).reshape(tuple(len(perm) for perm in perms))
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
        prod_rep,canonicalizing_perm = (product(cs)*reduce(lambda a,b: a*b,rs)).canonicalize()
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
    return SumRep(*ordered_reps,extra_perm=total_perm)


@cache(maxsize=None)
def rep_permutation(repsizes_all):
    """Permutation from block ordering to flattened ordering"""
    size_cumsums = [np.cumsum([0] + [size for size in repsizes]) for repsizes in repsizes_all]
    permutation = np.zeros([cumsum[-1] for cumsum in size_cumsums]).astype(int)
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
    def __init__(self,*reps,extra_perm=None,counter=None):
        #Two variants of the constructor:
        if counter is not None: #one with counter specified directly
            self.reps=counter
            self.reps,self.perm = self.compute_canonical([counter],[np.arange(self.size()) if extra_perm is None else extra_perm])
        else: # other with list
            # Get reps and permutations
            reps,perms = zip(*[rep.canonicalize() for rep in reps])
            rep_counters = [rep.reps if type(rep)==ProductRep else {rep:1} for rep in reps]
            
            # Combine reps and permutations: ∏_a ⊗ ∏_b = ∏_{a ∪ b}
            self.reps,perm = self.compute_canonical(rep_counters,perms)
            self.perm = extra_perm[perm] if extra_perm is not None else perm

        self.invperm = np.argsort(self.perm)
        self.canonical=(self.perm==self.invperm).all()
        Gs = tuple(set(rep.G for rep in self.reps.keys()))
        assert len(Gs)==1, f"Multiple different groups {Gs} in product rep {self}"
        self.G= Gs[0]
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())

    def size(self):
        return product([rep.size()**count for rep,count in self.reps.items()])

    def rho(self,Ms,lazy=False):
        if hasattr(self,'G') and isinstance(Ms,dict): Ms=Ms[self.G]
        canonical_lazy = LazyKron([rep.rho(Ms) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def drho(self,As):
        if hasattr(self,'G') and isinstance(As,dict): As=As[self.G]
        canonical_lazy = LazyKronsum([rep.drho(As) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def __hash__(self):
        assert self.canonical, f"Not canonical {repr(self)}? perm {self.perm}"
        return hash(tuple(self.reps.items()))
    def __eq__(self, other): #TODO: worry about non canonical?
        return isinstance(other,ProductRep) and self.reps==other.reps and (self.perm==other.perm).all()
    
    @property
    def concrete(self):
        return True

    @property
    def T(self):
        """ only swaps to adjoint representation, does not reorder elems"""
        return self.__class__(*[rep.T for rep,c in self.reps.items() for _ in range(c)],extra_perm=self.perm)
        #return self.__class__(counter={rep.T:c for rep,c in self.reps.items()},extra_perm=self.perm)
    def __str__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep)+(f"{c}".translate(superscript) if c>1 else "") for rep,c in self.reps.items()])

    def canonicalize(self):
        """Returns a canonically ordered rep with order np.arange(self.size()) and the
            permutation which achieves that ordering"""
        return self.__class__(counter=self.reps),self.perm

    @staticmethod
    def compute_canonical(rep_cnters,rep_perms):
        """ given that rep1_perm and rep2_perm are the canonical orderings for
            rep1 and rep2 (ie v[rep1_perm] is in canonical order) computes
            the canonical order for rep1 * rep2"""
        order = np.arange(product(len(perm) for perm in rep_perms))
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
        # reshaped but with inner axes within a collection explicitly expanded
        order = order.reshape(tuple(rep.size() for cnter in rep_cnters for rep,c in cnter.items() for _ in range(c)))
        final_order = np.transpose(order,axes_perm)
        return dict(merged_cnt),final_order.reshape(-1)


class DirectProduct(ProductRep):
    """ Tensor product of representations ρ₁⊗ρ₂, but where the sub representations
        ρ₁ and ρ₂ are representations of distinct groups (ie ρ₁⊗ρ₂ is a representation
        of the direct product of groups G=G₁×G₂). As a result, the solutions for the two
        sub representations can be solved independently and assembled together with the
        kronecker product: Q = Q₁⊗Q₂ and P = P₁⊗P₂"""
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
        self.is_permutation = all(rep.is_permutation for rep in self.reps.keys())
        assert all(count==1 for count in self.reps.values())

    def equivariant_basis(self):
        canon_Q = LazyKron([rep.equivariant_basis() for rep,c in self.reps.items()])
        return LazyPerm(self.invperm)@canon_Q

    def equivariant_projector(self):
        canon_P = LazyKron([rep.equivariant_projector() for rep,c in self.reps.items()])
        return LazyPerm(self.invperm)@canon_P@LazyPerm(self.perm)

    def rho(self,Ms):
        canonical_lazy = LazyKron([rep.rho(Ms) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def drho(self,As):
        canonical_lazy = LazyKronsum([rep.drho(As) for rep,c in self.reps.items() for _ in range(c)])
        return LazyPerm(self.invperm)@canonical_lazy@LazyPerm(self.perm)

    def __str__(self):
        superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "⊗".join([str(rep)+f"_{rep.G}" for rep,c in self.reps.items()])


class DeferredSumRep(Rep):
    def __init__(self,*reps):
        self.to_sum=[]
        for rep in reps:
            #assert not isinstance(rep,SumRep),f"{rep} of type {type(rep)} tosum {self.to_sum}"
            self.to_sum.extend(rep.to_sum if isinstance(rep,DeferredSumRep) else [rep])
       
    def __call__(self,G):
        if G is None: return self
        return SumRep(*[rep(G) for rep in self.to_sum])
    def __repr__(self):
        return '('+"+".join(f"{rep}" for rep in self.to_sum)+')'
    def __str__(self):
        return repr(self)
    @property
    def T(self):
        return DeferredSumRep(*[rep.T for rep in self.to_sum])
    @property
    def concrete(self):
        return False

class DeferredProductRep(Rep):
    def __init__(self,*reps):
        self.to_prod=[]
        for rep in reps:
            assert not isinstance(rep,ProductRep)
            self.to_prod.extend(rep.to_prod if isinstance(rep,DeferredProductRep) else [rep])
    def __call__(self,G):
        if G is None: return self
        return reduce(lambda a,b:a*b,[rep(G) for rep in self.to_prod])
    def __repr__(self):
        return "⊗".join(f"{rep}" for rep in self.to_prod)
    def __str__(self):
        return repr(self)
    @property
    def T(self):
        return DeferredProductRep(*[rep.T for rep in self.to_prod])
    @property
    def concrete(self):
        return False