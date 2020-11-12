import numpy as np
import torch
import collections,itertools
from functools import lru_cache
import scipy as sp
import scipy.linalg
import functools

class OrderedCounter(collections.Counter,collections.OrderedDict): pass

def repsize(rep,d):
    return sum(d**(p+q) for (p,q) in rep)

class TensorRep(object):
    def __init__(self,ranks,gen=None,shapes=None):
        """ Constructs a tensor type based on a list of tensor ranks
            and possibly the symmetry generators gen."""
        self.ranks = ranks
        self.shapes = shapes or (self.ranks,)
        self.gen=gen
        self.d = self.gen[0].shape[0] if self.gen is not None else None

    def __eq__(self, other):
        return len(self)==len(other) and all(r==rr for r,rr in zip(self.ranks,other.ranks))
    def __len__(self):
        return len(self.ranks)
    @property
    def shape(self):
        return tuple(repsize(r, self.d) for r in self.shapes)
    def size(self):
        if not self.d: return None
        return repsize(self.ranks,self.d)
    def __call__(self,gen):
        self.gen=gen
        self.d = self.gen[0].shape[0]
        return self
    def __add__(self, other):
        if isinstance(other,int): return self+other*Scalar
        return TensorRep(self.ranks+other.ranks,self.gen if self.gen is not None else other.gen)
    def __radd__(self, other):
        if isinstance(other,int): return other*Scalar+self
        else: assert False, f"Unsupported operand Rep*{type(other)}"
    def __mul__(self, other):
        if isinstance(other,int): return TensorRep(other*self.ranks,self.gen)
        elif isinstance(other,TensorRep):
            product_ranks = [(r1[0]+r2[0],r1[1]+r2[1]) for r1,r2 in itertools.product(self.ranks,other.ranks)]
            return TensorRep(product_ranks,self.gen if self.gen is not None else other.gen,self.shapes+other.shapes)
        else: assert False, f"Unsupported operand Rep*{type(other)}"

    def __rmul__(self, other):
        if isinstance(other, int): return TensorRep(other * self.ranks, self.gen)
        else: assert False, f"Unsupported operand Rep*{type(other)}"
    def __iter__(self):
        return iter(self.ranks)
    @property
    def T(self):
        """ only swaps to adjoint representation, does not reorder elems"""
        return TensorRep([rank[::-1] for rank in self.ranks], self.gen)

    def __matmul__(self, other):
        raise NotImplementedError

    def multiplicities(self):
        if not hasattr(self,'_multiplicities'): self._multiplicities = OrderedCounter(self.ranks)
        return self._multiplicities

    def __repr__(self):
        multiplicities=  self.multiplicities()
        tensors = "+".join(f"{v if v > 1 else ''}T{key}" for key, v in multiplicities.items())
        if self.gen is not None and (self.gen==-self.gen.transpose((0,2,1))).all():
            # If SO(d) subgroup, show T(p,q) as T(p+q)
            collapsed_mult = collections.defaultdict(int)
            for k,v in multiplicities.items():
                collapsed_mult[sum(k)]+=v
            tensors = "+".join(f"{v if v > 1 else ''}T({key})" for key, v in collapsed_mult.items())
        return tensors+f" @ d={self.d}" if self.d is not None else tensors
    def __str__(self):
        return repr(self)
    def __hash__(self):
        return hash(tuple(tuple(ranks) for ranks in self.shapes))

    def symmetric_subspace(self):
        dims,lazy_projection = get_active_subspaces(self.gen,self)
        if len(self.shape)==1:
            return dims,lazy_projection
        else:
            perm = rep_permutation(self)
            return dims, lambda t: lazy_projection(t)[perm].reshape(*self.shape)

    def show_subspace(self):
        dims,projection = self.symmetric_subspace()
        vals = projection(torch.arange(dims).float()+1)
        return torch.where(vals.abs()>1e-7,vals,torch.zeros_like(vals))

    def rho(self,G):
        return sp.linalg.block_diag(*[rho(G,rank) for rank in self.ranks])

    def drho(self,A):
        return sp.linalg.block_diag(*[drho(A,rank) for rank in self.ranks])


def T(p,q=0,gen=None):
    return TensorRep([(p,q)],gen=gen)


Scalar = T(0,0)
Vector = T(1,0)
Matrix = T(1,1)
Quad = T(0,2)

def size(rank,d):
    p,q = rank
    return d**(p+q)

def rho(G,rank):
    p,q = rank
    Gp = functools.reduce(np.kron,p*[G],1)
    GpGinvTq = functools.reduce(np.kron,q*[np.linalg.inv(G).T],Gp)
    return GpGinvTq

def drho(M,rank):
    """ Returns the Lie Algebra representation drho(M) of a matrix M
        acting on a rank (p,q) tensor.
        Inputs: [M (d,d)] [rank tuple(p,q)]
        Outputs: [drho(M) (d**(p+q),d**(p+q))]"""
    p,q = rank
    d=M.shape[0]
    rep_M = 0
    Ikron_powers = [1]
    for _ in range(p+q-1):
        Ikron_powers.append(np.kron(Ikron_powers[-1],np.eye(d)))
    for r in range(1,p+1):
        rep_M += np.kron(np.kron(Ikron_powers[r-1],M),Ikron_powers[p-r+q])
    for s in range(1,q+1):
        rep_M -= np.kron(np.kron(Ikron_powers[p+s-1],M.T),Ikron_powers[q-s])
    return rep_M

def projection_matrix(generators,rank):
    """ Given a sequence of exponential generators [M1,M2,...]
        and a tensor rank (p,q), the function concatenates the representations
        [drho(M1), drho(M2), ...] into a single large projection matrix.
        Input: [generators seq(tensor(d,d))], [rank tuple(p,q)], [d int] """
    drho_Ms = [drho(M,rank) for M in generators]
    P = np.concatenate(drho_Ms,axis=0)
    return P

def orthogonal_complement(proj):
    """ Computes the orthogonal complement to a given matrix proj"""
    U,S,VT = np.linalg.svd(proj,full_matrices=True)
    rank = (S>1e-10).sum()
    return VT[rank:]

def get_active_subspace(generators,rank):
    """ Given an array of generators [M1,M2,...] and tensor rank (p,q)
        this function computes the orthogonal complement to the projection
        matrix formed by stacking the rows of drho(Mi) together.
        Output [Q (r,) + (p+q)*(d,)] """
    if rank ==(0,0): return np.ones((1,1))
    P = projection_matrix(generators,rank)
    Q = orthogonal_complement(P)
    return Q

def rank_permutation(ranks,d):
    """ get the permutation given by converting
        from the order in ranks to the order when the ranks are grouped by
        first occurrence of a given type (p,q)."""
    rank_multiplicities = OrderedCounter(ranks)
    ranks_indices = collections.OrderedDict(((rank, []) for rank in rank_multiplicities))
    i=0
    for (p,q) in ranks:
        tensor_size = d**(p+q)
        ranks_indices[(p,q)].append(torch.arange(tensor_size)+i)
        i+= tensor_size
    permutation = torch.cat([torch.cat([idx for idx in indices]) for indices in ranks_indices.values()])
    return permutation

def get_active_subspaces(generators,rep):
    """ Given a representation which is a sequence of tensors
        with ranks (p_i,q_i), computes the orthogonal complement
        to the projection matrix drho(Mi). Function returns both the
        dimension of the active subspace (r) and also a function that
        maps an array of size (*,r) to a vector v with a representaiton
        given by the rnaks that satisfies drho(Mi)v=0 for each i.
        Inputs: [generators seq(tensor(d,d))] [ranks seq(tuple(p,q))]
        Outputs: [r int] [projection (tensor(*,r)->tensor(*,rep_dim))]"""
    rank_multiplicites = rep.multiplicities()
    Qs = {rank:get_active_subspace(generators,rank) for rank in rank_multiplicites}
    Qs = {rank:torch.from_numpy(Q).cuda().float() for rank,Q in Qs.items()}
    active_dims = sum([rank_multiplicites[rank]*Qs[rank].shape[0] for rank in Qs.keys()])
    # Get the permutation of the vector when grouped by tensor rank
    inverse_perm = torch.argsort(rank_permutation(rep.ranks,rep.d))
    # Apply the projections for each rank, concatenate, and permute back to orig rank order
    def lazy_projection(array):
        i=0
        Ws = []
        for rank, multiplicity in rank_multiplicites.items():
            Qr = Qs[rank].to(array.device)
            i_end = i+multiplicity*Qr.shape[0]
            elems = array[...,i:i_end].reshape(*array.shape[:-1],multiplicity,Qr.shape[0])@Qr
            Ws.append(elems.reshape(*array.shape[:-1],multiplicity*size(rank,rep.d)))
            i = i_end
        Ws = torch.cat(Ws,dim=-1) #concatenate over rep axis
        return Ws[...,inverse_perm] # reorder to original rank ordering
    return active_dims,lazy_projection

# def get_QQT_projection(generators,rep):
#     rank_multiplicites = rep.multiplicities()
#     Qs = {rank:get_active_subspace(generators,rank) for rank in rank_multiplicites}
#     Qs = {rank:torch.from_numpy(Q).cuda().float() for rank,Q in Qs.items()}
#     active_dims = sum([rank_multiplicites[rank]*Qs[rank].shape[0] for rank in Qs.keys()])
#     # Get the permutation of the vector when grouped by tensor rank
#     perm =rank_permutation(rep.ranks,rep.d)
#     inverse_perm = torch.argsort(perm)
#     # Apply the projections for each rank, concatenate, and permute back to orig rank order
#     def lazy_projection_down(array):
#         array[...,perm]
#         i=0
#         Ws = []
#         for rank, multiplicity in rank_multiplicites.items():
#             Qr = Qs[rank].to(array.device)
#             i_end = i+multiplicity*Qr.shape[0]
#             elems = array[...,i:i_end].reshape(*array.shape[:-1],multiplicity,Qr.shape[0])@Qr
#             Ws.append(elems.reshape(*array.shape[:-1],multiplicity*size(rank,rep.d)))
#             i = i_end
#         Ws = torch.cat(Ws,dim=-1) #concatenate over rep axis
#         return Ws[...,inverse_perm] # reorder to original rank ordering
#     # Apply the projections for each rank, concatenate, and permute back to orig rank order
#     def lazy_projection_up(array):
#         i=0
#         Ws = []
#         for rank, multiplicity in rank_multiplicites.items():
#             Qr = Qs[rank].to(array.device)
#             i_end = i+multiplicity*Qr.shape[0]
#             elems = array[...,i:i_end].reshape(*array.shape[:-1],multiplicity,Qr.shape[0])@Qr
#             Ws.append(elems.reshape(*array.shape[:-1],multiplicity*size(rank,rep.d)))
#             i = i_end
#         Ws = torch.cat(Ws,dim=-1) #concatenate over rep axis
#         return Ws[...,inverse_perm] # reorder to original rank ordering
#     return active_dims,lazy_projection


@lru_cache()
def rep_permutation(rep):
    arange = torch.arange(rep.size())
    size_cumsums = [np.cumsum([0] + [rep.d ** (p + q) for (p, q) in reps]) for reps in rep.shapes]
    permutation = torch.zeros(*[cumsum[-1] for cumsum in size_cumsums])
    indices_iter = itertools.product(*[range(len(reps)) for reps in rep.shapes])
    i = 0
    for indices in indices_iter:
        slices = [slice(cumsum[idx], cumsum[idx + 1]) for idx, cumsum in zip(indices, size_cumsums)]
        slice_lengths = [sl.stop - sl.start for sl in slices]
        chunk_size = np.prod(slice_lengths)
        permutation[slices] += arange[i:i + chunk_size].reshape(*slice_lengths)
        i += chunk_size
    return permutation.reshape(-1).long()

@lru_cache() #TODO: pre add self connections in tensor product before random choice (cause of variance)
def capped_tensor_ids(repin,maxrep):
    """Returns rep and ids for tensor product repin@repin
       but with terms >repin removed """
    product_rep = (repin*repin)
    tensor_multiplicities = product_rep.multiplicities()
    multiplicities = maxrep.multiplicities()
    min_mults = collections.OrderedDict((rank,min(tm,multiplicities[rank]))
                                        for rank,tm in tensor_multiplicities.items())
    # randomly select up to maxrep from each of the tensor ranks
    within_ids = collections.OrderedDict((rank,np.random.choice(v,min(v,multiplicities[rank])))
                                                for rank,v in tensor_multiplicities.items())
    all_ids= []
    i_all = 0
    d = repin.d
    for (p,q),ids in within_ids.items():
        interleaved_ids = (d**(p+q)*ids[:,None]+np.arange(d**(p+q))).reshape(-1)
        all_ids.extend(interleaved_ids+i_all)
        i_all += tensor_multiplicities[(p,q)]*d**(p+q)
    sorted_perm = rank_permutation(product_rep.ranks,d)
    out_ranks = []
    for rank,mul in min_mults.items():
        out_ranks.extend(mul*[rank])
    out_rep = TensorRep(out_ranks,gen=repin.gen)
    # have to do some gnarly permuting to account for block ordering vs elemnt ordering
    # and then the multiplicity sorted order and the original ordering
    ids = torch.argsort(rep_permutation(product_rep))[sorted_perm[all_ids]]
    return out_rep,ids