import numpy as np
import torch
import collections,itertools
from functools import lru_cache


def T(p,q):
    return [(p,q)]
Scalar = T(0,0)
Vector = T(1,0)
Matrix = T(1,1)
Quad = T(0,2)

def get_representation_matrix(M,rank,d=2):
    """ Returns the Lie Algebra representation drho(M) of a matrix M
        acting on a rank (p,q) tensor.
        Inputs: [M (d,d)] [rank tuple(p,q)]
        Outputs: [drho(M) (d**(p+q),d**(p+q))]"""
    p,q = rank
    rep_M = 0
    Ikron_powers = [1]
    for _ in range(p+q-1):
        Ikron_powers.append(np.kron(Ikron_powers[-1],np.eye(d)))
    for r in range(1,p+1):
        rep_M += np.kron(np.kron(Ikron_powers[r-1],M),Ikron_powers[p-r+q])
    for s in range(1,q+1):
        rep_M -= np.kron(np.kron(Ikron_powers[p+s-1],M.T),Ikron_powers[q-s])
    return rep_M

def get_projection_matrix(generators,rank,d=2):
    """ Given a sequence of exponential generators [M1,M2,...]
        and a tensor rank (p,q), the function concatenates the representations
        [drho(M1), drho(M2), ...] into a single large projection matrix.
        Input: [generators seq(tensor(d,d))], [rank tuple(p,q)], [d int] """
    drho_Ms = [get_representation_matrix(M,rank,d) for M in generators]
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
    d = generators[0].shape[0]
    P = get_projection_matrix(generators,rank,d)
    Q = orthogonal_complement(P)
    tensor_shape = sum(rank)*(d,)
    return Q

class OrderedCounter(collections.Counter,collections.OrderedDict): pass

def repsize(rep,d):
    return sum(d**(p+q) for (p,q) in rep)

def inverse_rank_permutation(ranks,d):
    """ get the inverse permutation for the permutation given by converting
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
    inverse_permutation = torch.argsort(permutation)
    return inverse_permutation

def get_active_subspaces(generators,ranks):
    """ Given a representation which is a sequence of tensors
        with ranks (p_i,q_i), computes the orthogonal complement
        to the projection matrix drho(Mi). Function returns both the
        dimension of the active subspace (r) and also a function that
        maps an array of size (*,r) to a vector v with a representaiton
        given by the rnaks that satisfies drho(Mi)v=0 for each i.
        Inputs: [generators seq(tensor(d,d))] [ranks seq(tuple(p,q))]
        Outputs: [r int] [projection (tensor(*,r)->tensor(*,rep_dim))]"""
    rank_multiplicites = OrderedCounter(ranks)
    Qs = {rank:get_active_subspace(generators,rank) for rank in rank_multiplicites}
    Qs = {rank:torch.from_numpy(Q).cuda().float() for rank,Q in Qs.items()}
    active_dims = sum([rank_multiplicites[rank]*Qs[rank].shape[0] for rank in Qs.keys()])
    # Get the permutation of the vector when grouped by tensor rank
    d = generators[0].shape[0]
    inverse_perm = inverse_rank_permutation(ranks,d)
    # APply the projections for each rank, concatenate, and permute back to orig rank order
    def lazy_projection(array):
        i=0
        Ws = []
        for (p,q), multiplicity in rank_multiplicites.items():
            Qr = Qs[(p,q)].to(array.device)
            i_end = i+multiplicity*Qr.shape[0]
            elems = array[...,i:i_end].reshape(*array.shape[:-1],multiplicity,Qr.shape[0])@Qr
            Ws.append(elems.reshape(*array.shape[:-1],multiplicity*d**(p+q)))
            i = i_end
        Ws = torch.cat(Ws,dim=-1) #concatenate over rep axis
        return Ws[...,inverse_perm] # reorder to original rank ordering
    return active_dims,lazy_projection


def _multilinear_reshape(flat_W,rep_out_list,rep_in_list,d=2):
    all_reps = rep_out_list+rep_in_list
    size_cumsums = [np.cumsum([0]+[d**(p+q) for (p,q) in reps]) for reps in all_reps]
    reshaped_W = torch.zeros(*[cumsum[-1] for cumsum in size_cumsums], device=flat_W.device,dtype=flat_W.dtype)
    indices_iter = itertools.product(*[range(len(reps)) for reps in all_reps])
    i=0
    for indices in indices_iter:
        slices = [slice(cumsum[idx],cumsum[idx+1]) for idx,cumsum in zip(indices,size_cumsums)]
        slice_lengths = [sl.stop-sl.start for sl in slices]
        chunk_size = np.prod(slice_lengths)
        reshaped_W[slices] += flat_W[i:i+chunk_size].reshape(*slice_lengths)
        i+=chunk_size
    return reshaped_W

@lru_cache()
def multilinear_reshape_permutation(rep_out_list,rep_in_list,d=2):
    """ Computes the permutation for the multilinear reshape. rep_out_list and
        rep_in_list should be tuples to make use of the caching."""
    size = np.prod([repsize(rep,d) for rep in rep_out_list+rep_in_list])
    arange = torch.arange(size)
    permutation = _multilinear_reshape(arange,rep_out_list,rep_in_list,d=d).reshape(-1).long()
    return permutation

def multilinear_reshape(flat_W,rep_out_list,rep_in_list,d=2):
    tuple_outlist = tuple(tuple(rep) for rep in rep_out_list)
    tuple_inlist = tuple(tuple(rep) for rep in rep_in_list)
    perm = multilinear_reshape_permutation(tuple_outlist,tuple_inlist,d=d)
    sizes = [repsize(rep,d) for rep in rep_out_list+rep_in_list]
    permuted_W = flat_W[perm].reshape(*sizes)
    return permuted_W

def multilinear_active_subspaces(generators,rep_out_list,rep_in_list):
    """ Like get_active_subspaces, this function returns function that maps
        to the space of multilinear maps that map from the input representations
        to the output representations.
        Inputs: [generators seq(tensor(d,d))] [rep_out_list seq(seq(tuple(q,p)))] [rep_in_list]
        Outputs r, fn(tensor(r)-> tensor(rep_out1,...,rep_outn,...,rep_in1,...,rep_inn))"""
    transposed_in_reps = [[(q,p) for (p,q) in rep] for rep in rep_in_list]
    rank_tuples = itertools.product(*rep_out_list,*transposed_in_reps)
    combined_rep = [tuple(sum(qps) for qps in zip(*ranks)) for ranks in rank_tuples]
    active_dims,active_combined = get_active_subspaces(generators,combined_rep)
    d = generators[0].shape[0]
    rep_sizes = [repsize(rep,d) for rep in rep_out_list+rep_in_list]
    active_combined_reshaped = lambda t: multilinear_reshape(active_combined(t),rep_out_list,rep_in_list,d)
    return active_dims,active_combined_reshaped

def matrix_active_subspaces(generators,rep_out,rep_in):
    """ Like get_active_subspaces, this function returns function that maps
        to the space of multilinear maps that map from the input representations
        to the output representations.
        Inputs: [generators seq(tensor(d,d))] [rep_out_list seq(seq(tuple(q,p)))] [rep_in_list]
        Outputs r, fn(tensor(r)-> tensor(rep_out1,...,rep_outn,...,rep_in1,...,rep_inn))"""
    return multilinear_active_subspaces(generators,[rep_out],[rep_in])