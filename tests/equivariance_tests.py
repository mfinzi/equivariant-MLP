
import numpy as np#
import copy
from emlp.reps import *
from emlp.groups import *
from emlp.nn import uniform_rep
import pytest#import unittest
from jax import vmap
import jax.numpy as jnp
import logging
import argparse
import sys
import copy
import inspect
#from functools import partialmethod,partial

def rel_error(t1,t2):
    error = jnp.sqrt(jnp.mean(jnp.abs(t1-t2)**2))
    scale = jnp.sqrt(jnp.mean(jnp.abs(t1)**2)) + jnp.sqrt(jnp.mean(jnp.abs(t2)**2))
    return error/jnp.maximum(scale,1e-7)

def scale_adjusted_rel_error(t1,t2,g):
    error = jnp.sqrt(jnp.mean(jnp.abs(t1-t2)**2))
    tscale = jnp.sqrt(jnp.mean(jnp.abs(t1)**2)) + jnp.sqrt(jnp.mean(jnp.abs(t2)**2))
    gscale = jnp.sqrt(jnp.mean(jnp.abs(g-jnp.eye(g.shape[-1]))**2))
    scale = jnp.maximum(tscale,gscale)
    return error/jnp.maximum(scale,1e-7)

def equivariance_error(W,repin,repout,G):
    """ Computes the equivariance relative error rel_err(Wρ₁(g),ρ₂(g)W)
        of the matrix W (dim(repout),dim(repin))
        according to the input and output representations and group G. """
    N=5
    x = np.random.rand(N,repin.size())
    gs = G.samples(N)
    ring = vmap(repin.rho_dense)(gs)
    routg = vmap(repout.rho_dense)(gs)
    equiv_err = scale_adjusted_rel_error(W@ring,routg@W,gs)
    return equiv_err

def strip_parens(string):
    return string.replace('(','').replace(')','')

def parametrize(cases,ids=None):
    """ Expands test cases with pytest.mark.parametrize but with argnames 
        assumed and ids given by the ids=[str(case) for case in cases] """
    def decorator(test_fn):
        argnames = ','.join(inspect.getfullargspec(test_fn).args)
        theids = [strip_parens(str(case)) for case in cases] if ids is None else ids
        return pytest.mark.parametrize(argnames,cases,ids=theids)(test_fn)
    return decorator
# def expand_cases(cls,argseq):
#     def class_decorator(testcase):
#         for args in argseq:
#             setattr(cls, f"{testcase.__name__}_{args}", partialmethod(testcase,*tuplify(args)))
#             #setattr(cls,f"{testcase.__name__}_{args}",types.MethodType(partial(testcase,*tuplify(args)),cls))
#         return testcase
#     return class_decorator

# def tuplify(x):
#     if not isinstance(x, tuple): return (x,)
#     return x

test_groups = [SO(n) for n in [2,3,4]]+[O(n) for n in [2,3,4]]+\
                    [SU(n) for n in [2,3,4]] + [U(n) for n in [2,3,4]] + \
                    [SL(n) for n in [2,3,4]] + [GL(n) for n in [2,3,4]] + \
                    [C(k) for k in [2,3,4,8]]+[D(k) for k in [2,3,4,8]]+\
                    [S(n) for n in [2,4,6]]+[Z(n) for n in [2,4,6]]+\
                    [SO11p(),SO13p(),SO13(),O13()] +[Sp(n) for n in [1,3]]+\
                    [RubiksCube(),Cube(),ZksZnxZn(2,2),ZksZnxZn(4,4)]
# class TestRepresentationSubspace(unittest.TestCase): pass
# expand_test_cases = partial(expand_cases,TestRepresentationSubspace)


#@pytest.mark.parametrize("G",test_groups,ids=test_group_names)
@parametrize(test_groups)
def test_sum(G):
    N=5
    rep = T(0,2)+3*(T(0,0)+T(1,0))+T(0,0)+T(1,1)+2*T(1,0)+T(0,2)+T(0,1)+3*T(0,2)+T(2,0)
    rep = rep(G)
    if G.num_constraints()*rep.size()>1e11 or rep.size()**2>10**7: return
    P = rep.equivariant_projector()
    v = np.random.rand(rep.size())
    v = P@v
    gs = G.samples(N)
    gv = (vmap(rep.rho_dense)(gs)*v).sum(-1)
    err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),gs).mean()
    assert err<1e-4,f"Symmetric vector fails err {err:.3e} with G={G}"

#@pytest.mark.parametrize("G",test_groups,ids=test_group_names)
@parametrize([group for group in test_groups if group.d<5])
def test_prod(G):
    N=5
    rep = T(0,1)*T(0,0)*T(1,0)**2*T(1,0)*T(0,0)**3*T(0,1)
    rep = rep(G)
    if G.num_constraints()*rep.size()>1e11 or rep.size()**2>10**7: return
    # P = rep.equivariant_projector()
    # v = np.random.rand(rep.size())
    # v = P@v
    Q = rep.equivariant_basis()
    v = Q@np.random.rand(Q.shape[-1])
    gs = G.samples(N)
    gv = (vmap(rep.rho_dense)(gs)*v).sum(-1)

    #print(f"g {gs[0]} and rho_dense {rep.rho_dense(gs[0])} {rep.rho_dense(gs[0]).shape}")
    err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),gs).mean()
    assert err<1e-4,f"Symmetric vector fails err {err:.3e} with G={G}"

#@pytest.mark.parametrize("G",test_groups,ids=test_group_names)
@parametrize(test_groups)
def test_high_rank_representations(G):
    N=5
    r = 10
    for p in range(r+1):
        for q in range(r-p+1):
            if G.num_constraints()*G.d**(3*(p+q))>1e11: continue
            if G.is_orthogonal and q>0: continue
            #try:
            #logging.info(f"{p},{q},{T(p,q)}")
            rep = T(p,q)(G)
            P = rep.equivariant_projector()
            v = np.random.rand(rep.size())
            v = P@v
            g = vmap(rep.rho_dense)(G.samples(N))
            gv = (g*v).sum(-1)
            #print(f"v{v.shape}, g{g.shape},gv{gv.shape},{G},T{p,q}")
            err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),g).mean()
            if np.isnan(err): continue  # deal with nans on cpu later
            assert err<1e-4,f"Symmetric vector fails err {err:.3e} with T{p,q} and G={G}"
            logging.info(f"Success with T{p,q} and G={G}")
            # except Exception as e:
            #     print(f"Failed with G={G} and T({p,q})")
            #     raise e

@parametrize([
    (SO(3),T(1)+2*T(0),T(1)+T(2)+2*T(0)+T(1)),
    (SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
    (SO(3),5*(T(0)+T(1)),2*(T(0)+T(1))+T(2)+T(1)),
    (SO(4), T(1)+2*T(2),(T(0)+T(3))*T(0)),
    (SO13p(),T(2)+4*T(1,0)+T(0,1),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1)),
    (Sp(2),(V+2*V**2)*(V.T+1*V).T + V.T,3*V**0 + V + V*V.T),
    (SU(3),T(2,0)+T(1,1)+T(0)+2*T(0,1),T(1,1)+V+V.T+T(0)+T(2,0)+T(0,2))
    ])
def test_equivariant_matrix(G,repin,repout):
    N=5
    repin = repin(G)
    repout = repout(G)
    #repW = repout*repin.T
    repW = repin>>repout
    P = repW.equivariant_projector()
    W = np.random.rand(repout.size(),repin.size())
    W = (P@W.reshape(-1)).reshape(*W.shape)
    
    x = np.random.rand(N,repin.size())
    gs = G.samples(N)
    ring = vmap(repin.rho_dense)(gs)
    routg = vmap(repout.rho_dense)(gs)
    gx = (ring@x[...,None])[...,0]
    Wgx =gx@W.T
    #print(g.shape,(x@W.T).shape)
    gWx = (routg@(x@W.T)[...,None])[...,0]
    equiv_err = rel_error(Wgx,gWx)
    assert equiv_err<1e-4,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}"

    # print(f"R {repW.rho(gs[0])}")
    # print(f"R1 x R2 {jnp.kron(routg[0],jnp.linalg.inv(ring[0]).T)}")
    gvecW = (vmap(repW.rho_dense)(gs)*W.reshape(-1)).sum(-1)
    gWerr =vmap(scale_adjusted_rel_error)(gvecW,W.reshape(-1)+jnp.zeros_like(gvecW),gs).mean()
    assert gWerr<1e-4,f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G}"

@parametrize([(SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
                    (SO13p(),4*T(1,0),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1))])    
def test_bilinear_layer(G,repin,repout):
    N=5
    repin = repin(G)
    repout = repout(G)
    repW = repout*repin.T
    Wdim,P = bilinear_weights(repout,repin)
    x = np.random.rand(N,repin.size())
    gs = G.samples(N)
    ring = vmap(repin.rho_dense)(gs)
    routg = vmap(repout.rho_dense)(gs)
    gx = (ring@x[...,None])[...,0]
    
    W = np.random.rand(Wdim)
    W_x = P(W,x)
    Wxx = (W_x@x[...,None])[...,0]
    gWxx = (routg@Wxx[...,None])[...,0]
    Wgxgx =(P(W,gx)@gx[...,None])[...,0]
    equiv_err = rel_error(Wgxgx,gWxx)
    assert equiv_err<1e-4,f"Bilinear Equivariance fails err {equiv_err:.3e} with G={G}"

@parametrize(test_groups)
def test_large_representations(G):
    N=5
    ch = 256
    rep =repin=repout= uniform_rep(ch,G)
    repW = rep>>rep
    P = repW.equivariant_projector()
    W = np.random.rand(repout.size(),repin.size())
    W = (P@W.reshape(-1)).reshape(*W.shape)
    
    x = np.random.rand(N,repin.size())
    gs = G.samples(N)
    ring = vmap(repin.rho_dense)(gs)
    routg = vmap(repout.rho_dense)(gs)
    gx = (ring@x[...,None])[...,0]
    Wgx =gx@W.T
    #print(g.shape,(x@W.T).shape)
    gWx = (routg@(x@W.T)[...,None])[...,0]
    equiv_err = vmap(scale_adjusted_rel_error)(Wgx,gWx,gs).mean()
    assert equiv_err<1e-4,f"Large Rep Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}"
    logging.info(f"Success with G={G}")

# #print(dir(TestRepresentationSubspace))
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--log", default="warning",help=("Logging Level Example --log debug', default='warning'"))
#     options,unknown_args = parser.parse_known_args()#["--log"])
#     levels = {'critical': logging.CRITICAL,'error': logging.ERROR,'warn': logging.WARNING,'warning': logging.WARNING,
#         'info': logging.INFO,'debug': logging.DEBUG}
#     level = levels.get(options.log.lower())
#     logging.getLogger().setLevel(level)
#     unit_argv = [sys.argv[0]] + unknown_args
#     unittest.main(argv=unit_argv)