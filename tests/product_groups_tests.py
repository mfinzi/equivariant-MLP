
import numpy as np#
import copy
from emlp.reps import *
from emlp.groups import *
from emlp.nn import uniform_rep
from equivariance_tests import parametrize,rel_error,scale_adjusted_rel_error
import unittest
from jax import vmap
import jax.numpy as jnp
import logging


@parametrize([(SO(3),S(5)),(S(5),SO(3))])
def test_symmetric_mixed_tensor(G1,G2):
    N=5
    rep = T(2)(G1)*T(1)(G2)
    P = rep.equivariant_projector()
    v = np.random.rand(rep.size())
    v = P@v
    samples = {G1:G1.samples(N),G2:G2.samples(N)}
    gv = (vmap(rep.rho_dense)(samples)*v).sum(-1)
    err = rel_error(gv,v+jnp.zeros_like(gv))
    assert err<3e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}"


@parametrize([(SO(3),S(5)),(S(5),SO(3))])
def test_symmetric_mixed_tensor_sum(G1,G2):
    N=5
    rep = T(2)(G1)*T(1)(G2) + 2*T(0)(G1)*T(2)(G2)+T(1)(G1) +T(1)(G2)
    P = rep.equivariant_projector()
    v = np.random.rand(rep.size())
    v = P@v
    samples = {G1:G1.samples(N),G2:G2.samples(N)}
    gv = (vmap(rep.rho_dense)(samples)*v).sum(-1)
    err = rel_error(gv,v+jnp.zeros_like(gv))
    assert err<3e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}"


@parametrize([(SO(3),S(5)),(S(5),SO(3))])
def test_symmetric_mixed_products(G1,G2):
    N=5
    rep1 = (T(0)+2*T(1)+T(2))(G1)
    rep2 = (T(0)+T(1))(G2)
    rep = rep2*rep1.T
    P = rep.equivariant_projector()
    v = np.random.rand(rep.size())
    v = P@v
    W = v.reshape((rep2.size(),rep1.size()))
    x = np.random.rand(N,rep1.size())
    g1s = G1.samples(N)
    g2s = G2.samples(N)
    ring = vmap(rep1.rho_dense)(g1s)
    routg = vmap(rep2.rho_dense)(g2s)
    gx = (ring@x[...,None])[...,0]
    Wgx =gx@W.T
    gWx = (routg@(x@W.T)[...,None])[...,0]
    equiv_err = rel_error(Wgx,gWx) 
    assert equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}"
    samples = {G1:g1s,G2:g2s}
    gv = (vmap(rep.rho_dense)(samples)*v).sum(-1)
    err = rel_error(gv,v+jnp.zeros_like(gv))
    assert err<3e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}"

@parametrize([(SO(3),S(5)),(S(5),SO(3))])
def test_equivariant_matrix(G1,G2):
    N=5
    repin = T(2)(G2) + 3*T(0)(G1) + T(1)(G2)+2*T(2)(G1)*T(1)(G2)
    repout = (T(1)(G1) + T(2)(G1)*T(0)(G2) + T(1)(G1)*T(1)(G2) + T(0)(G1)+T(2)(G1)*T(1)(G2))
    repW = repout*repin.T
    P = repW.equivariant_projector()
    W = np.random.rand(repout.size(),repin.size())
    W = (P@W.reshape(-1)).reshape(*W.shape)
    
    x = np.random.rand(N,repin.size())
    samples = {G1:G1.samples(N),G2:G2.samples(N)}
    ring = vmap(repin.rho_dense)(samples)
    routg = vmap(repout.rho_dense)(samples)
    gx = (ring@x[...,None])[...,0]
    Wgx =gx@W.T
    #print(g.shape,(x@W.T).shape)
    gWx = (routg@(x@W.T)[...,None])[...,0]
    equiv_err = rel_error(Wgx,gWx)
    assert equiv_err<3e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}"
    # too much memory to run
    # gvecW = (vmap(repW.rho_dense)(samples)*W.reshape(-1)).sum(-1)
    # for i in range(N):
    #     gWerr = rel_error(gvecW[i],W.reshape(-1))
    #     assert gWerr<1e-5,f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G1}x{G2}"