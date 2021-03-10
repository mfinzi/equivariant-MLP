
import numpy as np#
import copy
from emlp.solver.representation import *
from emlp.solver.groups import *
from emlp.models.mlp import uniform_rep
import unittest
from jax import vmap
import jax.numpy as jnp
import logging

def rel_error(t1,t2):
    return jnp.mean(jnp.abs(t1-t2))/(jnp.mean(jnp.abs(t1)) + jnp.mean(jnp.abs(t2))+1e-7)

def scale_adjusted_rel_error(t1,t2,g):
    return jnp.mean(jnp.abs(t1-t2))/(jnp.mean(jnp.abs(t1)) + jnp.mean(jnp.abs(t2))+jnp.mean(jnp.abs(g-jnp.eye(g.shape[-1])))+1e-7)

class TestRepresentationsMultipleGroups(unittest.TestCase):
    # test_groups = [SO(n) for n in [1,2,3,4]]+[O(n) for n in [1,2,3,4]]+\
    #                 [SU(n) for n in [2,3,4]]+[U(n) for n in [1,2,3,4]]+\
    #                 [C(k) for k in [2,3,4,8]]+[D(k) for k in [2,3,4,8]]+\
    #                 [Permutation(n) for n in [2,5,6]]+\
    #                 [DiscreteTranslation(n) for n in [2,5,6]]+[SO13p(),SO13(),O13()] ##[Symplectic(n) for n in [1,2,3,4]]+

    def test_symmetric_mixed_tensor(self):
        N=5
        test_groups = [(SO(3),S(5)),(S(5),SO(3))]
        for (G1,G2) in test_groups:
            rep = T(2)(G1)*T(1)(G2)
            P = rep.symmetric_projector()
            v = np.random.rand(rep.size())
            v = P@v
            samples = {G1:G1.samples(N),G2:G2.samples(N)}
            gv = (vmap(rep.rho_dense)(samples)*v).sum(-1)
            err = rel_error(gv,v+jnp.zeros_like(gv))
            self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}")

    def test_symmetric_mixed_tensor_sum(self):
        N=5
        test_groups = [(SO(3),S(5)),(S(5),SO(3))]
        for (G1,G2) in test_groups:
            rep = T(2)(G1)*T(1)(G2) + 2*T(0)(G1)*T(2)(G2)+T(1)(G1) +T(1)(G2)
            P = rep.symmetric_projector()
            v = np.random.rand(rep.size())
            v = P@v
            samples = {G1:G1.samples(N),G2:G2.samples(N)}
            gv = (vmap(rep.rho_dense)(samples)*v).sum(-1)
            err = rel_error(gv,v+jnp.zeros_like(gv))
            self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}")

    def test_symmetric_mixed_products(self):
        N=5
        test_groups = [(SO(3),S(5)),(S(5),SO(3))]
        for (G1,G2) in test_groups:
            rep1 = (T(0)+2*T(1)+T(2))(G1)
            rep2 = (T(0)+T(1))(G2)
            rep = rep2*rep1.T
            P = rep.symmetric_projector()
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
            self.assertTrue(equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}")
            samples = {G1:g1s,G2:g2s}
            gv = (vmap(rep.rho_dense)(samples)*v).sum(-1)
            err = rel_error(gv,v+jnp.zeros_like(gv))
            self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}")
    
    def test_equivariant_matrix(self):
        N=5
        test_groups = [(SO(3),S(5)),(S(5),SO(3))]
        for G1,G2 in test_groups:
            repin = T(2)(G2) + 3*T(0)(G1) + T(1)(G2)+2*T(2)(G1)*T(1)(G2)
            repout = (T(1)(G1) + T(2)(G1)*T(0)(G2) + T(1)(G1)*T(1)(G2) + T(0)(G1)+T(2)(G1)*T(1)(G2))
            repW = repout*repin.T
            P = repW.symmetric_projector()
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
            self.assertTrue(equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}")
            # gvecW = (vmap(repW.rho_dense)(samples)*W.reshape(-1)).sum(-1)
            # for i in range(N):
            #     gWerr = rel_error(gvecW[i],W.reshape(-1))
            #     self.assertTrue(gWerr<1e-6,f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G1}x{G2}")


if __name__ == '__main__':
    unittest.main()