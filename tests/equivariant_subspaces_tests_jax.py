
import numpy as np#
import copy
from emlp_jax.equivariant_subspaces import *
from emlp_jax.groups import *
import unittest
from jax import vmap

def rel_error(t1,t2):
    return np.mean(np.abs(t1-t2))/(np.mean(np.abs(t1)) + np.mean(np.abs(t2))+1e-7)


class TestRepresentationSubspace(unittest.TestCase):
    test_groups = [SO(n) for n in [1,2,3,4]]+[O(n) for n in [1,2,3,4]]+\
                    [C(k) for k in [2,3,4,8]]+[D(k) for k in [2,3,4,8]]+\
                    [Permutation(n) for n in [2,5,6]]+\
                    [DiscreteTranslation(n) for n in [2,5,6]]+[SO13p()] ##[Symplectic(n) for n in [1,2,3,4]]+


    # def test_symmetric_vector(self):
    #     N=5
    #     rep = T(0,0)+T(1,0)+T(0,0)+T(1,1)+T(1,0)+T(0,2)
    #     for G in self.test_groups:
    #         rep = rep(G)
    #         P = rep.symmetric_projection()
    #         v = np.random.randn(rep.size())
    #         v = P(v)
    #         gv = (vmap(rep.rho)(G.samples(N))*v).sum(-1)
    #         for i in range(N):
    #             err = rel_error(gv[i],v)
    #             self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G}")

    def test_equivariant_matrix(self):
        N=5
        repin = 5*T(0)+5*T(1)
        repout = T(2)
        for G in [SO(3)]:
            repin = repin(G)
            repout = repout(G)
            repW = repout*repin.T
            P = repW.symmetric_projection()
            W = np.random.randn(repout.size(),repin.size())
            W = P(W)
            
            x = np.random.randn(N,repin.size())
            gs = G.samples(N)
            ring = vmap(repin.rho)(gs)
            routg = vmap(repout.rho)(gs)
            gx = (ring@x[...,None])[...,0]
            Wgx =gx@W.T
            #print(g.shape,(x@W.T).shape)
            gWx = (routg@(x@W.T)[...,None])[...,0]
            equiv_err = rel_error(Wgx,gWx)
            self.assertTrue(equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}")
            gvecW = (vmap(repW.rho)(G.samples(N))*W.reshape(-1)).sum(-1)
            for i in range(N):
                gWerr = rel_error(gvecW[i],W.reshape(-1))
                self.assertTrue(gWerr<1e-5,f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G}")
                



if __name__ == '__main__':
    unittest.main()