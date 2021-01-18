
import numpy as np#
import copy
from emlp_jax.equivariant_subspaces import *
from emlp_jax.groups import *
from emlp_jax.mlp import uniform_rep
import unittest
from jax import vmap

def rel_error(t1,t2):
    return np.mean(np.abs(t1-t2))/(np.mean(np.abs(t1)) + np.mean(np.abs(t2))+1e-7)


class TestRepresentationSubspace(unittest.TestCase):
    test_groups = [SO(n) for n in [1,2,3,4]]+[O(n) for n in [1,2,3,4]]+\
                    [SU(n) for n in [2,3,4]]+[U(n) for n in [1,2,3,4]]+\
                    [C(k) for k in [2,3,4,8]]+[D(k) for k in [2,3,4,8]]+\
                    [Permutation(n) for n in [2,5,6]]+\
                    [DiscreteTranslation(n) for n in [2,5,6]]+[SO13p()] ##[Symplectic(n) for n in [1,2,3,4]]+


    # def test_symmetric_vector(self):
    #     N=5
    #     rep = T(0,0)+T(1,0)+T(0,0)+T(1,1)+T(1,0)+T(0,2)
    #     for G in self.test_groups:
    #         rep = rep(G)
    #         P = rep.symmetric_projection()
    #         v = np.random.rand(rep.size())
    #         v = P(v)
    #         gv = (vmap(rep.rho)(G.samples(N))*v).sum(-1)
    #         for i in range(N):
    #             err = rel_error(gv[i],v)
    #             self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G}")

    # def test_equivariant_matrix(self):
    #     N=5
    #     testcases = [(SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
    #                 (SO13p(),4*T(1,0),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1))]
        
    #     for G,repin,repout in testcases:
    #         repin = repin(G)
    #         repout = repout(G)
    #         repW = repout*repin.T
    #         P = repW.symmetric_projection()
    #         W = np.random.rand(repout.size(),repin.size())
    #         W = P(W)
            
    #         x = np.random.rand(N,repin.size())
    #         gs = G.samples(N)
    #         ring = vmap(repin.rho)(gs)
    #         routg = vmap(repout.rho)(gs)
    #         gx = (ring@x[...,None])[...,0]
    #         Wgx =gx@W.T
    #         #print(g.shape,(x@W.T).shape)
    #         gWx = (routg@(x@W.T)[...,None])[...,0]
    #         equiv_err = rel_error(Wgx,gWx)
    #         self.assertTrue(equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}")
    # #         gvecW = (vmap(repW.rho)(G.samples(N))*W.reshape(-1)).sum(-1)
    # #         for i in range(N):
    # #             gWerr = rel_error(gvecW[i],W.reshape(-1))
    # #             #self.assertTrue(gWerr<1e-6,f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G}")
    # #             #TODO fix composite representation matrix? will require reordering ranks


    # def test_bilinear_layer(self):
    #     N=5
    #     testcases = [(SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
    #                 (SO13p(),4*T(1,0),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1))]
        
    #     for G,repin,repout in testcases:
    #         repin = repin(G)
    #         repout = repout(G)
    #         repW = repout*repin.T
    #         Wdim,P = bilinear_weights(repW,repin)
    #         x = np.random.rand(N,repin.size())
    #         gs = G.samples(N)
    #         ring = vmap(repin.rho)(gs)
    #         routg = vmap(repout.rho)(gs)
    #         gx = (ring@x[...,None])[...,0]
            
    #         W = np.random.rand(Wdim)
    #         W_x = P(W,x)
    #         Wxx = (W_x@x[...,None])[...,0]
    #         gWxx = (routg@Wxx[...,None])[...,0]
    #         Wgxgx =(P(W,gx)@gx[...,None])[...,0]
    #         equiv_err = rel_error(Wgxgx,gWxx)
    #         self.assertTrue(equiv_err<1e-6,f"Bilinear Equivariance fails err {equiv_err:.3e} with G={G}")
    # def test_high_rank_representations(self):
    #     N=5
    #     r = 6
    #     G = SO13p()
    #     for i in range(r+1):
    #         for j in range(r-i+1):
    #             rep = T(i,j)(G)
    #             P = rep.symmetric_projection()
    #             v = np.random.rand(rep.size())
    #             v = P(v)
    #             gv = (vmap(rep.rho)(G.samples(N))*v).sum(-1)
    #             err = rel_error(gv,v+np.zeros_like(gv))
    #             self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with T({i,j}) and G={G}")
    def test_large_representations(self):
        N=5
        ch = 256
        for G in self.test_groups:#self.test_groups:
            rep =repin=repout= uniform_rep(ch,G)
            repW = rep*rep.T
            P = repW.symmetric_projection()
            W = np.random.rand(repout.size(),repin.size())
            W = P(W)
            
            x = np.random.rand(N,repin.size())
            gs = G.samples(N)
            ring = vmap(repin.rho)(gs)
            routg = vmap(repout.rho)(gs)
            gx = (ring@x[...,None])[...,0]
            Wgx =gx@W.T
            #print(g.shape,(x@W.T).shape)
            gWx = (routg@(x@W.T)[...,None])[...,0]
            equiv_err = rel_error(Wgx,gWx)
            self.assertTrue(equiv_err<1e-5,f"Large Rep Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}")

if __name__ == '__main__':
    unittest.main()