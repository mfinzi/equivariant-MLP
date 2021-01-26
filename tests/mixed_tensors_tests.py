
import numpy as np#
import copy
from emlp_jax.equivariant_subspaces import *
from emlp_jax.mixed_tensors import *
from emlp_jax.groups import *
from emlp_jax.mlp import uniform_rep
import unittest
from jax import vmap
import jax.numpy as jnp
import logging

def rel_error(t1,t2):
    return jnp.mean(jnp.abs(t1-t2))/(jnp.mean(jnp.abs(t1)) + jnp.mean(jnp.abs(t2))+1e-7)

def scale_adjusted_rel_error(t1,t2,g):
    return jnp.mean(jnp.abs(t1-t2))/(jnp.mean(jnp.abs(t1)) + jnp.mean(jnp.abs(t2))+jnp.mean(jnp.abs(g-jnp.eye(g.shape[-1])))+1e-7)

class TestRepresentationSubspace(unittest.TestCase):
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
            gv = (vmap(rep.rho)(samples)*v).sum(-1)
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
            gv = (vmap(rep.rho)(samples)*v).sum(-1)
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
            ring = vmap(rep1.rho)(g1s)
            routg = vmap(rep2.rho)(g2s)
            gx = (ring@x[...,None])[...,0]
            Wgx =gx@W.T
            #print(g.shape,(x@W.T).shape)
            gWx = (routg@(x@W.T)[...,None])[...,0]
            equiv_err = rel_error(Wgx,gWx)
            self.assertTrue(equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G1}x{G2}")
            # samples = {G1:g1s,G2:g2s}
            # gv = (vmap(rep.rho)(samples)*v).sum(-1)
            # err = rel_error(gv,v+jnp.zeros_like(gv))
            # self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G1}x{G2}")
    # def test_high_rank_representations(self):
    #     N=5
    #     r = 10
    #     for G in [SO13p(),SO13(),O13()]:#self.test_groups:
    #         for p in range(r+1):
    #             for q in range(r-p+1):
    #                 if G.num_constraints()*G.d**(3*(p+q))>1e12: continue
    #                 if G.is_orthogonal_rep() and q>0: continue
    #                 #try:
    #                 rep = T(p,q)(G)
    #                 P = rep.symmetric_projector()
    #                 v = np.random.rand(rep.size())
    #                 v = P@v
    #                 g = vmap(rep.rho)(G.samples(N))
    #                 gv = (g*v).sum(-1)
    #                 #print(f"v{v.shape}, g{g.shape},gv{gv.shape},{G},T{p,q}")
    #                 err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),g).mean()
    #                 self.assertTrue(err<3e-5,f"Symmetric vector fails err {err:.3e} with T{p,q} and G={G}")
    #                 print(f"Success with T{p,q} and G={G}")
    #                 # except Exception as e:
    #                 #     print(f"Failed with G={G} and T({p,q})")
    #                 #     raise e
                    
    # def test_equivariant_matrix(self):
    #     N=5
    #     testcases = [(SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
    #                 (SO13p(),4*T(1,0),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1))]
        
    #     for G,repin,repout in testcases:
    #         repin = repin(G)
    #         repout = repout(G)
    #         repW = repout*repin.T
    #         P = repW.symmetric_projector()
    #         W = np.random.rand(repout.size(),repin.size())
    #         W = (P@W.reshape(-1)).reshape(*W.shape)
            
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

    # def test_large_representations(self):
    #     N=5
    #     ch = 256
    #     test_groups = [SO(n) for n in [2,3]]+[O(n) for n in [2,3]]+\
    #                 [SU(n) for n in [2,3]]+[U(n) for n in [1,2,3]]+\
    #                 [Permutation(n) for n in [5,6]]+\
    #                 [DiscreteTranslation(n) for n in [5,6]]
    #     for G in test_groups:#self.test_groups:
    #         rep =repin=repout= uniform_rep(ch,G)
    #         repW = rep*rep.T
    #         P = repW.symmetric_projector()
    #         W = np.random.rand(repout.size(),repin.size())
    #         W = (P@W.reshape(-1)).reshape(*W.shape)
            
    #         x = np.random.rand(N,repin.size())
    #         gs = G.samples(N)
    #         ring = vmap(repin.rho)(gs)
    #         routg = vmap(repout.rho)(gs)
    #         gx = (ring@x[...,None])[...,0]
    #         Wgx =gx@W.T
    #         #print(g.shape,(x@W.T).shape)
    #         gWx = (routg@(x@W.T)[...,None])[...,0]
    #         equiv_err = rel_error(Wgx,gWx)
    #         self.assertTrue(equiv_err<1e-5,f"Large Rep Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}")
    #         print(f"Success with G={G}")

if __name__ == '__main__':
    unittest.main()