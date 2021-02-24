
import numpy as np#
import copy
from emlp.solver.representation import *
from emlp.solver.groups import *
from emlp.models.mlp import uniform_rep
import unittest
from jax import vmap
import jax.numpy as jnp
import logging
import argparse
import sys
import copy
from functools import partialmethod

def rel_error(t1,t2):
    error = jnp.sqrt(jnp.mean(jnp.absolute(t1-t2)**2))
    scale = jnp.sqrt(jnp.mean(jnp.absolute(t1)**2)) + jnp.sqrt(jnp.mean(jnp.absolute(t2)**2))
    return error/jnp.maximum(scale,1e-7)

def scale_adjusted_rel_error(t1,t2,g):
    error = jnp.sqrt(jnp.mean(jnp.absolute(t1-t2)**2))
    tscale = jnp.sqrt(jnp.mean(jnp.absolute(t1)**2)) + jnp.sqrt(jnp.mean(jnp.absolute(t2)**2))
    gscale = jnp.sqrt(jnp.mean(jnp.absolute(g-jnp.eye(g.shape[-1]))**2))
    scale = jnp.maximum(tscale,gscale)
    return error/jnp.maximum(scale,1e-7)

def expand_cases(cls,argseq):
    def class_decorator(testcase):
        for args in argseq:
            setattr(cls, f"{testcase.__name__}_{args}", partialmethod(testcase,*tuplify(args)))
        return None
    return class_decorator

def tuplify(x):
    if not isinstance(x, tuple): return (x,)
    return x

test_groups = [SO(n) for n in [2,3,4]]+[O(n) for n in [1,2,3,4]]+\
                    [SU(n) for n in [2,3,4]] +\
                    [C(k) for k in [2,3,4,8]]+[D(k) for k in [2,3,4,8]]+\
                    [S(n) for n in [2,5,6]]+[Z(n) for n in [2,5,6]]+\
                    [SO11p(),SO13p(),SO13(),O13()] +[Sp(n) for n in [1,2,3,4]]

class TestRepresentationSubspace(unittest.TestCase): pass
expand_test_cases = partial(expand_cases,TestRepresentationSubspace)


@expand_test_cases(test_groups)
def test_sum(self,G):
    N=5
    rep = T(0,2)+3*(T(0,0)+T(1,0))+T(0,0)+T(1,1)+2*T(1,0)+T(0,2)+T(0,1)+3*T(0,2)+T(2,0)
    #for G in self.test_groups:
    rep = rep(G)
    P = rep.symmetric_projector()
    v = np.random.rand(rep.size())
    v = P@v
    gs = G.samples(N)
    gv = (vmap(rep.rho_dense)(gs)*v).sum(-1)
    err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),gs).mean()
    self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G}")

@expand_test_cases(test_groups)
def test_prod(self,G):
    N=5
    #rep  = T(0,1)*T(0,0)*T(1,0)*T(0,1)*T(0,0)*T(0,1)
    #rep  = T(0,1)*T(1,0)*T(1,0)*T(0,1)
    # rep = T(0,1)*T(0,1)*T(1,0)*T(1,0)
    rep = T(0,1)*T(0,0)*T(2,0)*T(1,0)*T(0,0)**3*T(0,1)**2
    #for G in self.test_groups:
    rep = rep(G)
    # P = rep.symmetric_projector()
    # v = np.random.rand(rep.size())
    # v = P@v
    Q = rep.symmetric_basis()
    v = Q@np.random.rand(Q.shape[-1])
    gs = G.samples(N)
    gv = (vmap(rep.rho_dense)(gs)*v).sum(-1)

    #print(f"g {gs[0]} and rho_dense {rep.rho_dense(gs[0])} {rep.rho_dense(gs[0]).shape}")
    err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),gs).mean()
    self.assertTrue(err<1e-5,f"Symmetric vector fails err {err:.3e} with G={G}")

@expand_test_cases(test_groups)
def test_high_rank_representations(self,G):
    N=5
    r = 10
    for p in range(r+1):
        for q in range(r-p+1):
            if G.num_constraints()*G.d**(3*(p+q))>1e12: continue
            if G.is_orthogonal and q>0: continue
            #try:
            print(p,q,T(p,q))
            rep = T(p,q)(G)
            P = rep.symmetric_projector()
            v = np.random.rand(rep.size())
            v = P@v
            g = vmap(rep.rho_dense)(G.samples(N))
            gv = (g*v).sum(-1)
            #print(f"v{v.shape}, g{g.shape},gv{gv.shape},{G},T{p,q}")
            err = vmap(scale_adjusted_rel_error)(gv,v+jnp.zeros_like(gv),g).mean()
            self.assertTrue(err<1e-4,f"Symmetric vector fails err {err:.3e} with T{p,q} and G={G}")
            logging.info(f"Success with T{p,q} and G={G}")
            # except Exception as e:
            #     print(f"Failed with G={G} and T({p,q})")
            #     raise e

@expand_test_cases([#(SO(3),T(1),2*T(1)),
    #(SO(2),T(1)+2*T(0),T(1)+T(2)+2*T(0)+T(1))])#,
    # (SO(3),T(1)+2*T(0),T(1)+T(2)+2*T(0)+T(1)),
    # (SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
    # (SO(3),5*(T(0)+T(1)),2*(T(0)+T(1))+T(2)+T(1)),
    (SO13p(),T(2)+4*T(1,0)+T(0,1),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1))])           
def test_equivariant_matrix(self,G,repin,repout):
    N=5
    repin = repin(G)
    repout = repout(G)
    #repW = repout*repin.T
    repW = repin>>repout
    P = repW.symmetric_projector()
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
    self.assertTrue(equiv_err<1e-5,f"Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}")

    # print(f"R {repW.rho(gs[0])}")
    # print(f"R1 x R2 {jnp.kron(routg[0],jnp.linalg.inv(ring[0]).T)}")
    gvecW = (vmap(repW.rho_dense)(gs)*W.reshape(-1)).sum(-1)
    gWerr =vmap(scale_adjusted_rel_error)(gvecW,W.reshape(-1)+jnp.zeros_like(gvecW),gs).mean()
    self.assertTrue(gWerr<1e-6,f"Symmetric gvec(W)=vec(W) fails err {gWerr:.3e} with G={G}")

@expand_test_cases([(SO(3),5*T(0)+5*T(1),3*T(0)+T(2)+2*T(1)),
                    (SO13p(),4*T(1,0),10*T(0)+3*T(1,0)+3*T(0,1)+T(0,2)+T(2,0)+T(1,1))])    
def test_bilinear_layer(self,G,repin,repout):
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
    self.assertTrue(equiv_err<1e-6,f"Bilinear Equivariance fails err {equiv_err:.3e} with G={G}")

@expand_test_cases([SO(n) for n in [2,3]]+[O(n) for n in [2,3]]+\
                #[SU(n) for n in [2,3]]+[U(n) for n in [1,2,3]]+\
                [S(n) for n in [5,6]]+[Z(n) for n in [5,6]]+\
                [SO13p(),SO13(),O13()])# + [Sp(n) for n in [1,2,3,4]])
def test_large_representations(self,G): #Currently failing for lorentz and sp groups
    N=5
    ch = 256
    rep =repin=repout= uniform_rep(ch,G)
    repW = rep>>rep
    #print(repin)
    #print(repW)
    P = repW.symmetric_projector()
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
    self.assertTrue(equiv_err<1e-5,f"Large Rep Equivariant gWx=Wgx fails err {equiv_err:.3e} with G={G}")
    logging.info(f"Success with G={G}")

#print(dir(TestRepresentationSubspace))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="warning",help=("Logging Level Example --log debug', default='warning'"))
    options,unknown_args = parser.parse_known_args()#["--log"])
    levels = {'critical': logging.CRITICAL,'error': logging.ERROR,'warn': logging.WARNING,'warning': logging.WARNING,
        'info': logging.INFO,'debug': logging.DEBUG}
    level = levels.get(options.log.lower())
    logging.getLogger().setLevel(level)
    unit_argv = [sys.argv[0]] + unknown_args
    unittest.main(argv=unit_argv)