import jax
from jax import vmap
import numpy as np
import pytest
from torch.utils.data import DataLoader
from emlp.nn import uniform_rep,MLP,EMLP,Standardize
import emlp
from equivariance_tests import parametrize,rel_error,scale_adjusted_rel_error
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from emlp.datasets import Inertia,O5Synthetic,ParticleInteraction,InvertedCube
from jax import random


# def rel_err(a,b):
#     return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

# def scale_adjusted_rel_err(a,b,g):
#     return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean())+jnp.abs(g-jnp.eye(g.shape[-1])).mean())

def equivariance_err(model,mb,repin,repout,group):
    x,y = mb
    gs = group.samples(x.shape[0])
    rho_gin = vmap(repin(group).rho_dense)(gs)
    rho_gout = vmap(repout(group).rho_dense)(gs)
    y1 = model((rho_gin@x[...,None])[...,0])
    y2 = (rho_gout@model(x)[...,None])[...,0]
    return np.asarray(scale_adjusted_rel_error(y1,y2,gs))

def get_dsmb(dsclass):
    seed=2021
    bs=50
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        ds = dsclass(100)
    dataloader = DataLoader(ds,batch_size=min(bs,len(ds)),num_workers=0,pin_memory=False)
    mb = next(iter(dataloader))
    mb = jax.device_put(mb[0].numpy()),jax.device_put(mb[1].numpy())
    return ds,mb

@parametrize([Inertia,O5Synthetic,ParticleInteraction,InvertedCube])
def test_init_forward_and_equivariance(dsclass):
    network=emlp.nn.objax.EMLP
    ds,mb = get_dsmb(dsclass)
    model = network(ds.rep_in,ds.rep_out,group=ds.symmetry)
    assert equivariance_err(model,mb,ds.rep_in,ds.rep_out,ds.symmetry) < 1e-4, "Objax EMLP failed equivariance test"

@parametrize([Inertia])
def test_haiku_emlp(dsclass):
    import haiku as hk
    from emlp.nn.haiku import EMLP as hkEMLP
    network = hkEMLP
    ds,mb = get_dsmb(dsclass)
    net = network(ds.rep_in,ds.rep_out,group=ds.symmetry)
    net = hk.without_apply_rng(hk.transform(net))
    params = net.init(random.PRNGKey(42),mb[0])
    model = lambda x: net.apply(params,x)
    assert equivariance_err(model,mb,ds.rep_in,ds.rep_out,ds.symmetry) < 1e-4, "Haiku EMLP failed equivariance test"

@parametrize([Inertia])
def test_flax_emlp(dsclass):
    from emlp.nn.flax import EMLP as flaxEMLP
    network = flaxEMLP
    ds,mb = get_dsmb(dsclass)
    net = network(ds.rep_in,ds.rep_out,group=ds.symmetry)
    params = net.init(random.PRNGKey(42),mb[0])
    model = lambda x: net.apply(params,x)
    assert equivariance_err(model,mb,ds.rep_in,ds.rep_out,ds.symmetry) < 1e-4, "flax EMLP failed equivariance test"

@parametrize([Inertia])
def test_pytorch_emlp(dsclass):
    import torch
    from emlp.nn.pytorch import EMLP as ptEMLP
    network=ptEMLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds,mb = get_dsmb(dsclass)
    net = network(ds.rep_in,ds.rep_out,group=ds.symmetry).to(device)
    model = lambda x: jax.device_put(net(torch.from_numpy(np.asarray(x)).to(device)).cpu().data.numpy())
    assert equivariance_err(model,mb,ds.rep_in,ds.rep_out,ds.symmetry) < 1e-4, "Pytorch EMLP failed equivariance test"


from emlp.reps import vis, sparsify_basis, V,Rep
from emlp.groups import S,SO

def test_utilities():
    W = V(SO(3))
    vis(W,W)
    Q = (W**2>>W).equivariant_basis()
    SQ = sparsify_basis(Q)
    A = SQ@(1+np.arange(SQ.shape[-1]))
    nunique = len(np.unique(np.abs(A)))
    assert nunique in (SQ.shape[-1],SQ.shape[-1]+1), "Sparsify failes on SO(3) T3"


def test_bespoke_representations():
    class ProductSubRep(Rep):
        def __init__(self,G,subgroup_id,size):
            """   Produces the representation of the subgroup of G = G1 x G2
                with the index subgroup_id in {0,1} specifying G1 or G2.
                Also requires specifying the size of the representation given by G1.d or G2.d """
            self.G = G
            self.index = subgroup_id
            self._size = size
        def __str__(self):
            return "V_"+str(self.G).split('x')[self.index]
        def size(self):
            return self._size
        def rho(self,M): 
            # Given that M is a LazyKron object, we can just get the argument
            return M.Ms[self.index]
        def drho(self,A):
            return A.Ms[self.index]
        def __call__(self,G):
            # adding this will probably not be necessary in a future release,
            # necessary now because rep is __call__ed in nn.EMLP constructor
            assert self.G==G
            return self
    G1,G2 = SO(3),S(5)
    G = G1 * G2

    VSO3 = ProductSubRep(G,0,G1.d)
    VS5 = ProductSubRep(G,1,G2.d)
    Vin = VS5 + V(G)
    Vout = VSO3
    str(Vin>>Vout)
    model = emlp.nn.EMLP(Vin, Vout, group=G)
    input_point = np.random.randn(Vin.size())*10
    from emlp.reps.linear_operators import LazyKron
    lazy_G_sample = LazyKron([G1.sample(),G2.sample()])

    out1 = model(Vin.rho(lazy_G_sample)@input_point)
    out2 = Vout.rho(lazy_G_sample)@model(input_point)
    assert rel_error(out1,out2) < 1e-4, "EMLP equivariance fails on bespoke productsubrep"
