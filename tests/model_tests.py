import jax
from jax import vmap
import numpy as np
import pytest
from torch.utils.data import DataLoader
from emlp.nn import uniform_rep,MLP,EMLP,Standardize
import emlp
from equivariance_tests import parametrize,rel_error,scale_adjusted_rel_error
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from emlp.datasets import Inertia,O5Synthetic,ParticleInteraction
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

@parametrize([Inertia,O5Synthetic,ParticleInteraction])
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

