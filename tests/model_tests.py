import jax
from jax import vmap
import numpy as np
from torch.utils.data import DataLoader
from emlp.nn import uniform_rep,MLP,EMLP,Standardize
from equivariance_tests import parametrize,rel_error,scale_adjusted_rel_error
from oil.utils.utils import FixedNumpySeed, FixedPytorchSeed
from emlp.datasets import Inertia,O5Synthetic,ParticleInteraction

# def rel_err(a,b):
#     return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

# def scale_adjusted_rel_err(a,b,g):
#     return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean())+jnp.abs(g-jnp.eye(g.shape[-1])).mean())

def equivariance_err(model,mb,group=None):
    x,y = mb
    group = model.model.G if group is None else group
    gs = group.samples(x.shape[0])
    rho_gin = vmap(model.model.rep_in.rho_dense)(gs)
    rho_gout = vmap(model.model.rep_out.rho_dense)(gs)
    y1 = model((rho_gin@x[...,None])[...,0],training=False)
    y2 = (rho_gout@model(x,training=False)[...,None])[...,0]
    return np.asarray(scale_adjusted_rel_error(y1,y2,gs))

@parametrize([Inertia,O5Synthetic,ParticleInteraction])
def test_init_forward_and_equivariance(dataset):
    network=EMLP
    seed=2021
    bs=50
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        ds = dataset(100)
    model = network(ds.rep_in,ds.rep_out,group=ds.symmetry)
    model = Standardize(model,ds.stats)
    dataloader = DataLoader(ds,batch_size=min(bs,len(ds)),num_workers=0,pin_memory=False)
    mb = next(iter(dataloader))
    mb = jax.device_put(mb[0].numpy()),jax.device_put(mb[1].numpy())
    assert equivariance_err(model,mb) < 1e-4, "EMLP failed equivariance test"
