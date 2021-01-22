import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from emlp_jax.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp_jax.equivariant_subspaces import capped_tensor_ids,rep_permutation,bilinear_weights
from emlp_jax.groups import LearnedGroup
from emlp_jax.batchnorm import TensorMaskBN,gate_indices,MaskBN
import collections
from oil.utils.utils import Named,export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal
from emlp_jax.mlp import Sequential

class ApplyOnComponent(Module):
    def __init__(self,module,dim):#@
        super().__init__()
        self.module = module
        self.dim=dim
        #self.network = LieLinear(self.rep_in,self.rep_out)
    def __call__(self,*x,training=True):
        #print(f"apply on component started up with {len(x)} sized tuple")
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        #print(f"apply on component ended up with {len(x)} sized tuple")
        return tuple(xs)

def swish(x):
    return jax.nn.sigmoid(x)*x

def LinearBNact(chin,chout):
    """assumes that the inputs to the net are shape (bs,n,n,c)"""
    return Sequential(
        ApplyOnComponent(nn.Linear(chin,chout),dim=0),
        MaskBN(chout),
        ApplyOnComponent(swish,dim=0))


def WeightNet(in_dim,out_dim,k=32):
    return Sequential(
        *LinearBNact(in_dim, k),
        *LinearBNact(k, k),
        *LinearBNact(k, out_dim))

class PointConv(Module):
    def __init__(self,chin,chout,mean=True):
        super().__init__()
        self.chin = chin # input channels
        self.cmco_ci = 16 # a hyperparameter controlling size and bottleneck compute cost of weightnet
        self.weightnet = WeightNet(min(chin,5)*2, self.cmco_ci) # MLP - final layer to compute kernel vals (see A1)
        self.linear = nn.Linear(self.cmco_ci * chin, chout)        # final linear layer to compute kernel vals (see A1)
        self.mean=mean  # Whether or not to divide by the number of mc_samples

    def point_convolve(self,mlp_feats,vals,mask,training):
        """ mlp_feats: (bs,nout,nin,d)
            vals: (bs,n,ci)
            mask: (bs,n)"""
        bs, n, ci = vals.shape  # (bs,n,d) -> (bs,n,cm*co/ci)
        nbhd_mask = mask[:,:,None]*mask[:,None,:]
        penult_kernel_weights, _ = self.weightnet(mlp_feats,nbhd_mask,training=training)
        penult_kernel_weights_m = jnp.where(nbhd_mask[...,None],penult_kernel_weights,0*penult_kernel_weights)
        vals_m = jnp.where(mask[...,None],vals,0*vals)
        #      (bs,n,c) -> (bs,c,n) @ (bs,n, nout *cmco/ci) -> (bs,c,nout*cmco/ci) -> (bs,nout, cmco) 
        partial_convolved_vals = (vals.transpose((0,2,1))@penult_kernel_weights_m.transpose((0,2,1,3)).reshape(bs,n,-1)).reshape(bs,ci,n,self.cmco_ci).transpose((0,2,1,3)).reshape(bs,n,-1)
        convolved_vals = self.linear(partial_convolved_vals) #  (bs,n,cmco) -> (bs,n,co)
        if self.mean: convolved_vals /= jax.lax.clamp(1.,nbhd_mask.sum(-1,keepdims=True).astype(np.float32),np.inf)
        return convolved_vals

    def get_mlp_input(self,z_out,z_in):
        b,n,c = z_out.shape
        k = min(c,5)
        zout = jnp.broadcast_to(z_out[:,:,None,:k],(b,n,n,k))
        zin = jnp.broadcast_to(z_in[:,None,:,:k],(b,n,n,k))
        return jnp.concatenate([zout,zin],axis=-1)
    
    def __call__(self,vals,mask,training=True):
        """inputs, and outputs have shape ([vals (bs,n,c)], [mask (bs,n)])"""
        mlp_feats = self.get_mlp_input(vals,vals)
        transformed_vals = self.point_convolve(mlp_feats,vals,mask,training)
        return transformed_vals,mask


@export
def LieConvBNrelu(in_channels,out_channels,**kwargs):
    return Sequential(
        PointConv(in_channels,out_channels,**kwargs),
        MaskBN(out_channels),
        ApplyOnComponent(swish,dim=0)
    )


class BottleBlock(Module):
    """ A bottleneck residual block as described in figure 5"""
    def __init__(self,chin,chout):
        super().__init__()
        assert chin<= chout, f"unsupported channels chin{chin}, chout{chout}. No upsampling atm."
        self.net = Sequential(
            MaskBN(chin),
            ApplyOnComponent(swish,dim=0),
            ApplyOnComponent(nn.Linear(chin,chin//4),dim=0),
            MaskBN(chin//4),
            ApplyOnComponent(swish,dim=0),
            PointConv(chin//4,chout//4),
            MaskBN(chout//4),
            ApplyOnComponent(swish,dim=0),
            ApplyOnComponent(nn.Linear(chout//4,chout),dim=0),
        )
        self.chin = chin
    def __call__(self,vals,mask,training=True):
        #print("Started bottle call")
        new_values, mask = self.net(vals,mask,training=training)
        new_values = jax.ops.index_add(new_values, jax.ops.index[:self.chin], vals)
        #new_values[...,:self.chin] += vals
        #print("Finished bottle call")
        return (new_values, mask)

class GlobalPool(Module):
    """computes values reduced over all spatial locations (& group elements) in the mask"""
    def __init__(self,mean=True):
        super().__init__()
        self.mean = mean
        
    def __call__(self,vals,mask,training=True):
        """x [vals (bs,n,c), mask (bs,n)]"""
        summed = jnp.where(mask[...,None],vals,0*vals).sum(1)
        if self.mean:
            summed /= mask.sum(-1)[...,None]
        return summed

class ResNet(Module):
    def __init__(self, chin, cout, k=256, num_layers=6, mean=True, **kwargs):
        super().__init__()
        if isinstance(k,int):
            k = [k]*(num_layers+1)
        self.net = Sequential(
            MaskBN(chin),
            ApplyOnComponent(nn.Linear(chin,k[0]),dim=0), #embedding layer
            *[BottleBlock(k[i],k[i+1]) for i in range(num_layers)],
            MaskBN(k[-1]),
            ApplyOnComponent(swish,dim=0),
            ApplyOnComponent(nn.Linear(k[-1],cout),dim=0),
            GlobalPool(mean=mean),
            )

    def __call__(self, inp,training=True):
        vals,mask=inp
        out = self.net(vals,mask,training)#[...,0]
        #print(out[0].shape)
        #assert False, f"{out.shape}"
        return out
