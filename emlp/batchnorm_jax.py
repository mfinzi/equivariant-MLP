import torch
import torch.nn as nn
import copy
import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from emlp.equivariant_subspaces_jax import size
import logging

def gate_indices(rep):
    channels = rep.size()
    indices = np.arange(channels)
    num_nonscalars = 0
    i=0
    for rank in rep.ranks:
        if rank!=(0,0):
            indices[i:i+size(rank,rep.d)] = channels+num_nonscalars
            num_nonscalars+=1
        i+=size(rank,rep.d)
    return indices

def scalar_mask(rep):
    channels = rep.size()
    mask = np.ones(channels)>0
    i=0
    for rank in rep.ranks:
        if rank!=(0,0): mask[i:i+size(rank,rep.d)] = False
        i+=size(rank,rep.d)
    return mask

class TensorMaskBN(hk.BatchNorm):
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,rep,on=None):
        super().__init__(create_scale=True,create_offset=True,decay_rate=0.9)
        self.rep=rep
        self.on=on
        self.scale = hk.get_parameter("scale", [rep.size()], init=jnp.ones)
        self.offset = hk.get_parameter("offset", [rep.size()], init =jnp.zeros)
    def __call__(self,inp,is_training=True,test_local_stats=False):
        logging.debug(f"bn input shape{inp.shape}")
        if self.on:
            x = inp[self.on]
            mask = inp.get("mask",jnp.ones_like(x[...,0])>0)
        else:
            x = inp
            mask = jnp.ones_like(x[...,0])>0
        #x_or_zero = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))  # replace elements outside mask
        x_or_zero=x
        if is_training or test_local_stats:
            sum_dims = list(range(len(x.shape[:-1])))
            xsum = x_or_zero.sum(sum_dims)
            xxsum = (x_or_zero * x_or_zero).sum(sum_dims)
            numel_mask = (mask).sum()
            xmean = xsum / numel_mask
            sumvar = xxsum - xsum * xmean
            unbias_var = sumvar / (numel_mask - 1)
            bias_var = sumvar / numel_mask
        else:
            xmean, bias_var = self.mean_ema.average, self.var_ema.average
        if is_training:
            self.mean_ema(xmean)
            self.var_ema(unbias_var)

        eps = jax.lax.convert_element_type(self.eps, bias_var.dtype)
        std = jax.lax.clamp(eps,bias_var,np.inf)** 0.5
        smask = jax.device_put(scalar_mask(self.rep))
        #ratio = torch.where(mask,self.weight / std,torch.ones_like(self.weight))
        #output = (x_or_zero * ratio + (self.bias - xmean * ratio)*mask)
        output = jnp.where(smask,x_or_zero*self.scale/std + (self.offset-xmean*self.scale/std),x_or_zero)
        logging.debug(f"bn output shape: {output.shape}")
        if self.on:
            out_dict = copy.copy(inp)
            out_dict[self.on] = output
            return out_dict
        else:
            return output


