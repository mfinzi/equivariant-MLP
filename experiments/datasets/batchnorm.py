import torch
import torch.nn as nn
import copy
import numpy as np
import objax.nn as nn
import jax
from jax import jit
import jax.numpy as jnp
from emlp.reps import Scalar
from emlp.reps.product_sum_reps import SumRep
import logging
import objax.functional as F
from functools import partial
import objax
from functools import lru_cache as cache



@cache(maxsize=None)
def gate_indices(sumrep): #TODO: add support for mixed_tensors
    """ Indices for scalars, and also additional scalar gates
        added by gated(sumrep)"""
    assert isinstance(sumrep,SumRep), f"unexpected type for gate indices {type(sumrep)}"
    channels = sumrep.size()
    perm = sumrep.perm
    indices = np.arange(channels)
    num_nonscalars = 0
    i=0
    for rep in sumrep:
        if rep!=Scalar and not rep.is_regular:
            indices[perm[i:i+rep.size()]] = channels+num_nonscalars
            num_nonscalars+=1
        i+=rep.size()
    return indices

@cache(maxsize=None)
def scalar_mask(sumrep):
    channels = sumrep.size()
    perm = sumrep.perm
    mask = np.ones(channels)>0
    i=0
    for rep in sumrep:
        if rep!=Scalar: mask[perm[i:i+rep.size()]] = False
        i+=rep.size()
    return mask

@cache(maxsize=None)
def regular_mask(sumrep):
    channels = sumrep.size()
    mask = np.ones(channels)<0
    i=0
    for rep in sumrep:
        if rep.is_regular: mask[perm[i:i+rep.size()]] = True
        i+=rep.size()
    return mask

@export
class TensorBN(nn.BatchNorm0D): #TODO: add suport for mixed tensors.
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,rep):
        super().__init__(rep.size(),momentum=0.9)
        self.rep=rep
    def __call__(self,x,training): #TODO: support elementwise for regular reps
        #return x #DISABLE BN, harms performance!! !!
        smask = jax.device_put(scalar_mask(self.rep))
        if training:
            m = x.mean(self.redux, keepdims=True)
            v = (x ** 2).mean(self.redux, keepdims=True) - m ** 2
            v = jnp.where(smask,v,ragged_gather_scatter((x ** 2).mean(self.redux),self.rep)) #in non scalar indices, divide by sum squared
            self.running_mean.value += (1 - self.momentum) * (m - self.running_mean.value)
            self.running_var.value += (1 - self.momentum) * (v - self.running_var.value)
        else:
            m, v = self.running_mean.value, self.running_var.value
        y = jnp.where(smask,self.gamma.value * (x - m) * F.rsqrt(v + self.eps) + self.beta.value,x*F.rsqrt(v+self.eps))#(x-m)*F.rsqrt(v + self.eps))
        return y # switch to or (x-m)

class MaskBN(nn.BatchNorm0D):
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,ch):
        super().__init__(ch,momentum=0.9)

    def __call__(self,vals,mask,training=True):
        sum_dims = list(range(len(vals.shape[:-1])))
        x_or_zero = jnp.where(mask[...,None],vals,0*vals)
        if training:
            num_valid = mask.sum(sum_dims)
            m = x_or_zero.sum(sum_dims)/num_valid
            v = (x_or_zero ** 2).sum(sum_dims)/num_valid - m ** 2
            self.running_mean.value += (1 - self.momentum) * (m - self.running_mean.value)
            self.running_var.value += (1 - self.momentum) * (v - self.running_var.value)
        else:
            m, v = self.running_mean.value, self.running_var.value
        return ((x_or_zero-m)*self.gamma.value*F.rsqrt(v + self.eps) + self.beta.value,mask)

class TensorMaskBN(nn.BatchNorm0D): #TODO find discrepancies with pytorch version
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,rep):
        super().__init__(rep.size(),momentum=0.9)
        self.rep=rep
    def __call__(self,x,mask,training):
        sum_dims = list(range(len(vals.shape[:-1])))
        x_or_zero = jnp.where(mask[...,None],vals,0*vals)
        smask = jax.device_put(scalar_mask(self.rep))
        if training:
            num_valid = mask.sum(sum_dims)
            m = x_or_zero.sum(sum_dims)/num_valid
            x2 = (x_or_zero ** 2).sum(sum_dims)/num_valid
            v =  x2 - m ** 2
            v = jnp.where(smask,v,ragged_gather_scatter(x2,self.rep))
            self.running_mean.value += (1 - self.momentum) * (m - self.running_mean.value)
            self.running_var.value += (1 - self.momentum) * (v - self.running_var.value)
        else:
            m, v = self.running_mean.value, self.running_var.value
        y = jnp.where(smask,self.gamma.value * (x_or_zero - m) * F.rsqrt(v + self.eps) + \
                            self.beta.value,x_or_zero*F.rsqrt(v+self.eps))
        return y,mask # switch to or (x-m)

# @partial(jit,static_argnums=(1,))
# def ragged_gather_scatter(x,x_rep):
#     y = []
#     i=0
#     for rep in x_rep.reps: # sum -> mean
#         y.append(x[i:i+rep.size()].mean(keepdims=True).repeat(rep.size(),axis=-1))
#         i+=rep.size()
#     return jnp.concatenate(y,-1)

@partial(jit,static_argnums=(1,))
def ragged_gather_scatter(x,x_rep):
    perm = x_rep.argsort()
    invperm = np.argsort(perm)
    x_sorted = x[perm]
    i=0
    y=[]
    for rep, multiplicity in x_rep.multiplicities().items():
        i_end = i+multiplicity*rep.size()
        y.append(x_sorted[i:i_end].reshape(multiplicity,rep.size()).mean(-1,keepdims=True).repeat(rep.size(),axis=-1).reshape(-1))
        i = i_end
    return jnp.concatenate(y)[invperm]