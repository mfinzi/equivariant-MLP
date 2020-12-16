import torch
import torch.nn as nn
import copy
import numpy as np
import objax.nn as nn
import jax
import jax.numpy as jnp
from emlp.equivariant_subspaces_jax import size
import logging
import objax.functional as F

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

class TensorMaskBN(nn.BatchNorm0D): #TODO find discrepancies with pytorch version
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,rep):
        super().__init__(rep.size(),momentum=0.9)
        self.rep=rep

    def __call__(self,x,training):
        #logging.debug(f"bn input shape{inp.shape}")

        if training:
            m = x.mean(self.redux, keepdims=True)
            v = (x ** 2).mean(self.redux, keepdims=True) - m ** 2
            self.running_mean.value += (1 - self.momentum) * (m - self.running_mean.value)
            self.running_var.value += (1 - self.momentum) * (v - self.running_var.value)
        else:
            m, v = self.running_mean.value, self.running_var.value
        smask = jax.device_put(scalar_mask(self.rep))
        y = jnp.where(smask,self.gamma.value * (x - m) * F.rsqrt(v + self.eps) + self.beta.value,x)
        return y # switch to or (x-m)


