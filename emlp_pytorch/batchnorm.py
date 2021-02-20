import torch
import torch.nn as nn
import copy
import numpy as np
from emlp.equivariant_subspaces import size

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
    mask = torch.ones(channels)>0
    i=0
    for rank in rep.ranks:
        if rank!=(0,0): mask[i:i+size(rank,rep.d)] = False
        i+=size(rank,rep.d)
    return mask

class TensorMaskBN(nn.BatchNorm1d):
    """ Equivariant Batchnorm for tensor representations.
        Applies BN on Scalar channels and Mean only BN on others """
    def __init__(self,rep,on=None):
        super().__init__(rep.size())
        self.rep=rep
        self.on=on
    def forward(self,inp):
        if self.on:
            x = inp[self.on]
            mask = inp.get("mask",torch.ones_like(x[...,0])>0)
        else:
            x = inp
            mask = torch.ones_like(x[...,0])>0
        #x_or_zero = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))  # replace elements outside mask
        x_or_zero=x
        if self.training or not self.track_running_stats:
            sum_dims = list(range(len(x.shape[:-1])))
            xsum = x_or_zero.sum(dim=sum_dims)
            xxsum = (x_or_zero * x_or_zero).sum(dim=sum_dims)
            numel_mask = (mask).sum()
            xmean = xsum / numel_mask
            sumvar = xxsum - xsum * xmean
            unbias_var = sumvar / (numel_mask - 1)
            bias_var = sumvar / numel_mask
            self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * xmean.detach())
            self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * unbias_var.detach())
        else:
            xmean, bias_var = self.running_mean, self.running_var
        std = bias_var.clamp(self.eps) ** 0.5
        mask = scalar_mask(self.rep).to(x.device)
        #ratio = torch.where(mask,self.weight / std,torch.ones_like(self.weight))
        #output = (x_or_zero * ratio + (self.bias - xmean * ratio)*mask)
        output = torch.where(mask,x_or_zero*self.weight/std + (self.bias-xmean*self.weight/std),x_or_zero)
        if self.on:
            out_dict = copy.copy(inp)
            out_dict[self.on] = output
            return out_dict
        else:
            return output


