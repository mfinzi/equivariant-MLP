import torch
import torch.nn as nn
import copy
import numpy as np
from emlp.equivariant_subspaces import size

def gate_indices(rep):
    channels = rep.size()
    indices = np.arange(channels)
    num_nonscalars = 0
    for i, rank in enumerate(rep):
        if rank!=(0,0):
            indices[i:i+size(rank,rep.d)] = channels+num_nonscalars
            num_nonscalars+=1
    return indices

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
        x_or_zero = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))  # replace elements outside mask
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
        scalar_mask = torch.from_numpy(gate_indices(self.rep)==np.arange(self.rep.size())).to(device=x.device)
        ratio = torch.where(scalar_mask,self.weight / std,torch.ones_like(self.weight))
        output = (x_or_zero * ratio + (self.bias - xmean * ratio))
        if self.on:
            out_dict = copy.copy(inp)
            out_dict[self.on] = output
            return out_dict
        else:
            return output


