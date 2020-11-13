import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from emlp.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp.equivariant_subspaces import capped_tensor_ids,rep_permutation,bilinear_weights
from emlp.batchnorm import TensorMaskBN,gate_indices
import collections

# class LieLinear(nn.Module):  #
#     def __init__(self, repin, repout):
#         super().__init__()
#         rep_W = repout*repin.T
#         rep_bias = repout

#         Wdim, self.weight_proj = rep_W.symmetric_subspace()
#         self._weight_params = nn.Parameter(torch.randn(Wdim))
#         bias_dim, self.bias_proj = rep_bias.symmetric_subspace()
#         self._bias_params = nn.Parameter(torch.randn(bias_dim))
#         print(f"W components:{rep_W.size()} dim:{Wdim} shape:{rep_W.shape} rep:{rep_W}")
#         print(f"bias components:{rep_bias.size()} dim:{bias_dim} shape:{rep_bias} rep:{rep_bias}")

#     def forward(self, x):
#         W = self.weight_proj(self._weight_params)
#         #print("Using W",W)
#         b = self.bias_proj(self._bias_params)
#         return x@W.T + b

class LieLinear(nn.Linear):  #
    def __init__(self, repin, repout):
        super().__init__(repin.size(),repout.size())
        rep_W = repout*repin.T
        rep_bias = repout
        self.weight_proj = rep_W.symmetric_projection()
        self.bias_proj = rep_bias.symmetric_projection()

    def forward(self, x):
        return F.linear(x,self.weight_proj(self.weight),self.bias_proj(self.bias))

class BiLinear(nn.Module):
    def __init__(self, repin, repout):
        super().__init__()
        rep_W = repout*repin.T
        self.matrix_perm = rep_permutation(rep_W)
        self.W_shape = rep_W.shape
        Wdim, self.weight_proj = bilinear_weights(rep_W,repin)
        self._weight_params = nn.Parameter(torch.randn(Wdim))
        self.random_mask = torch.rand(Wdim)>0#<.1
        print(f"BiW components:{rep_W.size()} dim:{Wdim} shape:{rep_W.shape} rep:{rep_W}")

    def forward(self, x):
        W = self.weight_proj(self._weight_params*self.random_mask.to(x.device),x)[:,self.matrix_perm].reshape(x.shape[0],*self.W_shape)
        #print(x.shape,W.shape)
        return .05*(W@x.unsqueeze(-1)).squeeze(-1)

class TensorBiLinear(LieLinear):
    def __init__(self,repin,repout):
        super().__init__(repin+repout,repout)
        self.bilinear = BiLinear(repin,repout)
    def forward(self,values):
        in_vals = torch.cat([values,self.bilinear(values)/10],dim=-1)
        return super().forward(in_vals)

class Sum(nn.Sequential):
    def __init__(self,*modules):
        super().__init__(*modules)
    def forward(self,*args,**kwargs):
        return sum(mod(*args,**kwargs) for mod in self)


class Sum2(nn.Module):
    def __init__(self,m1,m2):
        super().__init__()
        self.m1=m1
        self.m2=m2
    def forward(self,*args,**kwargs):
        return self.m1(*args,**kwargs)+self.m2(*args,**kwargs)

def gated(rep):#
    return rep+sum([1 for t in rep if t!=(0,0)])*Scalar

class GatedNonlinearity(nn.Module):
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
    def forward(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = gate_scalars.sigmoid() * values[..., :self.rep.size()]
        return activations



def LieLinearBNSwish(repin,repout):
    return nn.Sequential(LieLinear(repin,gated(repout)),
                         TensorMaskBN(gated(repout)),
                         GatedNonlinearity(repout))


# def capped_tensor_product(repin,values,maxrep):
#     capped_tensor_rep,ids = capped_tensor_ids(repin,maxrep)
#     tensored_vals = (values[...,:,None]*values[...,None,:]).reshape(*values.shape[:-1],-1)
#     tensored_vals = tensored_vals[...,ids]
#     return repin+capped_tensor_rep,torch.cat([values,tensored_vals],dim=-1)

class TensorLinear(LieLinear):
    def __init__(self,repin,repout):
        capped_tensor_rep,self.ids = capped_tensor_ids(repin,repout)
        tensored_rep = repin+capped_tensor_rep
        super().__init__(tensored_rep,repout)
    def forward(self,values):
        tensored_vals = (values[...,:,None]*values[...,None,:]).reshape(*values.shape[:-1],-1)/10
        in_vals = torch.cat([values,tensored_vals[...,self.ids]],dim=-1)
        return super().forward(in_vals)

def TensorLinearBNSwish(repin,repout):
    #return nn.Sequential(TensorLinear(repin,gated(repout)),TensorMaskBN(gated(repout)),GatedNonlinearity(repout))
    return nn.Sequential(Sum2(BiLinear(repin,gated(repout)),LieLinear(repin,gated(repout))),
                         TensorMaskBN(gated(repout)),
                         GatedNonlinearity(repout))

class EMLP(nn.Module):
    def __init__(self,rep_in,rep_out,rep_middle,num_layers,algebra):#@
        super().__init__()
        reps = [rep_in(algebra)]+(num_layers-1)*[rep_middle(algebra)]
        self.network = nn.Sequential(
            *[TensorLinearBNSwish(rin,rout) for rin,rout in zip(reps,reps[1:])],#
            TensorLinear(rep_middle(algebra),rep_out(algebra))
        )
    def forward(self,x):
        return self.network(x).squeeze(-1)

class Swish(nn.Module):
    def forward(self,x):
        return x.sigmoid()*x

def LinearBNSwish(cin,cout):
    return nn.Sequential(nn.Linear(cin,cout),nn.BatchNorm1d(cout),Swish())

class MLP(nn.Module):
    def __init__(self,rep_in,rep_out,rep_middle,num_layers,algebra):
        super().__init__()
        cin = rep_in(algebra).size()
        cmid = rep_middle(algebra).size()
        cmid=256
        cout = rep_out(algebra).size()
        self.network = nn.Sequential(
            LinearBNSwish(cin,cmid),
            LinearBNSwish(cmid,cmid),
            LinearBNSwish(cmid,cmid),
            nn.Linear(cmid,cout)
        )
    def forward(self,x):
        return self.network(x).squeeze(-1)