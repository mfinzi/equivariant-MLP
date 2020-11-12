import torch
import torch.nn as nn
import numpy as np
from emlp.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp.equivariant_subspaces import capped_tensor_ids
from emlp.batchnorm import TensorMaskBN,gate_indices
import collections

class LieLinear(nn.Module):  #
    def __init__(self, repin, repout):
        super().__init__()
        rep_W = repout*repin.T
        rep_bias = repout

        Wdim, self.weight_proj = rep_W.symmetric_subspace()
        self._weight_params = nn.Parameter(torch.randn(Wdim))
        bias_dim, self.bias_proj = rep_bias.symmetric_subspace()
        self._bias_params = nn.Parameter(torch.randn(bias_dim))
        print(f"W components:{rep_W.size()} dim:{Wdim} shape:{rep_W.shape} rep:{rep_W}")
        print(f"bias components:{rep_bias.size()} dim:{bias_dim} shape:{rep_bias} rep:{rep_bias}")

    def forward(self, x):
        W = self.weight_proj(self._weight_params)
        #print("Using W",W)
        b = self.bias_proj(self._bias_params)
        return x@W.T + b

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
    #return nn.Sequential(TensorLinear(repin,repout),TensorMaskBN(repout))
    return nn.Sequential(TensorLinear(repin,gated(repout)),
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