import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from emlp.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp.equivariant_subspaces import capped_tensor_ids,rep_permutation,bilinear_weights
from emlp.batchnorm import TensorMaskBN,gate_indices
import collections
from oil.utils.utils import Named
import scipy as sp
import scipy.special
import random

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
        #print("Linear sizes:",repin.size(),repout.size())
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
        W = self.weight_proj(self._weight_params,x)[:,self.matrix_perm].reshape(x.shape[0],*self.W_shape)
        #print(x.shape,W.shape)
        return .05*(W@x.unsqueeze(-1)).squeeze(-1)

class BiLinearNew(nn.Module):
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
        return .05*self.weight_proj(self._weight_params,x)

class TensorBiLinear(LieLinear):
    def __init__(self,repin,repout):
        super().__init__(repin+repout,repout)
        self.bilinear = BiLinear(repin,repout)
    def forward(self,values):
        in_vals = torch.cat([values,self.bilinear(values)/10],dim=-1)
        return super().forward(in_vals)

class Sum2(nn.Sequential):
    def __init__(self,*modules):
        super().__init__(*modules)
    def forward(self,*args,**kwargs):
        return sum(mod(*args,**kwargs) for mod in self)


class Sum(nn.Module):
    def __init__(self,m1,m2):
        super().__init__()
        self.m1=m1
        self.m2=m2
    def forward(self,*args,**kwargs):
        a1 = self.m1(*args,**kwargs)
        a2 = self.m2(*args,**kwargs)
        #print(a1.shape,a2.shape)
        return a1+a2

def gated(rep):#
    return rep+sum([1 for t in rep.ranks if t!=(0,0)])*Scalar

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
    return nn.Sequential(Sum(BiLinear(repin,gated(repout)),LieLinear(repin,gated(repout))),
                         TensorMaskBN(gated(repout)),
                         GatedNonlinearity(repout))

def uniform_rep(ch,group):
    """ A heuristic method for allocating a given number of channels (ch)
        into tensor types. Attempts to distribute the channels evenly across
        the different tensor types. """
    d = group.d
    Ns = np.zeros((lambertW(ch,d)+1,),int) # number of tensors of each rank
    while ch>0:
        max_rank = lambertW(ch,d) # compute the max rank tensor that can fit up to
        Ns[:max_rank+1] += np.array([d**(max_rank-r) for r in range(max_rank+1)],dtype=int)
        ch -= (max_rank+1)*d**max_rank # compute leftover channels
    return sum(uniform_allocation(nr,r)(group) for r,nr in enumerate(Ns))

def lambertW(ch,d):
    """ Returns solution to x*d^x = ch rounded down."""
    max_rank=0
    while (max_rank+1)*d**max_rank <= ch:
        max_rank += 1
    max_rank -= 1
    return max_rank

def uniform_allocation(N,rank):
    """ Uniformly allocates N of tensors of total rank r=(p+q) into
        T(k,r-k) for k=0,1,...,r. For unimodular groups there is no
        distinction between p and q, so this op is equivalent to N*T(rank)."""
    if N==0: return 0
    even_split = sum((N//(rank+1))*T(k,rank-k) for k in range(rank+1))
    ragged = sum(random.sample([T(k,rank-k) for k in range(rank+1)],N%(rank+1)))
    return even_split+ragged

class EMLP(nn.Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        super().__init__()
        repmiddle = uniform_rep(ch,group)
        reps = [rep_in(group)]+num_layers*[repmiddle]
        print(reps)
        self.network = nn.Sequential(
            *[TensorLinearBNSwish(rin,rout) for rin,rout in zip(reps,reps[1:])],#
            TensorLinear(repmiddle,rep_out(group))
        )
    def forward(self,x):
        return self.network(x).squeeze(-1)

class Swish(nn.Module):
    def forward(self,x):
        return x.sigmoid()*x

def LinearBNSwish(cin,cout):
    return nn.Sequential(nn.Linear(cin,cout),nn.BatchNorm1d(cout),Swish())

class MLP(nn.Module,metaclass=Named):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        cin = rep_in(group).size()
        chs = [cin]+num_layers*[ch]
        cout = rep_out(group).size()
        self.network = nn.Sequential(
            *[LinearBNSwish(c1,c2) for c1,c2 in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def forward(self,x):
        return self.network(x).squeeze(-1)