import torch
import torch.nn as nn
from emlp.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp.batchnorm import TensorMaskBN,gate_indices

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
        print("Using W",W)
        b = self.bias_proj(self._bias_params)
        return x@W.T# + b

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


class EMLP(nn.Module):
    def __init__(self,rep_in,rep_out,rep_middle,num_layers,algebra):#@
        super().__init__()
        reps = [rep_in(algebra)]+(num_layers-1)*[rep_middle(algebra)]
        self.network = nn.Sequential(
            *[LieLinearBNSwish(rin,rout) for rin,rout in zip(reps,reps[1:])],#
            LieLinear(rep_middle,rep_out(algebra))
        )
    def forward(self,x):
        return self.network(x).squeeze(-1)



# def nonscalar_arange(rep,d):
#     indices = []
#     i = 0
#     for rank in rep:
#         if rank[0] == Scalar:
#             indices.append(-1)
#         else:
#             indices.extend(size(rank, d) * [i])
#             i += 1
#     return indices

#
# def gated_nonlinearity(values, gate_scalars, rep,d):
#     indices=  nonscalar_arange(rep,d)
#     expanded_gate_scalars = gate_scalars[...,indices]
#     gated_vals = expanded_gate_scalars.sigmoid()*values
#     swished_vals = values.sigmoid()*values
#     scalar_mask = torch.zeros(values.shape[-1]+1,device=values.device)
#     scalar_mask[-1]=1
#     scalar_mask = scalar_mask[indices].reshape(*(len(values.shape[:-1])*[1]),-1).expand_as(values)
#     result = torch.where(scalar_mask>0,swished_vals,gated_vals)
#     return result




