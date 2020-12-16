import jax
import jax.numpy as jnp
import haiku as hk#import objax.nn as nn
import numpy as np
from emlp.equivariant_subspaces_jax import T,Scalar,Vector,Matrix,Quad,size
from emlp.equivariant_subspaces_jax import capped_tensor_ids,rep_permutation,bilinear_weights
from emlp.batchnorm_jax import TensorMaskBN,gate_indices
import collections
from oil.utils.utils import Named
import scipy as sp
import scipy.special
import random
import logging

class LieLinear(hk.Module):  #
    def __init__(self, repin, repout,name=None):
        super().__init__(name)
        cin,cout = repin.size(),repout.size()
        scale= 1. / np.sqrt(cin)
        w_init = hk.initializers.TruncatedNormal(scale)
        self.W = hk.get_parameter("w", shape=[cout, cin], init=w_init)
        self.b = hk.get_parameter("b", shape=[cout], init=hk.initializers.RandomUniform(-scale,scale))
        #print("Linear sizes:",repin.size(),repout.size())
        rep_W = repout*repin.T
        rep_bias = repout
        self.weight_proj = rep_W.symmetric_projection()
        self.bias_proj = rep_bias.symmetric_projection()
        logging.info(f"Linear W components:{rep_W.size()} shape:{rep_W.shape} rep:{rep_W}")
    def __call__(self, x,is_training=True): # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        W = self.weight_proj(self.W)
        b = self.bias_proj(self.b)
        out = x@W.T+b
        logging.debug(f"linear out shape:{out.shape}")
        return out

class BiLinear(hk.Module):
    def __init__(self, repin, repout,name=None):
        super().__init__(name)
        rep_W = repout*repin.T
        self.matrix_perm = rep_permutation(rep_W)
        self.W_shape = rep_W.shape
        Wdim, self.weight_proj = bilinear_weights(rep_W,repin)
        w_init = hk.initializers.TruncatedNormal(1./np.sqrt(Wdim))
        self._weight_params = hk.get_parameter("w",shape=[Wdim],init=w_init)
        logging.info(f"BiW components:{rep_W.size()} dim:{Wdim} shape:{rep_W.shape} rep:{rep_W}")

    def __call__(self, x,is_training=True):
        logging.debug(f"bilinear in shape: {x.shape}")
        W = self.weight_proj(self._weight_params,x)[:,self.matrix_perm].reshape(-1,*self.W_shape)
        out= .05*(W@x[...,None])[...,0]
        #import pdb; pdb.set_trace()
        logging.debug(f"bilinear out shape: {out.shape}")
        return out

class Sequential(hk.Sequential):
    def __call__(self,*args,**kwargs):
        for mod in self.layers:
            args = (mod(*args,**kwargs),)
            #print(args[0].shape)
        return args[0]

class Sum(hk.Sequential):
    def __call__(self,*args,**kwargs):
        return sum(mod(*args,**kwargs) for mod in self.layers)


def gated(rep):
    return rep+sum([1 for t in rep.ranks if t!=(0,0)])*Scalar

class GatedNonlinearity(hk.Module):
    def __init__(self,rep,name=None):
        super().__init__(name)
        self.rep=rep
    def __call__(self,values,is_training=True):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return activations


def LieLinearBNSwish(repin,repout):
    return Sequential([LieLinear(repin,gated(repout)),
                         TensorMaskBN(gated(repout)),
                         GatedNonlinearity(repout)])

def TensorLinearBNSwish(repin,repout): #TODO: investigate BiLinear after LieLinear instead of parallel
    #return nn.Sequential(TensorLinear(repin,gated(repout)),TensorMaskBN(gated(repout)),GatedNonlinearity(repout))
    return Sequential([Sum([BiLinear(repin,gated(repout)),LieLinear(repin,gated(repout))]),
                         TensorMaskBN(gated(repout)),
                         GatedNonlinearity(repout)])

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

class EMLP(hk.Module):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3,name=None):#@
        super().__init__(name)
        logging.warning("Initing EMLP")
        repmiddle = uniform_rep(ch,group)
        reps = [rep_in(group)]+num_layers*[repmiddle]
        logging.debug(reps)
        self.network = hk.Sequential(
            [TensorLinearBNSwish(rin,rout) for rin,rout in zip(reps,reps[1:])]+
            [LieLinear(repmiddle,rep_out(group))]
        )
    def __call__(self,x,is_training=True):
        return self.network(x,is_training=is_training).squeeze(-1)

def swish(x):
    return jax.nn.sigmoid(x)*x

def LinearBNSwish(cout):
    return hk.Sequential([hk.Linear(cout),hk.BatchNorm(True,True,.9),swish])

def mlp(repin,repout,group,ch,num_layers):
    return hk.nets.MLP(num_layers*[ch]+[repout.size()],activation=swish)

class MLP(hk.Module):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.chs = num_layers*[ch]
        self.cout = rep_out(group).size()
        logging.warning("Initing MLP")
    def __call__(self,x,is_training):
        for c in self.chs:
            x = hk.Linear(c)(x)
            x = hk.BatchNorm(True,True,.9)(x,is_training=is_training)
            x = swish(x)
        x = hk.Linear(self.cout)(x)
        return x.squeeze(-1) if self.cout==1 else x