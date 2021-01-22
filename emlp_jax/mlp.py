import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from emlp_jax.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp_jax.equivariant_subspaces import capped_tensor_ids,rep_permutation,bilinear_weights
from emlp_jax.groups import LearnedGroup
from emlp_jax.batchnorm import TensorBN,gate_indices
import collections
from oil.utils.utils import Named,export
import scipy as sp
import scipy.special
import random
import logging
from objax.variable import TrainVar, StateVar
from objax.nn.init import kaiming_normal, xavier_normal
from objax.module import Module
import objax
from objax.nn.init import orthogonal

def Sequential(*args):
    return nn.Sequential(args)

class LieLinear(nn.Linear):  #
    def __init__(self, repin, repout):
        nin,nout = repin.size(),repout.size()
        super().__init__(nin,nout)
        self.b = TrainVar(objax.random.uniform((nout,))/jnp.sqrt(nout))
        self.w = TrainVar(orthogonal((nout, nin)))
        #print("Linear sizes:",repin.size(),repout.size())
        self.rep_W = rep_W = repout*repin.T
        rep_bias = repout
        self.weight_proj = rep_W.symmetric_projection()
        self.bias_proj = rep_bias.symmetric_projection()
        logging.info(f"Linear W components:{rep_W.size()} shape:{rep_W.shape} rep:{rep_W}")
    def __call__(self, x): # (cin) -> (cout)
        logging.debug(f"linear in shape: {x.shape}")
        W = self.weight_proj(self.w.value)
        #logging.error(f"W shape {self.w.value.shape}, repW shape {self.rep_W.shape}")
        #assert False
        b = self.bias_proj(self.b.value)
        out = x@W.T+b
        logging.debug(f"linear out shape:{out.shape}")
        return out

class BiLinear(Module):
    def __init__(self, repin, repout):
        super().__init__()
        rep_W = repout*repin.T
        Wdim, self.weight_proj = bilinear_weights(rep_W,repin)
        self.w = TrainVar(objax.random.normal((Wdim,)))#xavier_normal((Wdim,))) #TODO: revert to xavier
        logging.info(f"BiW components:{rep_W.size()} dim:{Wdim} shape:{rep_W.shape} rep:{rep_W}")

    def __call__(self, x,training=True):
        logging.debug(f"bilinear in shape: {x.shape}")
        W = self.weight_proj(self.w.value,x)
        out= .25*(W@x[...,None])[...,0] #TODO: set back to .05
        #import pdb; pdb.set_trace()
        logging.debug(f"bilinear out shape: {out.shape}")
        return out


def gated(rep):
    return rep+sum([1 for t in rep.ranks if t!=(0,0)])*Scalar

class GatedNonlinearity(Module):
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
    def __call__(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return activations

class EMLPBlock(Module):
    def __init__(self,rep_in,rep_out):
        super().__init__()
        self.linear = LieLinear(rep_in,gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out),gated(rep_out))
        self.bn = TensorBN(gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)
    def __call__(self,x,training=True):
        lin = self.linear(x)
        preact = self.bn(self.bilinear(lin)+lin,training=training)
        return self.nonlinearity(preact)

class EResBlock(Module):
    def __init__(self,rep_in,rep_out):
        super().__init__()
        grep_in = gated(rep_in)
        grep_out = gated(rep_out)

        self.bn1 = TensorBN(grep_in)
        self.nonlinearity1 = GatedNonlinearity(rep_in)
        self.linear1 = LieLinear(rep_in,grep_out)

        self.bn2 = TensorBN(grep_out)
        self.nonlinearity2 = GatedNonlinearity(rep_out)
        self.linear2 = LieLinear(rep_out,grep_out)
        

        self.bilinear1 = BiLinear(grep_in,grep_out)
        #self.bilinear2 = BiLinear(gated(rep_out),gated(rep_out))
        self.shortcut = LieLinear(grep_in,grep_out) if rep_in!=rep_out else Sequential()
    def __call__(self,x,training=True):

        z = self.nonlinearity1(self.bn1(x,training=training))
        z = self.linear1(z)
        z = self.nonlinearity2(self.bn2(x,training=training))
        z = self.linear2(z)
        return (z+self.shortcut(x)+self.bilinear1(x))/3

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

@export
class EMLP(Module):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        repmiddle = uniform_rep(ch,group)
        reps = [self.rep_in]+num_layers*[repmiddle]
        logging.debug(reps)
        self.network = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            LieLinear(repmiddle,self.rep_out)
        )
        #self.network = LieLinear(self.rep_in,self.rep_out)
    def __call__(self,x,training=True):
        y = self.network(x,training=training)
        return y

@export
class EMLP2(Module):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):#@
        super().__init__()
        logging.info("Initing EMLP")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        repmiddle = uniform_rep(ch,group)
        #reps = [self.rep_in]+
        reps = num_layers*[repmiddle]# + [self.rep_out]
        logging.debug(reps)
        self.network = Sequential(
            LieLinear(self.rep_in,gated(repmiddle)),
            *[EResBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            TensorBN(gated(repmiddle)),
            GatedNonlinearity(repmiddle),
            LieLinear(repmiddle,self.rep_out)
        )
    def __call__(self,x,training=True):
        y = self.network(x,training=training)
        return y

def swish(x):
    return jax.nn.sigmoid(x)*x

def MLPBlock(cin,cout): # works better without batchnorm?
    return Sequential(nn.Linear(cin,cout),nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(Module):
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.warning("Initing MLP")
        self.net = Sequential(
            *[MLPBlock(cin,cout) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )
    def __call__(self,x,training=True):
        y = self.net(x,training=training)
        return y

@export
class Standardize(Module):
    def __init__(self,model,ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats=ds_stats
    def __call__(self,x,training):
        if len(self.ds_stats)==2:
            muin,sin = self.ds_stats
            return self.model((x-muin)/sin,training=training)
        else:
            muin,sin,muout,sout = self.ds_stats
            y = sout*self.model((x-muin)/sin,training=training)+muout
            return y
