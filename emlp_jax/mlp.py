import jax
import jax.numpy as jnp
import objax.nn as nn
import objax.functional as F
import numpy as np
from emlp_jax.equivariant_subspaces import T,Scalar,Vector,Matrix,Quad,size
from emlp_jax.equivariant_subspaces import capped_tensor_ids,rep_permutation,bilinear_weights
from emlp_jax.groups import LearnedGroup
from emlp_jax.batchnorm import TensorMaskBN,gate_indices
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


# def LieLinearBNSwish(repin,repout):
#     return Sequential([LieLinear(repin,gated(repout)),
#                          TensorMaskBN(gated(repout)),
#                          GatedNonlinearity(repout)])

# def TensorLinearBNSwish(repin,repout): #TODO: investigate BiLinear after LieLinear instead of parallel
#     #return Sequential(TensorLinear(repin,gated(repout)),TensorMaskBN(gated(repout)),GatedNonlinearity(repout))
#     return Sequential([Sum([BiLinear(repin,gated(repout)),LieLinear(repin,gated(repout))]),
#                          TensorMaskBN(gated(repout)),
#                          GatedNonlinearity(repout)])

class EMLPBlock(Module):
    def __init__(self,rep_in,rep_out):
        super().__init__()
        self.linear = LieLinear(rep_in,gated(rep_out))
        self.bilinear = BiLinear(rep_in,gated(rep_out))
        self.bn = TensorMaskBN(gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)
    def __call__(self,x,training=True):
        preact = self.bn(self.linear(x)+self.bilinear(x),training=training)#training=training)
        return self.nonlinearity(preact)

# class EMLPBlock(Module):
#     def __init__(self,rep_in,rep_out):
#         super().__init__()
#         self.linear = LieLinear(rep_in,gated(rep_out))
#         self.bilinear = BiLinear(gated(rep_out),gated(rep_out))
#         self.bn = TensorMaskBN(gated(rep_out))
#         self.nonlinearity = GatedNonlinearity(rep_out)
#     def __call__(self,x,training=True):
#         lin = self.linear(x)
#         preact = self.bn(self.bilinear(lin)+lin,training=training)
#         #preact = self.bn(self.linear(x)+self.bilinear(x),training=training)
#         return self.nonlinearity(preact)
# Linear variant for testing
# def EMLPBlock(rep_in,rep_out):
#     return LieLinear(rep_in,rep_out)
# class EMLPBlock(Module):
#     def __init__(self,rep_in,rep_out):
#         super().__init__()
#         self.linear = LieLinear(rep_in,gated(rep_out))
#         self.bilinear = BiLinear(rep_in,gated(rep_out))
#         self.bn = TensorMaskBN(gated(rep_out))
#         self.nonlinearity = GatedNonlinearity(rep_out)
#     def __call__(self,x,training=True):
#         preact = self.bn(self.linear(x)+self.bilinear(x),training=training)#training=training)
#         return self.nonlinearity(preact)

class EResBlock(Module):
    def __init__(self,rep_in,rep_out):
        super().__init__()
        grep_in = gated(rep_in)
        grep_out = gated(rep_out)

        self.bn1 = TensorMaskBN(grep_in)
        self.nonlinearity1 = GatedNonlinearity(rep_in)
        self.linear1 = LieLinear(rep_in,grep_out)

        self.bn2 = TensorMaskBN(grep_out)
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
            TensorMaskBN(gated(repmiddle)),
            GatedNonlinearity(repmiddle),
            LieLinear(repmiddle,self.rep_out)
        )
    def __call__(self,x,training=True):
        y = self.network(x,training=training)
        return y
# class EMLPLearned(Module):
#     def __init__(self,rep_in,rep_out,d,ch=384,num_layers=3):#@
#         super().__init__()
#         self.group = LearnedGroup(d,ndiscrete=0)
#         logging.info("Initing EMLP")
#         repmiddle = uniform_rep(ch,self.group)
#         reps = [rep_in(self.group)]+num_layers*[repmiddle]
#         logging.debug(reps)
#         self.network = Sequential(
#             *[EMLPBlockLearned(rin,rout) for rin,rout in zip(reps,reps[1:])],
#             LieLinearLearned(repmiddle,rep_out(self.group))
#         )
#     def __call__(self,x,training=True):
#         # if training: 
#         #     logging.info(f"Learned Lie Algebra: {str(self.group.lie_algebra)}")
#         #     logging.info(f"Learned generators: {str(self.group.discrete_generators)}")
#         y = self.network(x,training=training)
#         return y.squeeze(-1)# if y.shape[-1]==1 else y

def swish(x):
    return jax.nn.sigmoid(x)*x

def MLPBlock(cin,cout): # works better without batchnorm?
    return Sequential(nn.Linear(cin,cout),nn.BatchNorm0D(cout,momentum=.9),swish)#,

# class MLPBlock(Module):
#     def __init__(self,cin,cout):
#         super().__init__()
#         self.linear = nn.Linear(cin,cout)
#         self.bn = nn.BatchNorm0D(cout,momentum=.9)
#     def __call__(self,x,training=True):
#         return swish(self.bn(self.linear(x),training=training))

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
    def __call__(self,x,training,Training=True):
        muin,sin,muout,sout = self.ds_stats
        y = sout*self.model((x-muin)/sin,training=training)+muout
        return y