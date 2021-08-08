import jax
import jax.numpy as jnp
import numpy as np
from emlp.reps import T,Rep,Scalar
from emlp.reps import bilinear_weights
#from emlp.reps import LinearOperator # why does this not work?
from emlp.reps.linear_operator_base import LinearOperator
from emlp.reps.product_sum_reps import SumRep
from emlp.groups import Group
from emlp.utils import Named,export
from flax import linen as nn
import logging
from emlp.nn import gated,gate_indices,uniform_rep
from typing import Union,Iterable,Optional
# def Sequential(*args):
#     """ Wrapped to mimic pytorch syntax"""
#     return nn.Sequential(args)




@export
def Linear(repin,repout):
    """ Basic equivariant Linear layer from repin to repout."""
    cin =repin.size()
    cout = repout.size()
    rep_W = repin>>repout
    Pw = rep_W.equivariant_projector()
    Pb = repout.equivariant_projector()
    logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    return _Linear(Pw,Pb,cout)

class _Linear(nn.Module):
    Pw:LinearOperator
    Pb:LinearOperator
    cout:int
    @nn.compact
    def __call__(self,x):
        w = self.param('w',nn.initializers.lecun_normal(),(self.cout,x.shape[-1]))
        b = self.param('b',nn.initializers.zeros,(self.cout,))
        W = (self.Pw@w.reshape(-1)).reshape(*w.shape)
        B = self.Pb@b
        return x@W.T+B

@export
def BiLinear(repin,repout):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    Wdim, weight_proj = bilinear_weights(repout,repin)
        #self.w = TrainVar(objax.random.normal((Wdim,)))#xavier_normal((Wdim,))) #TODO: revert to xavier
    logging.info(f"BiW components: dim:{Wdim}")
    return _BiLinear(Wdim,weight_proj)

class _BiLinear(nn.Module):
    Wdim:int
    weight_proj:callable

    @nn.compact
    def __call__(self, x):
        w = self.param('w',nn.initializers.normal(),(self.Wdim,)) #TODO: change to standard normal
        W = self.weight_proj(w,x)
        out= .1*(W@x[...,None])[...,0]
        return out


@export
class GatedNonlinearity(nn.Module): #TODO: add support for mixed tensors and non sumreps
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    rep:Rep
    def __call__(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return activations

@export
def EMLPBlock(rep_in,rep_out):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    linear = Linear(rep_in,gated(rep_out))
    bilinear = BiLinear(gated(rep_out),gated(rep_out))
    nonlinearity = GatedNonlinearity(rep_out)
    return _EMLPBlock(linear,bilinear,nonlinearity)

class _EMLPBlock(nn.Module):
    linear:nn.Module
    bilinear:nn.Module
    nonlinearity:nn.Module

    def __call__(self,x):
        lin = self.linear(x)
        preact =self.bilinear(lin)+lin
        return self.nonlinearity(preact)

@export
def EMLP(rep_in,rep_out,group,ch=384,num_layers=3):
    """ Equivariant MultiLayer Perceptron. 
        If the input ch argument is an int, uses the hands off uniform_rep heuristic.
        If the ch argument is a representation, uses this representation for the hidden layers.
        Individual layer representations can be set explicitly by using a list of ints or a list of
        representations, rather than use the same for each hidden layer.

        Args:
            rep_in (Rep): input representation
            rep_out (Rep): output representation
            group (Group): symmetry group
            ch (int or list[int] or Rep or list[Rep]): number of channels in the hidden layers
            num_layers (int): number of hidden layers

        Returns:
            Module: the EMLP objax module."""
    logging.info("Initing EMLP (flax)")
    rep_in = rep_in(group)
    rep_out = rep_out(group)
    if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]
    elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
    else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
    reps = [rep_in]+middle_layers
    logging.info(f"Reps: {reps}")
    return Sequential(*[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],Linear(reps[-1],rep_out))

def swish(x):
    return jax.nn.sigmoid(x)*x

class _Sequential(nn.Module):
    modules:Iterable[callable]
    def __call__(self,x):
        for module in self.modules:
            x = module(x)
        return x

def Sequential(*layers):
    return _Sequential(layers)

def MLPBlock(cout):
    return Sequential(nn.Dense(cout),swish)  # ,nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(nn.Module,metaclass=Named):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    rep_in:Rep
    rep_out:Rep
    group:Group
    ch: Optional[InterruptedError]=384
    num_layers:Optional[int]=3
    def setup(self):
        logging.info("Initing MLP (flax)")
        cout = self.rep_out(self.group).size()
        self.modules = [MLPBlock(self.ch) for _ in range(self.num_layers)]+[nn.Dense(cout)]
    def __call__(self,x):
        for module in self.modules:
            x = module(x)
        return x
