import jax
import jax.numpy as jnp
import numpy as np
from emlp.reps import Rep
from emlp.reps import bilinear_weights
from emlp.utils import export
import logging
import haiku as hk
from emlp.nn import gated,gate_indices,uniform_rep

def Sequential(*args):
    """ Wrapped to mimic pytorch syntax"""
    return lambda x: hk.Sequential(args)(x)

@export
def Linear(repin,repout):
    rep_W = repout << repin
    logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")
    rep_bias = repout
    Pw = rep_W.equivariant_projector()
    Pb = rep_bias.equivariant_projector()
    return lambda x: hkLinear(Pw,Pb,(repout.size(),repin.size()))(x)

class hkLinear(hk.Module):
    """ Basic equivariant Linear layer from repin to repout."""
    def __init__(self, Pw,Pb,shape,name=None):
        super().__init__(name=name)
        self.Pw = Pw
        self.Pb = Pb
        self.shape=shape

    def __call__(self, x):  # (cin) -> (cout)
        i,j = self.shape
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(i))
        w = hk.get_parameter("w", shape=self.shape, dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[i], dtype=x.dtype, init=w_init)
        W = (self.Pw@w.reshape(-1)).reshape(*self.shape)
        b = self.Pb@b
        return x@W.T+b

@export
def BiLinear(repin,repout):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    Wdim, weight_proj = bilinear_weights(repout,repin)
    return lambda x: hkBiLinear(weight_proj,Wdim)(x)


class hkBiLinear(hk.Module):
    def __init__(self, weight_proj,Wdim,name=None):
        super().__init__(name=name)
        self.weight_proj=weight_proj
        self.Wdim=Wdim

    def __call__(self, x):
        # compatible with non sumreps? need to check
        w_init = hk.initializers.TruncatedNormal(1.)
        w = hk.get_parameter("w", shape=[self.Wdim], dtype=x.dtype, init=w_init)
        W = self.weight_proj(w,x)
        return .1*(W@x[...,None])[...,0]

@export
class GatedNonlinearity(object):  # TODO: add support for mixed tensors and non sumreps
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    def __init__(self,rep,name=None):
        super().__init__()
        self.rep=rep

    def __call__(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = jax.nn.sigmoid(gate_scalars) * values[..., :self.rep.size()]
        return activations


@export
def EMLPBlock(repin,repout):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    linear = Linear(repin,gated(repout))
    bilinear = BiLinear(gated(repout),gated(repout))
    nonlinearity = GatedNonlinearity(repout)
    def block(x):
        lin = linear(x)
        preact =bilinear(lin)+lin
        return nonlinearity(preact)
    return block


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
    logging.info("Initing EMLP (Haiku)")
    rep_in =rep_in(group)
    rep_out = rep_out(group)
    # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
    if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]
    elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
    else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
    # assert all((not rep.G is None) for rep in middle_layers[0].reps)
    reps = [rep_in]+middle_layers
    # logging.info(f"Reps: {reps}")
    network = Sequential(
        *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
        Linear(reps[-1],rep_out)
    )
    return network

@export
def MLP(rep_in,rep_out,group,ch=384,num_layers=3):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    cout = rep_out(group).size()
    mlp = lambda x: Sequential(
        *[Sequential(hk.Linear(ch),jax.nn.swish) for _ in range(num_layers)],
        hk.Linear(cout)
    )(x)
    return mlp
