import torch
from torch.autograd import Function
import jax
from jax import jit
from jax.tree_util import tree_flatten, tree_unflatten
import types
import copy
import jax.numpy as jnp
import numpy as np
import types
from functools import partial
from emlp.reps import T,Rep,Scalar
from emlp.reps import bilinear_weights
from emlp.utils import Named,export
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from emlp.nn import gated,gate_indices,uniform_rep

def torch2jax(arr):
    if isinstance(arr,torch.Tensor):
        return jnp.asarray(arr.cpu().data.numpy())
    else:
        return arr

def jax2torch(arr):
    if isinstance(arr,(jnp.ndarray,np.ndarray)):
        if jax.devices()[0].platform=='gpu':
            device = torch.device('cuda') 
        else: device = torch.device('cpu')
        return torch.from_numpy(np.array(arr)).to(device)
    else:
        return arr

def to_jax(pytree):
    flat_values, tree_type = tree_flatten(pytree)
    transformed_flat = [torch2jax(v) for v in flat_values]
    return tree_unflatten(tree_type, transformed_flat)


def to_pytorch(pytree):
    flat_values, tree_type = tree_flatten(pytree)
    transformed_flat = [jax2torch(v) for v in flat_values]
    return tree_unflatten(tree_type, transformed_flat)

@export
def torchify_fn(function):
    """ A method to enable interopability between jax and pytorch autograd.
        Calling torchify on a given function that has pytrees of jax.ndarray
        objects as inputs and outputs will return a function that first converts
        the inputs to jax, runs through the jax function, and converts the output
        back to torch but preserving the gradients of the operation to be called
        with pytorch autograd. """
    vjp = jit(lambda *args: jax.vjp(function,*args))
    class torched_fn(Function):
        @staticmethod
        def forward(ctx,*args):
            if any(ctx.needs_input_grad):
                y,ctx.vjp_fn = vjp(*to_jax(args))#jax.vjp(function,*to_jax(args))
                return to_pytorch(y)
            return to_pytorch(function(*to_jax(args)))
        @staticmethod
        def backward(ctx,*grad_outputs):
            return to_pytorch(ctx.vjp_fn(*to_jax(grad_outputs)))
    return torched_fn.apply #TORCHED #Roasted


@export
class Linear(nn.Linear):
    """ Basic equivariant Linear layer from repin to repout."""
    def __init__(self, repin, repout):
        nin,nout = repin.size(),repout.size()
        super().__init__(nin,nout)
        rep_W = repout*repin.T
        rep_bias = repout
        Pw = rep_W.equivariant_projector()
        Pb = rep_bias.equivariant_projector()
        self.proj_b = torchify_fn(jit(lambda b: Pb@b))
        self.proj_w = torchify_fn(jit(lambda w:(Pw@w.reshape(-1)).reshape(nout,nin)))
        logging.info(f"Linear W components:{rep_W.size()} rep:{rep_W}")

    def forward(self, x): # (cin) -> (cout)
        return F.linear(x,self.proj_w(self.weight),self.proj_b(self.bias))

@export
class BiLinear(nn.Module):
    """ Cheap bilinear layer (adds parameters for each part of the input which can be
        interpreted as a linear map from a part of the input to the output representation)."""
    def __init__(self, repin, repout):
        super().__init__()
        Wdim, weight_proj = bilinear_weights(repout,repin)
        self.weight_proj = torchify_fn(jit(weight_proj))
        self.bi_params = nn.Parameter(torch.randn(Wdim))
        logging.info(f"BiW components: dim:{Wdim}")

    def forward(self, x,training=True):
        # compatible with non sumreps? need to check
        W = self.weight_proj(self.bi_params,x)
        out= .1*(W@x[...,None])[...,0]
        return out

@export
class GatedNonlinearity(nn.Module): #TODO: add support for mixed tensors and non sumreps
    """ Gated nonlinearity. Requires input to have the additional gate scalars
        for every non regular and non scalar rep. Applies swish to regular and
        scalar reps. (Right now assumes rep is a SumRep)"""
    def __init__(self,rep):
        super().__init__()
        self.rep=rep
    def forward(self,values):
        gate_scalars = values[..., gate_indices(self.rep)]
        activations = gate_scalars.sigmoid() * values[..., :self.rep.size()]
        return activations

@export
class EMLPBlock(nn.Module):
    """ Basic building block of EMLP consisting of G-Linear, biLinear,
        and gated nonlinearity. """
    def __init__(self,rep_in,rep_out):
        super().__init__()
        self.linear = Linear(rep_in,gated(rep_out))
        self.bilinear = BiLinear(gated(rep_out),gated(rep_out))
        self.nonlinearity = GatedNonlinearity(rep_out)

    def forward(self,x):
        lin = self.linear(x)
        preact =self.bilinear(lin)+lin
        return self.nonlinearity(preact)

@export
class EMLP(nn.Module):
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
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        logging.info("Initing EMLP (PyTorch)")
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        
        self.G=group
        # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
        if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]#[uniform_rep(ch,group) for _ in range(num_layers)]
        elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
        else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
        #assert all((not rep.G is None) for rep in middle_layers[0].reps)
        reps = [self.rep_in]+middle_layers
        #logging.info(f"Reps: {reps}")
        self.network = nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        )
    def forward(self,x):
        return self.network(x)

class Swish(nn.Module):
    def forward(self,x):
        return x.sigmoid()*x

def MLPBlock(cin,cout):
    return nn.Sequential(nn.Linear(cin,cout),Swish())#,nn.BatchNorm0D(cout,momentum=.9),swish)#,

@export
class MLP(nn.Module):
    """ Standard baseline MLP. Representations and group are used for shapes only. """
    def __init__(self,rep_in,rep_out,group,ch=384,num_layers=3):
        super().__init__()
        self.rep_in =rep_in(group)
        self.rep_out = rep_out(group)
        self.G = group
        chs = [self.rep_in.size()] + num_layers*[ch]
        cout = self.rep_out.size()
        logging.info("Initing MLP")
        self.net = nn.Sequential(
            *[MLPBlock(cin,cout) for cin,cout in zip(chs,chs[1:])],
            nn.Linear(chs[-1],cout)
        )

    def forward(self,x):
        y = self.net(x)
        return y

@export
class Standardize(nn.Module):
    """ A convenience module to wrap a given module, normalize its input
        by some dataset x mean and std stats, and unnormalize its output by
        the dataset y mean and std stats. 

        Args:
            model (Module): model to wrap
            ds_stats ((μx,σx,μy,σy) or (μx,σx)): tuple of the normalization stats
        
        Returns:
            Module: Wrapped model with input normalization (and output unnormalization)"""
    def __init__(self,model,ds_stats):
        super().__init__()
        self.model = model
        self.ds_stats=ds_stats

    def forward(self,x,training):
        if len(self.ds_stats)==2:
            muin,sin = self.ds_stats
            return self.model((x-muin)/sin,training=training)
        else:
            muin,sin,muout,sout = self.ds_stats
            y = sout*self.model((x-muin)/sin,training=training)+muout
            return y
