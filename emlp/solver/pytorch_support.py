import torch
from torch.autograd import Function
import jax
from jax.tree_util import tree_flatten, tree_unflatten
import types
import copy
import jax.numpy as jnp
import numpy as np
import types
from functools import partial

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

def torchify_fn(function):
    """ A method to enable interopability between jax and pytorch autograd.
        Calling torchify on a given function that has pytrees of jax.ndarray
        objects as inputs and outputs will return a function that first converts
        the inputs to jax, runs through the jax function, and converts the output
        back to torch but preserving the gradients of the operation to be called
        with pytorch autograd. """
    class torched_fn(Function):
        @staticmethod
        def forward(ctx,*args):
            if any(ctx.needs_input_grad):
                y,ctx.vjp_fn = jax.vjp(function,*to_jax(args))
                return to_pytorch(y)
            return to_pytorch(function(*to_jax(args)))
        @staticmethod
        def backward(ctx,*grad_outputs):
            return to_pytorch(ctx.vjp_fn(*to_jax(grad_outputs)))
    return torched_fn.apply #TORCHED #Roasted
