import jax
import functools
from oil.utils.utils import imap
import numpy as np
def minibatch_to(mb):
    try:
        if isinstance(mb,np.ndarray):
            return jax.device_put(mb)
        return jax.device_put(mb.numpy())
    except AttributeError:
        if isinstance(mb,dict):
            return type(mb)(((k,minibatch_to(v)) for k,v in mb.items()))
        else:
            return type(mb)(minibatch_to(elem) for elem in mb)

def LoaderTo(loader):
    return imap(functools.partial(minibatch_to),loader)