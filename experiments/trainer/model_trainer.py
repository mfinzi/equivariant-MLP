import torch
import torch.nn as nn
from oil.utils.utils import export
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import objax
from .classifier import Regressor,Classifier
from functools import partial
from itertools import islice

def rel_err(a,b):
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))#

def scale_adjusted_rel_err(a,b,g):
    return  jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean())+jnp.abs(g-jnp.eye(g.shape[-1])).mean())

def equivariance_err(model,mb,group=None):
    x,y = mb
    group = model.model.G if group is None else group
    gs = group.samples(x.shape[0])
    rho_gin = vmap(model.model.rep_in.rho_dense)(gs)
    rho_gout = vmap(model.model.rep_out.rho_dense)(gs)
    y1 = model.predict((rho_gin@x[...,None])[...,0])
    y2 = (rho_gout@model.predict(x)[...,None])[...,0]
    return np.asarray(scale_adjusted_rel_err(y1,y2,gs))

@export
class RegressorPlus(Regressor):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        fastloss = objax.Jit(self.loss,model.vars())
        self.gradvals = objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
        #self.model.predict = lambda x: self.model(x,training=False)
    def loss(self,minibatch):
        """ Standard cross-entropy loss """
        x,y = minibatch
        mse = jnp.mean((self.model(x,training=True)-y)**2)#jnp.mean(jnp.abs(self.model(x,training=True)-y))
        return mse

    def metrics(self,loader):
        mse = lambda mb: np.asarray(jax.device_get(jnp.mean((self.model.predict(mb[0])-mb[1])**2)))
        return {'MSE':self.evalAverageMetrics(loader,mse)}
    def logStuff(self, step, minibatch=None):
        metrics = {}
        metrics['test_equivar_err'] = self.evalAverageMetrics(islice(self.dataloaders['test'],0,None,5),
                                partial(equivariance_err,self.model)) # subsample by 5x so it doesn't take too long
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step,minibatch)

@export
class ClassifierPlus(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        
        fastloss = objax.Jit(self.loss,model.vars())
        self.gradvals = objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
        #self.model.predict = lambda x: self.model(x,training=False)
    

    def logStuff(self, step, minibatch=None):
        metrics = {}
        metrics['test_equivar_err'] = self.evalAverageMetrics(islice(self.dataloaders['test'],0,None,5),
                                partial(equivariance_err,self.model)) # subsample by 5x so it doesn't take too long
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step,minibatch)