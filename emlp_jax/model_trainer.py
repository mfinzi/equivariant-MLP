import torch
import torch.nn as nn
from slax.utils.utils import export
import jax
import jax.numpy as jnp
import numpy as np
import objax
from slax.model_trainers.classifier import Regressor
from functools import partial
from itertools import islice

def rel_err(a,b):
    return jnp.sqrt(((a-b)**2).mean())/(jnp.sqrt((a**2).mean())+jnp.sqrt((b**2).mean()))

def equivariance_err(model,mb,group=None):
    x,y = mb
    group = model.model.rep_in.G if group is None else group
    rho_gin = jnp.stack([model.model.rep_in.rho(g) for g in group.samples(x.shape[0])])
    rho_gout = jnp.stack([model.model.rep_out.rho(g) for g in group.samples(x.shape[0])])
    y1 = model.predict((rho_gin@x[...,None])[...,0])
    y2 = (rho_gout@model.predict(x)[...,None])[...,0]
    return rel_err(y1,y2)

@export
class RegressorPlus(Regressor):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
        #forward = objax.Jit(objax.ForceArgs(model.__call__,training=True),model.vars())
        #self.model.__call__ = lambda x,training=True: forward(x) if training else predict(x)
        #self.model =model= objax.Jit(model,static_argnums=(1,))
        fastloss = objax.Jit(self.loss,model.vars())
        self.gradvals = objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        
        self.model.predict = predict
        #self._model = model
        #self.model= objax.Jit(objax.ForceArgs(model,training=True)) #TODO: figure out static nums
        #self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
        #self._model = model
        #self.model = objax.ForceArgs(model,training=True)
        #self.model.predict = objax.ForceArgs(model.__call__,training=False)
        #self.model = objax.Jit(lambda x, training: model(x,training=training),model.vars(),static_argnums=(1,))
        #self.model = objax.Jit(model,static_argnums=(1,))
    # def loss(self,minibatch):
    #     """ Standard cross-entropy loss """
    #     x,y = minibatch
    #     #l2 = 0.5 * sum((v.value ** 2).sum() for k, v in self._model.vars().items() if k.endswith('.w'))
    #     #extra_loss = sum(exl.value for k, exl in self._model.vars().items() if k.endswith('.extra_loss'))
    #     mse = jnp.mean((self.model(x,training=True)-y)**2)
    #     return mse

    # def metrics(self,loader):
    #     mse = lambda mb: np.asarray(jax.device_get(jnp.mean((self.model.predict(mb[0])-mb[1])**2)))
    #     return {'MSE':self.evalAverageMetrics(loader,mse)}
    def loss(self,minibatch):
        """ Standard cross-entropy loss """
        x,y = minibatch
        mse = jnp.mean(jnp.abs(self.model(x,training=True)-y))
        return mse

    def metrics(self,loader):
        mse = lambda mb: np.asarray(jax.device_get(jnp.mean((self.model.predict(mb[0])-mb[1])**2)))
        return {'MSE':self.evalAverageMetrics(loader,mse)}
    def logStuff(self, step, minibatch=None):
        metrics = {}
        metrics['test_equivar_err'] = self.evalAverageMetrics(islice(self.dataloaders['test'],0,None,5),
                                        partial(equivariance_err,self.model))
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(step,minibatch)