import torch
import torch.nn as nn
from oil.utils.utils import export
from .trainer import Trainer
import jax
import jax.numpy as jnp
import numpy as np

def cross_entropy(logprobs, targets):
    ll = jnp.take_along_axis(logprobs, jnp.expand_dims(targets, axis=1), axis=1)
    ce = -jnp.mean(ll)
    return ce

@export
class Classifier(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    
    def loss(self,minibatch):
        """ Standard cross-entropy loss """ #TODO: support class weights
        x,y = minibatch
        logits = self.model(x,training=True)
        logp = jax.nn.log_softmax(logits)
        return cross_entropy(logp,y)

    def metrics(self,loader):
        acc = lambda mb: np.asarray(jax.device_get(jnp.mean(jnp.argmax(self.model.predict(mb[0]),axis=-1)==mb[1])))
        return {'Acc':self.evalAverageMetrics(loader,acc)}

@export
class Regressor(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self,minibatch):
        """ Standard cross-entropy loss """
        x,y = minibatch
        mse = jnp.mean((self.model(x,training=True)-y)**2)
        return mse

    def metrics(self,loader):
        mse = lambda mb: np.asarray(jax.device_get(jnp.mean((self.model.predict(mb[0])-mb[1])**2)))
        return {'MSE':self.evalAverageMetrics(loader,mse)}