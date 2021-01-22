from emlp_jax.mlp import MLP,EMLP#,LinearBNSwish
from emlp_jax.datasets import Fr,ParticleInteraction
import jax.numpy as jnp
import jax
from emlp_jax.equivariant_subspaces import T,Scalar,Matrix,Vector,Quad,repsize
from emlp_jax.groups import SO,O,Trivial,Lorentz,O13,SO13,SO13p
from emlp_jax.mlp import EMLP,LieLinear,Standardize,EMLP2
from emlp_jax.model_trainer import RegressorPlus
import itertools
import numpy as np
import torch
from emlp_jax.datasets import Inertia,Fr,ParticleInteraction
from emlp_jax.particle_dataset import TopTagging,collate_fn
import objax
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from slax.utils import LoaderTo
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from slax.model_trainers import Classifier,Trainer
from functools import partial
import torch.nn as nn
import logging
import emlp_jax
from emlp_jax.pointconv_base import ResNet

class ClassifierPlus(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        
        fastloss = objax.Jit(self.loss,model.vars())
        self.gradvals = objax.Jit(objax.GradValues(fastloss,model.vars()),model.vars())
        self.model.predict = objax.Jit(objax.ForceArgs(model.__call__,training=False),model.vars())
        #self.model.predict = lambda x: self.model(x,training=False)
    # def loss(self,minibatch):
    #     """ Standard cross-entropy loss """
    #     x,y = minibatch
    #     mse = jnp.mean((self.model(x,training=True)-y)**2)#jnp.mean(jnp.abs(self.model(x,training=True)-y))
    #     return mse

    # def metrics(self,loader):
    #     mse = lambda mb: np.asarray(jax.device_get(jnp.mean((self.model.predict(mb[0])-mb[1])**2)))
    #     return {'MSE':self.evalAverageMetrics(loader,mse)}
    # def logStuff(self, step, minibatch=None):
    #     metrics = {}
    #     metrics['test_equivar_err'] = self.evalAverageMetrics(islice(self.dataloaders['test'],0,None,5),
    #                             partial(equivariance_err,self.model)) # subsample by 5x so it doesn't take too long
    #     self.logger.add_scalars('metrics', metrics, step)
    #     super().logStuff(step,minibatch)

def makeTrainer(*,network=ResNet,num_epochs=5,seed=2020,aug=False,
                bs=30,lr=1e-3,device='cuda',split={'train':-1,'val':10000},
                net_config={'k':512,'num_layers':4},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.25}},save=False):
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        datasets = split_dataset(TopTagging(),splits=split)
    model = network(4,2,**net_config)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False,collate_fn=collate_fn,drop_last=True)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    #equivariance_test(model,dataloaders['train'],net_config['group'])
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr*cosLr(num_epochs)(e)
    return ClassifierPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    Trial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(emlp_jax.groups,emlp_jax.datasets,emlp_jax.mlp)))