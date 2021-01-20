from emlp_jax.mlp import MLP,EMLP#,LinearBNSwish
from emlp_jax.datasets import Fr,ParticleInteraction
import jax.numpy as jnp
import jax
import optax
from emlp_jax.equivariant_subspaces import T,Scalar,Matrix,Vector,Quad,repsize
from emlp_jax.groups import SO,O,Trivial,Lorentz,O13,SO13,SO13p
from emlp_jax.mlp import EMLP,LieLinear,Standardize,EMLP2
import itertools
import numpy as np
import torch
from emlp_jax.datasets import Inertia,Fr,ParticleInteraction
import objax
import torch
from torch.utils.data import DataLoader
from slax.utils.utils import LoaderTo,cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from slax.tuning.study import train_trial
from slax.datasetup.datasets import split_dataset
from emlp_jax.model_trainer import RegressorPlus
from slax.tuning.args import argupdated_config
from functools import partial
import torch.nn as nn
import logging
import emlp_jax
#repmiddle = 100*T(0)+30*T(1)+10*T(2)+3*T(3)#+1*T(4)

def makeTrainer(*,dataset=Fr,network=EMLP2,num_epochs=500,ndata=1000+1000,seed=2020,aug=False,
                bs=500,lr=3e-3,device='cuda',split={'train':-1,'test':1000},
                net_config={'num_layers':3,'group':SO(3)},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.25}},save=False):
    #logging.basicConfig(level='logging.'+log_level)    
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        datasets = split_dataset(dataset(ndata),splits=split)
    model = Standardize(network(datasets['train'].rep_in,datasets['train'].rep_out,**net_config),datasets['train'].stats)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr*cosLr(num_epochs)(e)
    return RegressorPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    Trial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(emlp_jax.groups,emlp_jax.datasets,emlp_jax.mlp)))