from emlp.models.mlp import MLP,EMLP#,LinearBNSwish
from emlp.models.datasets import O5Synthetic,ParticleInteraction
import jax.numpy as jnp
import jax
from emlp.solver.representation import T,Scalar
from emlp.solver.groups import SO,O,Trivial,Lorentz,O13,SO13,SO13p
from emlp.models.mlp import EMLP,LieLinear,Standardize
from emlp.models.model_trainer import RegressorPlus
import itertools
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from emlp.slax.utils import LoaderTo
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
import torch.nn as nn
import logging
import emlp.models
import objax
#repmiddle = 100*T(0)+30*T(1)+10*T(2)+3*T(3)#+1*T(4)


def makeTrainer(*,dataset=ParticleInteraction,network=EMLP,num_epochs=300,ndata=1000+2000,seed=2021,aug=False,
                bs=500,lr=3e-3,device='cuda',split={'train':-1,'val':1000,'test':1000},
                net_config={'num_layers':3,'ch':384,'group':SO13p()},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75},'early_stop_metric':'val_MSE'},
                save=False,):
    levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                        'warn': logging.WARNING,'warning': logging.WARNING,
                        'info': logging.INFO,'debug': logging.DEBUG}
    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        datasets = split_dataset(dataset(ndata),splits=split)
    model = network(datasets['train'].rep_in,datasets['train'].rep_out,**net_config)
    if aug: model = datasets['train'].default_aug(model)
    model = Standardize(model,datasets['train'].stats)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    #equivariance_test(model,dataloaders['train'],net_config['group'])
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    return RegressorPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    cfg,outcome = Trial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(emlp.solver.groups,emlp.models.datasets,emlp.models.mlp)))
    print(outcome)