from emlp.models.mlp import MLP,EMLP#,LinearBNSwish
from emlp.models.datasets import Fr,ParticleInteraction
import jax.numpy as jnp
import jax
from emlp.solver.representation import T,Scalar,Matrix,Vector,Quad,repsize
from emlp.solver.groups import SO,O,Trivial,Lorentz,O13,SO13,SO13p
from emlp.models.mlp import EMLP,LieLinear,Standardize,EMLP2
from emlp.models.model_trainer import RegressorPlus
import itertools
import numpy as np
import torch
from emlp.models.datasets import Inertia,Fr,ParticleInteraction
from emlp.models.particle_dataset import TopTagging,collate_fn
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
import emlp.models
from emlp.models.pointconv_base import ResNet
from emlp.models.model_trainer import ClassifierPlus

def makeTrainer(*,network=ResNet,num_epochs=5,seed=2020,aug=False,
                bs=30,lr=1e-3,device='cuda',split={'train':-1,'val':10000},
                net_config={'k':512,'num_layers':4},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.2}},save=False):
    # Prep the datasets splits, model, and dataloaders
    datasets = {split:TopTagging(split=split) for split in ['train','val']}
    model = network(4,2,**net_config)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False,collate_fn=collate_fn,drop_last=True)) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],0,None,10) #for logging subsample dataset by 5x
    #equivariance_test(model,dataloaders['train'],net_config['group'])
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr*cosLr(num_epochs)(e)
    return ClassifierPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    Trial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(emlp.solver.groups,emlp.models.datasets,emlp.models.mlp)))