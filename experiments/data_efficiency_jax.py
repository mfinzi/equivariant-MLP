from emlp_jax.mlp import MLP,EMLP#,LinearBNSwish
from emlp_jax.datasets import O5Synthetic,ParticleInteraction,Inertia
import jax.numpy as jnp
import jax
from emlp_jax.equivariant_subspaces import T,Scalar,Matrix,Vector
from emlp_jax.groups import SO,O,Trivial,Lorentz,O13,SO13,SO13p
from emlp_jax.mlp import EMLP,LieLinear,Standardize
from emlp_jax.model_trainer import RegressorPlus
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from slax.utils import LoaderTo
from oil.tuning.study import train_trial,Study
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
import logging
import emlp_jax
import objax
import copy

def makeTrainer(*,dataset=O5Synthetic,network=EMLP,num_epochs=3,ndata=30000+6000,seed=2021,aug=False,
                bs=500,lr=3e-3,split={'train':100,'val':1000,'test':5000},
                net_config={'num_layers':3,'ch':384,'group':None},
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':1.},
                'early_stop_metric':'val_MSE'},save=False,
                study_name='data_efficiency_nobn'):

    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_dataset = dataset(ndata)
        datasets = split_dataset(base_dataset,splits=split)
    if net_config['group'] is None: net_config['group']=base_dataset.symmetry
    model = network(base_dataset.rep_in,base_dataset.rep_out,**net_config)
    if aug: model = base_dataset.default_aug(model)
    model = Standardize(model,datasets['train'].stats)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr*min(1,e/(num_epochs/10))
    return RegressorPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__=="__main__":
    Trial = train_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    name = config_spec.pop('study_name')
    config_spec.update({
        'dataset':O5Synthetic,#[Inertia,Fr],
        'network':MLP,'aug':[False,True],
        'num_epochs':(lambda cfg: min(int(30*30000/cfg['split']['train']),1000)),
        'split':{'train':[30,100,300,1000,3000,10000,30000],'test':5000,'val':1000},
    })
    config_spec = argupdated_config(config_spec,namespace=(emlp_jax.groups,emlp_jax.datasets,emlp_jax.mlp))
    name = f"{name}_{config_spec['dataset']}"
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    
    print(config_spec)
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)
    # Now the EMLP runs
    config_spec['network']=EMLP
    config_spec['aug'] = False
    groups = {O5Synthetic:[SO(5),O(5)],Inertia:[SO(3),O(3)],ParticleInteraction:[SO13p(),SO13(),O13()]}
    config_spec['net_config']['group'] = groups[config_spec['dataset']]
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)
    print(thestudy.results_df())
