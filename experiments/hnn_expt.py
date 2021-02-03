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

from emlp_jax.mlp import MLP,EMLP,MLPH,EMLPH#,LinearBNSwish
from emlp_jax.datasets import O5Synthetic,ParticleInteraction
import jax.numpy as jnp
import jax
from emlp_jax.equivariant_subspaces import T,Scalar,Matrix,Vector
from emlp_jax.groups import SO2eR3,O2eR3,DkeR3,Trivial
from emlp_jax.mlp import EMLP,LieLinear,Standardize
from emlp_jax.model_trainer import RegressorPlus
from emlp_jax.hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum,hnn_trial
import itertools
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed,Named
from slax.utils import LoaderTo
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
import torch.nn as nn
import logging
import emlp_jax
import objax
from emlp_jax.mlp import MLPBlock,Sequential,swish
import objax.nn as nn
import objax.functional as F
from objax.module import Module
import experiments



def makeTrainer(*,dataset=DoubleSpringPendulum,network=MLPH,num_epochs=1000,ndata=5000,seed=2021,aug=False,
                bs=500,lr=3e-3,device='cuda',split={'train':500,'val':.1,'test':.1},
                net_config={'num_layers':3,'ch':128,'group':O2eR3()},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75}},#'early_stop_metric':'val_MSE'},
                save=False,):
    levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                        'warn': logging.WARNING,'warning': logging.WARNING,
                        'info': logging.INFO,'debug': logging.DEBUG}
    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        base_ds = dataset(n_systems=ndata,chunk_len=5)
        datasets = split_dataset(base_ds,splits=split)
    if net_config['group'] is None: net_config['group']=base_ds.symmetry
    model = network(base_ds.rep_in,Scalar,**net_config)
    #if aug: model = datasets['train'].default_aug(model)
    #model = Standardize(model,datasets['train'].stats)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    #equivariance_test(model,dataloaders['train'],net_config['group'])
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr#*cosLr(num_epochs)(e)#*min(1,e/(num_epochs/10))
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)



if __name__=="__main__":
    Trial = hnn_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    name = "hnn_expt"#config_spec.pop('study_name')
    
    #config_spec = argupdated_config(config_spec,namespace=(emlp_jax.groups,emlp_jax.datasets,emlp_jax.mlp))
    #name = f"{name}_{config_spec['dataset']}"
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    config_spec['network'] = EMLPH
    config_spec['net_config']['group'] = [O2eR3(),SO2eR3(),DkeR3(6),DkeR3(2)]
    thestudy.run(num_trials=-5,new_config_spec=config_spec,ordered=True)
    config_spec['network'] = MLPH
    config_spec['net_config']['group'] = None
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)
    print(thestudy.results_df())
