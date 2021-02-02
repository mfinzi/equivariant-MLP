from emlp_jax.mlp import MLP,EMLP,MLPH,EMLPH#,LinearBNSwish
from emlp_jax.datasets import O5Synthetic,ParticleInteraction
import jax.numpy as jnp
import jax
from emlp_jax.equivariant_subspaces import T,Scalar,Matrix,Vector
from emlp_jax.groups import R3embeddedSO2,R3embeddedO2,Trivial
from emlp_jax.mlp import EMLP,LieLinear,Standardize
from emlp_jax.model_trainer import RegressorPlus
from emlp_jax.hamiltonian_dynamics import IntegratedDynamicsTrainer,DoubleSpringPendulum
import itertools
import numpy as np
import torch
import torch
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed,Named
from slax.utils import LoaderTo
from oil.tuning.study import train_trial
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
                net_config={'num_layers':3,'ch':384,'group':Trivial(3)},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75},},#'early_stop_metric':'val_MSE'},
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

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    cfg,outcome = Trial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(emlp_jax.groups,emlp_jax.datasets,emlp_jax.mlp)))
    print(outcome)




# #Trial = train_trial(makeTrainer)
# if __name__ == "__main__":
#     with FixedNumpySeed(0):
#         #rollouts = trainer.test_rollouts(angular_to_euclidean= not issubclass(cfg['network'],(CH,CL)))
#         #print(f"rollout error GeoMean {rollouts[0][:,1:].log().mean().exp():.3E}")
#         #fname = f"rollout_errs_{cfg['network']}_{cfg['body']}.np"
#         #with open(fname,'wb') as f:
#         #    pickle.dump(rollouts,f)
#         #defaults["trainer_config"]["early_stop_metric"] = "val_MSE"
#         #print(Trial()))