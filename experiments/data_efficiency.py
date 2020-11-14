from emlp.equivariant_subspaces import T,Scalar,Matrix,Vector,Quad,repsize
from emlp.mlp import MLP, EMLP,LieLinear
import itertools
import numpy as np
import torch
from emlp.datasets import Inertia
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,AdamW
from oil.utils.utils import LoaderTo, cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from oil.tuning.study import train_trial,Study,train_trial
from oil.datasetup.datasets import split_dataset
from oil.model_trainers.classifier import Regressor
from functools import partial
from emlp.equivariant_subspaces import T,Scalar,Matrix,Vector,Quad,repsize
from emlp.groups import SO,O,Trivial
from emlp.mlp import MLP, EMLP,LieLinear
import itertools
import numpy as np
import torch
from emlp.datasets import Inertia,Fr
import copy

repmiddle = 100*T(0)+30*T(1)+10*T(2)+3*T(3)#+1*T(4)
def makeTrainer(*,dataset=Fr,network=EMLP,num_epochs=300,ndata=10000+1000,seed=2020,
                bs=500,lr=3e-3,optim=AdamW,device='cuda',split={'train':100,'test':1000},
                net_config={'rep_middle':repmiddle,'num_layers':4,'group':Trivial(3)},opt_config={'weight_decay':0*3e-6},
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02}},save=False):

    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        datasets = split_dataset(dataset(ndata),splits=split)
    device = torch.device(device)
    model = network(datasets['train'].rep_in,datasets['train'].rep_out,**net_config).to(device)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return Regressor(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__=="__main__":
    Trial = train_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    name = 'EMLP_synthetic_study'#config_spec.pop('study_name')
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    config_spec.update({
        'dataset':[Inertia,Fr],
        'network':MLP,
        'num_epochs':(lambda cfg: int(30*10**4/cfg['split']['train'])),
        'split':{'train':[25,50,100,400,1000,3000,10000],'test':1000},
    })
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)

    config_spec['network']=EMLP
    config_spec['net_config']['group'] = [Trivial(3),SO(3),O(3)]
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)
    print(thestudy.results_df())