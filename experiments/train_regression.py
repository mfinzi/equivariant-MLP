from emlp.nn import MLP, EMLP, Standardize
from trainer.model_trainer import RegressorPlus
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, FixedNumpySeed, FixedPytorchSeed
from trainer.utils import LoaderTo
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import logging
import emlp.nn
import emlp.reps
import emlp.groups
import objax
import emlp.datasets
from emlp.datasets import Inertia,O5Synthetic,ParticleInteraction

log_levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                        'warn': logging.WARNING,'warning': logging.WARNING,
                        'info': logging.INFO,'debug': logging.DEBUG}

def makeTrainer(*,dataset=Inertia,network=EMLP,num_epochs=300,ndata=1000+2000,seed=2021,aug=False,
                bs=500,lr=3e-3,device='cuda',split={'train':-1,'val':1000,'test':1000},
                net_config={'num_layers':3,'ch':384,'group':None},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':.75},
                'early_stop_metric':'val_MSE'},save=False,):
    
    logging.getLogger().setLevel(log_levels[log_level])
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
    lr_sched = lambda e: lr#*min(1,e/(num_epochs/10)) # Learning rate warmup
    return RegressorPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    cfg = argupdated_config(makeTrainer.__kwdefaults__,
                    namespace=(emlp.groups,emlp.datasets,emlp.nn))
    trainer = makeTrainer(**cfg)
    trainer.train(cfg['num_epochs'])
