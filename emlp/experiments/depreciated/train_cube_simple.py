
from emlp.models.datasets import InvertedCube
from emlp.solver.groups import Cube
from emlp.models.mlp import MLP,EMLP,Standardize
from emlp.models.model_trainer import ClassifierPlus
from emlp.solver.representation import T
from slax.model_trainers import Classifier
from torch.utils.data import DataLoader
from oil.utils.utils import cosLr, islice, export,FixedNumpySeed,FixedPytorchSeed
from slax.utils import LoaderTo
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
import objax
import logging
import emlp.models

# intermediate_rep1 = (100*T(0)+5*T(1))#+T(2))
# intermediate_rep2 = 100*T(0)+10*T(1)
#middle_reps = [intermediate_rep1,intermediate_rep2,intermediate_rep1]
rep = 138*T(0)+23*T(1)+T(2)
def makeTrainer(*,network=EMLP,num_epochs=500,seed=2020,aug=False,
                bs=50,lr=1e-3,device='cuda',
                net_config={'num_layers':3,'ch':rep,'group':Cube()},log_level='info',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':.02,'timeFrac':50}},save=False):
    levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                        'warn': logging.WARNING,'warning': logging.WARNING,
                        'info': logging.INFO,'debug': logging.DEBUG}
    logging.getLogger().setLevel(levels[log_level])
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(seed),FixedPytorchSeed(seed):
        datasets = {'train':InvertedCube(train=True),'test':InvertedCube(train=False)}
    model = Standardize(network(datasets['train'].rep_in,datasets['train'].rep_out,**net_config),datasets['train'].stats)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(v)),shuffle=(k=='train'),
                num_workers=0,pin_memory=False)) for k,v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    opt_constr = objax.optimizer.Adam
    lr_sched = lambda e: lr*cosLr(num_epochs)(e)
    return ClassifierPlus(model,dataloaders,opt_constr,lr_sched,**trainer_config)

if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    Trial(argupdated_config(makeTrainer.__kwdefaults__,namespace=(emlp.solver.groups,emlp.models.datasets,emlp.models.mlp)))