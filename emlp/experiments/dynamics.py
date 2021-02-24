import copy, warnings
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam,AdamW
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from biases.systems.chain_pendulum import ChainPendulum
import biases.datasets as datasets
from biases.models import HNN,LNN,NN,CHNN,CH,CL
from biases.datasets import RigidBodyDataset
from biases.dynamics_trainer import IntegratedDynamicsTrainer
import biases.models as models
import biases.systems as systems
import pickle

import torch.nn as nn
from biases.models.utils import FCtanh,FCswish,FCsoftplus, Linear, Reshape
from typing import Optional, Tuple, Union
from oil.utils.utils import export, Named
import torch.nn.functional as F
from emlp.solver.representation import T,Scalar,Matrix,Vector
from emlp.solver.groups import SO,O,Trivial
from emlp.models.mlp import MLP, EMLP,LieLinear, uniform_rep, TensorLinearBNSwish,TensorLinear


class ECHNN(CHNN):
    def __init__(self,G,group=Trivial(3),dof_ndim=None,
                    hidden_size= 256,num_layers=3,**kwargs):
        super().__init__(G=G, dof_ndim=dof_ndim, **kwargs)
        ambient_d = self.dof_ndim
        n = len(G.nodes())
        print(ambient_d,n)
        rep_in = n*Vector
        rep_out = Scalar
        repmiddle = (100*T(0)+30*T(1)+10*T(2)+3*T(3))(group)#uniform_rep(hidden_size,group)
        reps = [rep_in(group)]+num_layers*[repmiddle]
        self.potential_net = nn.Sequential(
            *[TensorLinearBNSwish(rin,rout) for rin,rout in zip(reps,reps[1:])],#
            TensorLinear(repmiddle,rep_out(group)),
            Reshape(-1),
        )

    def compute_V(self, x):
        """ Input is a canonical position variable and the system parameters,
        Args:
            x: (bs x n_dof x dof_ndim) sized Tensor representing the position in
            Cartesian coordinates
        Returns: a length bs Tensor representing the potential energy
        """
        assert x.ndim == 3
        return self.potential_net(x.reshape(x.size(0), -1))


# network = HNN, LNN, NN, CHNN
def makeTrainer(*,network=ECHNN,net_cfg={},lr=3e-3,n_train=800,regen=False,
        dataset=RigidBodyDataset,body=systems.CoupledPendulum(3),C=5,
        dtype=torch.float32,device=torch.device("cuda"),
        bs=200,num_epochs=100,trainer_config={},
        opt_cfg={'weight_decay':1e-5}):
    # Create Training set and model
    angular = not issubclass(network,(CH,CL))
    splits = {"train": n_train,"test": 200}
    with FixedNumpySeed(0):
        dataset = dataset(n_systems=n_train+200, regen=regen, chunk_len=C,body=body,angular_coords=angular)
        datasets = split_dataset(dataset, splits)

    
    dof_ndim = dataset.body.D if angular else dataset.body.d
    model = network(dataset.body.body_graph,dof_ndim =dof_ndim,
                    angular_dims=dataset.body.angular_dims,**net_cfg)
    model = model.to(device=device, dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k: LoaderTo(
                DataLoader(v, batch_size=min(bs, splits[k]), num_workers=0, shuffle=(k == "train")),
                device=device,dtype=dtype) for k, v in datasets.items()}
    dataloaders["Train"] = dataloaders["train"]
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: AdamW(params, lr=lr,**opt_cfg)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,
                            log_args={"timeFrac": 1 / 4, "minPeriod": 0.0},**trainer_config)


#Trial = train_trial(makeTrainer)
if __name__ == "__main__":
    with FixedNumpySeed(0):
        defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
        defaults["save"] = False
        namespace = (datasets,systems,models)
        cfg = argupdated_config(defaults, namespace=namespace)
        cfg.pop('local_rank')
        save = cfg.pop('save')
        trainer = makeTrainer(**cfg)
        trainer.train(cfg['num_epochs'])
        if save: print(f"saved at: {trainer.save_checkpoint()}")
        #rollouts = trainer.test_rollouts(angular_to_euclidean= not issubclass(cfg['network'],(CH,CL)))
        #print(f"rollout error GeoMean {rollouts[0][:,1:].log().mean().exp():.3E}")
        #fname = f"rollout_errs_{cfg['network']}_{cfg['body']}.np"
        #with open(fname,'wb') as f:
        #    pickle.dump(rollouts,f)
        #defaults["trainer_config"]["early_stop_metric"] = "val_MSE"
        #print(Trial()))