from emlp.models.datasets import Inertia
from emlp.solver.groups import O
from emlp.experiments.train_basic import makeTrainer
from oil.tuning.args import argupdated_config
import emlp
if __name__ == "__main__":
    cfg = makeTrainer.__kwdefaults__
    
    cfg['dataset'] = Inertia
    cfg['net_config']['group'] = O(3)

    cfg = argupdated_config(cfg,namespace=(emlp.solver.groups,emlp.models.datasets,emlp.models.mlp))
    trainer = makeTrainer(**cfg)
    trainer.train(cfg['num_epochs'])
    