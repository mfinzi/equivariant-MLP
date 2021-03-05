from emlp.models.datasets import ParticleInteraction
from emlp.solver.groups import O13
from emlp.experiments.train_basic import makeTrainer
from oil.tuning.args import argupdated_config
import emlp
if __name__ == "__main__":
    cfg = makeTrainer.__kwdefaults__
    
    cfg['dataset'] = ParticleInteraction
    cfg['net_config']['group'] = O13()

    cfg = argupdated_config(cfg,namespace=(emlp.solver.groups,emlp.models.datasets,emlp.models.mlp))
    trainer = makeTrainer(**cfg)
    trainer.train(cfg['num_epochs'])