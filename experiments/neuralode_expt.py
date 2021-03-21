from emlp.nn import MLP,EMLP,MLPH,EMLPH,EMLPode,MLPode#,LinearBNSwish
from emlp.groups import SO2eR3,O2eR3,DkeR3,Trivial
from oil.tuning.study import Study

import copy
from trainer.hamiltonian_dynamics import ode_trial
from neuralode import makeTrainer

if __name__ == "__main__":
    Trial = ode_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    name = "ode_expt"#config_spec.pop('study_name')
    
    #name = f"{name}_{config_spec['dataset']}"
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    config_spec['network'] = EMLPode
    config_spec['net_config']['group'] = [O2eR3(),SO2eR3(),DkeR3(6),DkeR3(2)]
    thestudy.run(num_trials=-5,new_config_spec=config_spec,ordered=True)
    config_spec['network'] = MLPode
    config_spec['net_config']['group'] = None
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)
    print(thestudy.results_df())