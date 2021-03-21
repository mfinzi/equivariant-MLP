from oil.tuning.study import Study
from emlp.nn import MLP,EMLP,MLPH,EMLPH
from emlp.groups import SO2eR3,O2eR3,DkeR3,Trivial

import copy
from trainer.hamiltonian_dynamics import hnn_trial
from hnn import makeTrainer

if __name__=="__main__":
    Trial = hnn_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    name = "hnn_expt"#config_spec.pop('study_name')
    #name = f"{name}_{config_spec['dataset']}"
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    config_spec['network'] = EMLPH
    config_spec['net_config']['group'] = [O2eR3(),SO2eR3(),DkeR3(6),DkeR3(2)]
    thestudy.run(num_trials=-5,new_config_spec=config_spec,ordered=True)
    config_spec['network'] = MLPH
    config_spec['net_config']['group'] = None
    thestudy.run(num_trials=-3,new_config_spec=config_spec,ordered=True)
    print(thestudy.results_df())
