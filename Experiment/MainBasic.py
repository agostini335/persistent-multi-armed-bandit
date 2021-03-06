#%%
import sys
sys.path.append("Experiment")
from ConfigManager import ConfigManager
from Environment import Environment
from Bound1Learner_myopic import Bound1Learner_myopic
from Bound1Learner_farsighted import Bound1Learner_farsighted
from BaselineLearner_myopic import BaselineLearner_myopic
from BaselineLearner_farsighted import BaselineLearner_farsighted
from Bound1Learner_farsighted_adaptive import Bound1Learner_farsighted_adaptive

from BaselineLearner_farsighted_adaptive import BaselineLearner_farsighted_adaptive

from Bound1Learner_myopic_adaptive import Bound1Learner_myopic_adaptive
from BaselineLearner_myopic_adaptive import BaselineLearner_myopic_adaptive
from Oracle import Oracle
from tqdm import tqdm
from Parser.ArmSet import ArmSet
from Parser.ExperimentDescription import ExperimentDescription
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


def calculate_cumulative_pseudo_regret(collected_arm_values,oracle_arm_value):
    istant_regret = oracle_arm_value - np.array(collected_arm_values)
    cumulative_regret = np.zeros(len(collected_arm_values))
    cumulative_regret[0]=istant_regret[0]
    for i in range(1,len(collected_arm_values)):
        cumulative_regret[i]=cumulative_regret[i-1]+istant_regret[i]
    return cumulative_regret

def save_run_results(learners,run_id,exp_name):
    results = []
    names = []
    for l in learners:
        results.append(l.played_arms_id)
        names.append(l.name)
    #creo le tuple per pandas
    results_tuples_list = list(zip(*results))
    # Converting lists of tuples into 
    # pandas Dataframe. 
    df = pd.DataFrame(results_tuples_list, columns = names)
    name = "Results/"+exp_name+"_id_results_run_"+str(run_id)+".csv"
    df.to_csv(name, index= True)

    #faccio lo stesso ma con gli arm values
    results = []
    names = []
    for l in learners:
        results.append(l.collected_arm_values)
        names.append(l.name)
    results_tuples_list = list(zip(*results))
    df = pd.DataFrame(results_tuples_list, columns = names)
    name = "Results/"+exp_name+"_arm_value_results_run_"+str(run_id)+".csv"
    df.to_csv(name, index= True)

def save_experiment_description(cnfg_m):
    #converting list of arms in arm set extended
    arms = cnfg_m.get_arm_list()
    arm_set_list = []
    # creo una lista estesa di arm rispetto al file di configurazione
    # in questo modo ho tutti gli id corretti
    for arm in arms:
        a_set =ArmSet(arm.alpha,arm.beta,arm.reward,quantity = 1,increment = 0, id_code = arm.id_code,arm_value = arm.value)
        arm_set_list.append(a_set)
        experiment_desc = ExperimentDescription(cnfg_m.get_tmax(),cnfg_m.get_time(),arm_set_list,cnfg_m.get_n_runs())

    #saving
    name = str("Results/EXT_"+cnfg_m.name)
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(experiment_desc,   f, default=lambda o: o.__dict__,ensure_ascii=False, indent=4)



experiment_name = "experiment_100"
config_manager = ConfigManager(path ="Experiment/Parser/ConfigFiles", name = experiment_name+".json")
env = Environment(config_manager.get_n_arms(), tmax = config_manager.get_tmax())
T = config_manager.get_time()       
n_runs = config_manager.get_n_runs()


for run in range(n_runs):
    print("run number: "+str(run+1)+" of "+str(n_runs))
    #LEARNERS SET UP
    bound1_learner_m = Bound1Learner_myopic(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    bound1_learner_f = Bound1Learner_farsighted(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    bound1_learner_f_a = Bound1Learner_farsighted_adaptive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    bound1_learner_f_m = Bound1Learner_myopic_adaptive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    

    baselineLearner_m_adp = BaselineLearner_myopic_adaptive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),config_manager.get_tmin())
    baselineLearner_f_adp = BaselineLearner_farsighted_adaptive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),config_manager.get_tmin())
    
    baseline_m = BaselineLearner_myopic(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),config_manager.get_tmin())
    baseline_f = BaselineLearner_farsighted(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),config_manager.get_tmin())
    oracle = Oracle(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())

    print("ORACLE:"+str(oracle.best_arm.value))

    learners = [bound1_learner_m,bound1_learner_f,bound1_learner_f_a,bound1_learner_f_m,baselineLearner_m_adp,baselineLearner_f_adp,baseline_f,baseline_m]
        
    #EXECUTION
    for i in tqdm(range(T)):    
        for learner in learners:
            arm = learner.pull_arm()
            bucket = env.round(arm)
            learner.update_observations(arm,bucket)
            

    #PLOT
    for l in learners:
        plt.plot(calculate_cumulative_pseudo_regret(l.collected_arm_values,oracle.best_arm.value),label = l.name)

    plt.ylabel('cumulative pseudo regret')
    plt.title('cumulative pseudo regret '+experiment_name+"_run"+str(run))
    plt.legend(loc='lower left')
    plt.xlabel('Time t')
    plt_name = "Results/"+experiment_name+"_run"+str(run)+".png"
    #plt.savefig(plt_name, bbox_inches='tight')
    plt.show()
    plt.clf()

    #SAVE CSV
    #save_run_results(learners=learners ,run_id=run,exp_name=experiment_name)

    #SAVE_DESCPRITION
    #save_experiment_description(config_manager)