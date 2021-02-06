#%%
import sys
sys.path.append("Experiment")
from AnalyticsManager import AnalyticsManager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def calculate_cumulative_pseudo_regret(collected_arm_values,oracle_arm_value):
    istant_regret = oracle_arm_value - np.array(collected_arm_values)
    cumulative_regret = np.zeros(len(collected_arm_values))
    cumulative_regret[0]=istant_regret[0]
    for i in range(1,len(collected_arm_values)):
        cumulative_regret[i]=cumulative_regret[i-1]+istant_regret[i]
    return cumulative_regret

def compute_experiment_pseudo_regret(learner_names,learner_arm_value_dict,oracle_value,n_runs,file_name,inc_step):
    learner_pseudo_regret_dict = {}
    learner_average_pseudo_regret = {}
    learner_std = {}
    "computing pseudo regret:"
    for learner in tqdm(learner_names) :
        pseudo_regret_of_learner = []
        for arm_values_run in learner_arm_value_dict[learner]:
            pseudo_regret_of_learner.append(calculate_cumulative_pseudo_regret(arm_values_run,oracle_value))
        learner_pseudo_regret_dict[learner] = pseudo_regret_of_learner
        learner_average_pseudo_regret[learner] = np.mean(pseudo_regret_of_learner,axis = 0)
        learner_std[learner] = np.std(pseudo_regret_of_learner,axis = 0)

    for learner in learner_names:
        x = np.arange(len(learner_average_pseudo_regret[learner]))
        confidence_interval = 2*learner_std[learner]/np.sqrt(n_runs)
        plt.errorbar(x,learner_average_pseudo_regret[learner],confidence_interval,elinewidth=0.9,capsize=2,errorevery=7000 ,label = learner)

    plt.ylabel('cumulative pseudo regret')
    plt.title('cumulative pseudo regret '+ str(file_name)+"inc:"+str(inc_step))
    plt.legend(loc='upper left')
    plt.xlabel('Time t')
    plt.savefig("/home/ago/Documenti/MABResults/incrementale_full/cumulative_pseudo_regret_"+ str(file_name)+" n_runs:"+str(n_runs)+"inc"+str(inc_step)+".png",dpi=600)
    plt.show()
    plt.clf()






increment_steps = [1,1.5,2,2.5,3,3.5,4]


N_RUNS = 10
learner_names = ["Baseline_myopic","Baseline_farsighted","Bound1_myopic","Bound1_farsighted"]
#time solo per asserzione
time = 100000
file_name = "INCexperiment_0_20"

Oracle_best_arm_value = 210.0 #TODO MODIFICARE

for increment_step in increment_steps:

    #dict learner-> list of list of played arms
    learner_arm_value_dict = {}

    for l in range(len(learner_names)):
        results_list_of_l = [] #it must have n_runs element
        for i in range(N_RUNS):
            full_name ="/home/ago/Documenti/MABResults/incrementale_full/Results/"+str(file_name)+"_arm_value_results_run_"+str(i)+"_"+str(increment_step)+".csv"
            data = pd.read_csv(full_name)
            arm_values_of_l_on_run_i = data[learner_names[l]]
            assert(len(arm_values_of_l_on_run_i) == time)
            print(arm_values_of_l_on_run_i.head)
            results_list_of_l.append(arm_values_of_l_on_run_i)
        learner_arm_value_dict[learner_names[l]] = results_list_of_l

    compute_experiment_pseudo_regret(learner_names=learner_names,learner_arm_value_dict=learner_arm_value_dict,oracle_value=Oracle_best_arm_value,n_runs=N_RUNS,file_name=file_name,inc_step=increment_step)
