#%%
import sys
sys.path.append("Experiment")
from AnalyticsManager import AnalyticsManager
from ConfigManager import ConfigManager
from Environment.Environment import Environment
from Learners.Bound1Learner_myopic import Bound1Learner_myopic
from Learners.Bound1Learner_myopic_adaptive import Bound1Learner_myopic_adaptive
from Learners.Bound1Learner_farsighted import Bound1Learner_farsighted
from Learners.Bound1Learner_farsighted_adaptive import Bound1Learner_farsighted_adaptive
from Learners.BaselineLearner_myopic import BaselineLearner_myopic
from Learners.BaselineLearner_farsighted import BaselineLearner_farsighted
from Learners.ThompsonLearner import ThompsonLearner,ThompsonBaseline, BayesUCBPersistent
from Learners.BaselineLearner_farsighted_adaptive import BaselineLearner_farsighted_adaptive
from Learners.BaselineLearner_myopic_adaptive import BaselineLearner_myopic_adaptive
from Learners.Idea2 import Idea2, Idea2Positive
from Learners.Oracle import Oracle
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


def compute_experiment_pseudo_regret(learner_names,learner_arm_value_dict,oracle_value,n_runs,file_name):
    learner_pseudo_regret_dict = {}
    learner_average_pseudo_regret = {}
    learner_std = {}
    #"computing pseudo regret:"
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
    plt.title('cumulative pseudo regret '+ str(file_name))
    plt.legend(loc='upper left')
    plt.xlabel('Time t')
    print(file_name)
    print(n_runs)
    plt.savefig("Results/"+str(file_name)+"_FULL_ANALYSYS.png",dpi=600)
    #plt.show()
    plt.clf()    

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






experiment_name = "experiment_B"
config_manager = ConfigManager(path ="Experiment/Parser/ConfigFiles", name = experiment_name+".json")
env = Environment(config_manager.get_n_arms(), tmax = config_manager.get_tmax())
T = config_manager.get_time()       
n_runs = config_manager.get_n_runs()

print(T)

for run in range(n_runs):
    print("run number: "+str(run+1)+" of "+str(n_runs))
    #LEARNERS SET UP
    learners = []

    bound1_learner_m = Bound1Learner_myopic(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    bound1_learner_f = Bound1Learner_farsighted(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    baseline_learner_m = BaselineLearner_myopic(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),config_manager.get_tmin())
    baseline_learner_f = BaselineLearner_farsighted(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),config_manager.get_tmin())

    thompson_learner_m_s_mono = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=False,adaptive=False, monotonic = True)
    thompson_learner_m_a_mono = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=False,adaptive=True, monotonic = True)
    thompson_learner_f_s_mono = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=True,adaptive=False, monotonic = True)
    thompson_learner_f_a_mono = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=True,adaptive=True, monotonic = True)
    thompson_learner_m_s = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=False,adaptive=False, monotonic = False)
    thompson_learner_m_a = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=False,adaptive=True, monotonic = False)
    thompson_learner_f_s = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=True,adaptive=False, monotonic = False)
    thompson_learner_f_a = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=True,adaptive=True, monotonic = False)
    thompson_learner_m_s_opti = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=False,adaptive=False, monotonic = False, optimistic= True)
    thompson_learner_m_s_mono_opti = ThompsonLearner(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=False,adaptive=False, monotonic = True, optimistic= True)
    
    bound1_learner_f_adp = Bound1Learner_farsighted_adaptive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    bound1_learner_m_adp = Bound1Learner_myopic_adaptive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())


    thomposon_baseline_m = ThompsonBaseline(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),optimistic=False,farsighted=False)
    thomposon_baseline_f = ThompsonBaseline(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),optimistic=False,farsighted=True)
    bayes_m = BayesUCBPersistent(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    bayes_f = BayesUCBPersistent(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=True)
    idea2 = Idea2(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
    idea2_pos = Idea2Positive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())

    


    oracle = Oracle(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())

    print("ORACLE:"+str(oracle.best_arm.value))

    
    #learners = [bound1_learner_m,bound1_learner_f,bound1_learner_f_adp,bound1_learner_m_adp,baseline_learner_m,baseline_learner_f,thompson_learner_m_s,thompson_learner_m_a,thompson_learner_f_s,thompson_learner_f_a,
    #            thompson_learner_m_s_mono,thompson_learner_m_a_mono,thompson_learner_f_s_mono,thompson_learner_f_a_mono,thompson_learner_m_s_opti]
   
    #learners = [bound1_learner_m,bound1_learner_f,baseline_learner_m,baseline_learner_f,thompson_learner_m_s,thompson_learner_f_s,thompson_learner_m_s_mono,thompson_learner_m_s_opti]

    learners = [idea2,idea2_pos,baseline_learner_m,bound1_learner_m,baseline_learner_f,bound1_learner_f,thomposon_baseline_m,thomposon_baseline_f,bayes_f,bayes_m]
        
    #EXECUTION
    for i in tqdm(range(T)):    
        for learner in learners:
                arm = learner.pull_arm()
                bucket = env.round(arm)
                learner.update_observations(arm,bucket)
            

    #PLOT
    for l in learners:
        if l.name.__contains__('Thompson'):
            if l.name.__contains__("adp"):
                plt.plot(calculate_cumulative_pseudo_regret(l.collected_arm_values,oracle.best_arm.value),label = l.name, ls='--' )
            else:
                plt.plot(calculate_cumulative_pseudo_regret(l.collected_arm_values,oracle.best_arm.value),label = l.name, ls=':' )
        else:
            plt.plot(calculate_cumulative_pseudo_regret(l.collected_arm_values,oracle.best_arm.value),label = l.name)

    plt.ylabel('cumulative pseudo regret')
    plt.title('cumulative pseudo regret '+experiment_name+"_run"+str(run))
    plt.legend(loc='lower left')
    plt.xlabel('Time t')
    plt_name = "Results/"+experiment_name+"_run"+str(run)+".png"
    plt.savefig(plt_name, bbox_inches='tight')
    #plt.show()
    plt.clf()

    #SAVE CSV
    save_run_results(learners=learners ,run_id=run,exp_name=experiment_name)

    #SAVE_DESCPRITION
    save_experiment_description(config_manager)












#ANALYTICS
#-------------------------CONFIG----------------------------------#

N_RUNS = config_manager.get_n_runs()
time = config_manager.get_time()
Oracle_best_arm_value = oracle.best_arm.value  #TODO CONTROLLARE
print(Oracle_best_arm_value)
learner_names=[]
for learner in learners:
    learner_names.append(learner.name)

exp_name = experiment_name
step = ""

print(len(learner_names))
#dict learner-> list of list of played arms
learner_arm_value_dict = {}
for l in range(len(learner_names)):
    print(l)
    results_list_of_l = [] #it must have n_runs element
    for i in range(N_RUNS):
        full_name="Results/"+exp_name+"_arm_value_results_run_"+str(i)+".csv"
        data = pd.read_csv(full_name)
        arm_values_of_l_on_run_i = data[learner_names[l]]
        assert(len(arm_values_of_l_on_run_i) == time)
        #print(arm_values_of_l_on_run_i.head)
        results_list_of_l.append(arm_values_of_l_on_run_i)
    learner_arm_value_dict[learner_names[l]] = results_list_of_l

    #PLOT SINGLE RUN di VERIFICA
for run_id in range(N_RUNS):
    for l in learner_names:
        if l.__contains__('Thompson'):
            if l.__contains__("adp"):
                plt.plot(calculate_cumulative_pseudo_regret(learner_arm_value_dict[l][run_id],Oracle_best_arm_value),label = l, ls='--' )
            else:
                plt.plot(calculate_cumulative_pseudo_regret(learner_arm_value_dict[l][run_id],Oracle_best_arm_value),label = l, ls='-' )
        else:
            plt.plot(calculate_cumulative_pseudo_regret(learner_arm_value_dict[l][run_id],Oracle_best_arm_value),label = l)

    plt.ylabel('cumulative pseudo regret')
    plt.title('cumulative pseudo regret '+"exp_0_"+str(step)+"_run"+str(run_id))
    plt.legend(loc='lower left')
    plt.xlabel('Time t')
    plt_name = "Results/VERIFICA"+"experiment_4_"+str(step)+"_run_"+str(run_id)+".png"
    plt.savefig(plt_name, bbox_inches='tight')
    plt.clf()

output_name = str(step)
    #avg
compute_experiment_pseudo_regret(learner_names=learner_names,learner_arm_value_dict=learner_arm_value_dict,oracle_value=Oracle_best_arm_value,n_runs=N_RUNS,file_name=exp_name)



