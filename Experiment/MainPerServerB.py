#%%
import sys
sys.path.append("Experiment")
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
import json




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



ex_list = ["experiment_C_50"]

for ex in ex_list:
    experiment_name = ex
    config_manager = ConfigManager(path ="Experiment/Parser/ConfigFiles", name = experiment_name+".json")
    env = Environment(config_manager.get_n_arms(), tmax = config_manager.get_tmax())
    T = config_manager.get_time()       
    n_runs = config_manager.get_n_runs()

    print(T)

    for run in range(n_runs):
        print("run number: "+str(run+1)+" of "+str(n_runs)+ experiment_name)
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
        idea2_pos_m = Idea2Positive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())
        idea2_pos_f = Idea2Positive(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax(),farsighted=True)

        


        oracle = Oracle(config_manager.get_n_arms(),config_manager.get_arm_list(),config_manager.get_tmax())

        print("ORACLE:"+str(oracle.best_arm.value))

        learners = [idea2_pos_m,idea2_pos_f,baseline_learner_m,baseline_learner_f,bound1_learner_m,bound1_learner_f,thomposon_baseline_m,thomposon_baseline_f,bayes_m,bayes_f]
            
        #EXECUTION
        for i in tqdm(range(T)):    
            for learner in learners:
                    arm = learner.pull_arm()
                    bucket = env.round(arm)
                    learner.update_observations(arm,bucket)
                

        #SAVE CSV
        save_run_results(learners=learners ,run_id=run,exp_name=experiment_name)

        #SAVE_DESCPRITION
        save_experiment_description(config_manager)