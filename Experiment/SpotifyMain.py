#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from datetime import datetime
import matplotlib.lines as mlines
from tqdm import tqdm
import hashlib

'''
play_0     1888
play_2      840
play_1      694
play_4      487
play_6      143
play_8      110
'''

selected_playlist_rids = ["play_0", "play_2","play_1","play_4","play_6","play_8"] 

#loading
df = pd.read_csv("Datasets/spotifydf_012.csv",sep=",")
df = df[["session_id","r_playlist_id","track_id_clean","skip_1","skip_2","skip_3","not_skipped"]]
print(df.head(60))


#%%
play_dict = dict()
for pl in selected_playlist_rids:
    play_dict[pl] = []
   
    df_sel = df[df["r_playlist_id"] == pl]
    session_list = df_sel.session_id.unique()

    for ses in session_list:
        df_sess = df_sel[df_sel["session_id"] == ses]
        bucket = []
        for i,r in df_sess.iterrows():
            bucket.append(int((r['skip_1']) == False))
            bucket.append(int((r['skip_2']) == False))
            bucket.append(int((r['skip_3']) == False))
            bucket.append(int((r['not_skipped']) == True))
        play_dict[pl].append(bucket)


            
            
#%% STATS
import numpy
pl_avg = dict()
for pl in selected_playlist_rids:
    print(pl+str(" len:"+str(len(play_dict[pl]))))
    summed = []
    for x in play_dict[pl]:
        summed.append(sum(x))
    print("avg" + str(np.average(summed)))
    pl_avg[pl] = np.average(summed)
    print("std" + str(np.std(summed)))
    print("")
print(pl_avg)
#%%
''''
#%% STATS... spotify ok
play_0 len:1888
avg39.23146186440678
std22.04624678804135
play_2 len:840
avg51.55357142857143
std20.733501452552417
play_1 len:694
avg37.09365994236311
std22.764896383557325
play_4 len:487
avg42.53593429158111
std23.484851211949966
play_6 len:143
avg45.86013986013986
std24.016486586352865
play_8 len:110
avg36.8
std24.08839781079076

spotify 012

play_0 len:5539
avg38.59180357465247
std21.836365913544693

play_2 len:2551
avg52.35280282242258
std20.10897864468572

play_1 len:2052
avg38.44249512670565
std23.029710178144935

play_4 len:1399
avg43.89706933523946
std23.14085251299254

play_6 len:430
avg45.146511627906975
std23.483976217711746

play_8 len:329
avg36.20668693009119
std23.8058219545636




'''






#%%
import sys
sys.path.append("Experiment")
from SpotifyScenario.SpotifyBucket import SpotifyBucket
from SpotifyScenario.SpotifyArm import SpotifyArm
from AnalyticsManager import AnalyticsManager
from ConfigManager import ConfigManager
from Environment import Environment
from Bound1Learner_myopic import Bound1Learner_myopic
from Bound1Learner_myopic_adaptive import Bound1Learner_myopic_adaptive
from Bound1Learner_farsighted import Bound1Learner_farsighted
from Bound1Learner_farsighted_adaptive import Bound1Learner_farsighted_adaptive
from BaselineLearner_myopic import BaselineLearner_myopic
from BaselineLearner_farsighted import BaselineLearner_farsighted
from ThompsonLearner import ThompsonLearner
from BaselineLearner_farsighted_adaptive import BaselineLearner_farsighted_adaptive
from BaselineLearner_myopic_adaptive import BaselineLearner_myopic_adaptive
from ThompsonLearner import ThompsonLearnerSpotify
from ThompsonLearner import ThompsonBaselineSpotify
from ThompsonLearner import ThompsonLearnerExplorerSpotify
from Oracle import Oracle
from tqdm import tqdm
from Parser.ArmSet import ArmSet
from Parser.ExperimentDescription import ExperimentDescription
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random

#%%

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
    plt.savefig("Results/"+ str(file_name)+" n_runs:"+str(n_runs)+".png",dpi=600)
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

def get_arms():
    _arms = []
    for pl in selected_playlist_rids:
        _arms.append(SpotifyArm(pl_avg[pl],pl))
    return _arms
        



def get_n_arms():
    return len(selected_playlist_rids)


experiment_name = "experiment_spotify"
T = 50000       
n_runs = 500
tmax = 80
oracle = Oracle(get_n_arms(),get_arms(),tmax)
#%%

for run in range(n_runs):
    print("run number: "+str(run+1)+" of "+str(n_runs))
    #LEARNERS SET UP
    learners = []

    thompson_learner = ThompsonLearnerSpotify(get_n_arms(),get_arms(),tmax,optimistic=False)
    thompson_learner_opti = ThompsonLearnerSpotify(get_n_arms(),get_arms(),tmax,optimistic=True)
    thompson_learner_baseline = ThompsonBaselineSpotify(get_n_arms(),get_arms(),tmax)
    thompson_learner_exp = ThompsonLearnerExplorerSpotify(get_n_arms(),get_arms(),tmax,exploration_factor=0.005)
    thompson_learner_exp2 = ThompsonLearnerExplorerSpotify(get_n_arms(),get_arms(),tmax,exploration_factor=0.0025)
    thompson_learner_exp3 = ThompsonLearnerExplorerSpotify(get_n_arms(),get_arms(),tmax,exploration_factor=0.0075)     
    
    
    bound1_learner_m = Bound1Learner_myopic(get_n_arms(),get_arms(),tmax)
   
   
    baseline_learner_m = BaselineLearner_myopic(get_n_arms(),get_arms(),tmax=tmax,tmin=0)


    oracle = Oracle(get_n_arms(),get_arms(),tmax)

    print("ORACLE:"+str(oracle.best_arm.value))

    learners = [ bound1_learner_m,baseline_learner_m,thompson_learner_baseline,thompson_learner_exp2,thompson_learner_exp,thompson_learner_exp3]
        
    #EXECUTION
    for i in tqdm(range(T)):    
        for learner in learners:
            arm = learner.pull_arm()
            bucket = SpotifyBucket(list(random.choice(play_dict[arm.id_code])))
            learner.update_observations(arm,bucket)
            
    #print(baseline_learner_m.criterion)
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
    # save_experiment_description(config_manager)









#%%


#ANALYTICS
#-------------------------CONFIG----------------------------------#
#T = 3000       
#n_runs = 5
#tmax = 80


N_RUNS = n_runs
time = T
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
compute_experiment_pseudo_regret(learner_names=learner_names,learner_arm_value_dict=learner_arm_value_dict,oracle_value=Oracle_best_arm_value,n_runs=N_RUNS,file_name=step)


#%%
