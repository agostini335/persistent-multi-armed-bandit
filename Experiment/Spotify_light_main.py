#%%IMPORT SECTION
import sys
sys.path.append("Experiment")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import matplotlib.lines as mlines
from SpotifyScenario.SpotifyBucket import SpotifyBucket
from SpotifyScenario.SpotifyArm import SpotifyArm
from AnalyticsManager import AnalyticsManager
from ConfigManager import ConfigManager
from Environment import Environment
from Learners.Bound1Learner_myopic import Bound1Learner_myopic
from Learners.Bound1Learner_myopic_adaptive import Bound1Learner_myopic_adaptive
from Learners.Bound1Learner_farsighted import Bound1Learner_farsighted
from Learners.Bound1Learner_farsighted_adaptive import Bound1Learner_farsighted_adaptive
from Learners.BaselineLearner_myopic import BaselineLearner_myopic
from Learners.BaselineLearner_farsighted import BaselineLearner_farsighted
from Learners.ThompsonLearner import ThompsonLearner
from Learners.BaselineLearner_farsighted_adaptive import BaselineLearner_farsighted_adaptive
from Learners.BaselineLearner_myopic_adaptive import BaselineLearner_myopic_adaptive
from Learners.ThompsonLearner import ThompsonLearnerSpotify
from Learners.ThompsonLearner import ThompsonBaselineSpotify
from Learners.ThompsonLearner import ThompsonLearnerExplorerSpotify
from Learners.ThompsonLearner import BayesUCBPersistentSpotify
from Learners.Idea2 import Idea2,Idea2Positive,Idea2Spotify,Idea2PositiveSpotify
from Learners.Oracle import Oracle
from tqdm import tqdm
from Parser.ArmSet import ArmSet
from Parser.ExperimentDescription import ExperimentDescription
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import Utils
import pickle

#FUNCTION DEFINITION
def get_arms():
    _arms = []
    for pl in selected_playlist_rids:
        _arms.append(SpotifyArm(pl_avg[pl],pl))
    return _arms

def get_n_arms():
    return len(selected_playlist_rids)

def save_and_plot_run_results(show):
    for l in learners:
        plt.plot(Utils.calculate_cumulative_pseudo_regret(l.collected_arm_values,oracle.best_arm.value),label = l.name)
    plt.ylabel('cumulative pseudo regret')
    plt.title('cumulative pseudo regret '+experiment_name+"_run"+str(run))
    plt.legend(loc='lower left')
    plt.xlabel('Time t')
    plt_name = "Results/"+experiment_name+"_run"+str(run)+".png"
    plt.savefig(plt_name, bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()
    #SAVE CSV
    Utils.save_run_results(learners=learners ,run_id=run,exp_name=experiment_name)



#DATA LOADING
selected_playlist_rids = ["play_0", "play_2","play_1","play_4","play_6","play_8"] 

infile = open("pl_avg_output",'rb')
new_dict = pickle.load(infile)
infile.close()

pl_avg = new_dict

infile = open("pl_dict_output",'rb')
new_dict = pickle.load(infile)
infile.close()

play_dict = new_dict

#SETUP CONFIG
experiment_name = "experiment_spotify10k20r"
T = 20000 
n_runs = 50
tmax = 80
oracle = Oracle(get_n_arms(),get_arms(),tmax)

#%%RUN
for run in range(n_runs):
    print("run number: "+str(run+1)+" of "+str(n_runs))
    #LEARNERS SET UP
    learners = []
    thompson_learner_baseline = ThompsonBaselineSpotify(get_n_arms(),get_arms(),tmax)
    #thompson_learner_exp = ThompsonLearnerExplorerSpotify(get_n_arms(),get_arms(),tmax,exploration_factor=0.005)
    #thompson_learner_exp2 = ThompsonLearnerExplorerSpotify(get_n_arms(),get_arms(),tmax,exploration_factor=0.0025)
    #thompson_learner_exp3 = ThompsonLearnerExplorerSpotify(get_n_arms(),get_arms(),tmax,exploration_factor=0.0075)
    bayesUCB = BayesUCBPersistentSpotify(get_n_arms(),get_arms(),tmax)         
    bound1_learner_m = Bound1Learner_myopic(get_n_arms(),get_arms(),tmax)
    baseline_learner_m = BaselineLearner_myopic(get_n_arms(),get_arms(),tmax=tmax,tmin=0)
    idea2_learner = Idea2Spotify(get_n_arms(),get_arms(),tmax)
    idea2_learner_p = Idea2PositiveSpotify(get_n_arms(),get_arms(),tmax)
    oracle = Oracle(get_n_arms(),get_arms(),tmax)

    learners = [ baseline_learner_m, bound1_learner_m, bayesUCB, thompson_learner_baseline, idea2_learner,idea2_learner_p]

                
    #EXECUTION
    for i in tqdm(range(T)):    
        for learner in learners:
            arm = learner.pull_arm()
            bucket = SpotifyBucket(list(random.choice(play_dict[arm.id_code])))
            learner.update_observations(arm,bucket)
            
    save_and_plot_run_results(show = False)

#%% ANALYTICS
Utils.compute_full_analytics(n_runs=n_runs,time_horizon=T,oracle=oracle,_learners=learners,experiment_name=experiment_name)



    

  








