import sys
sys.path.append("Experiment")
from csv import writer
import numpy as np
import tqdm
import pandas as pd
from Parser.ArmSet import ArmSet
from Parser.ExperimentDescription import ExperimentDescription
from matplotlib import rc
import matplotlib.pyplot as plt
plt.style.use("seaborn")

from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 


plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

color_list = ["purple","violet","darkred","red","mediumblue","dodgerblue","darkorange","gold","darkgreen","limegreen","black","dimgray","purple","fuchsia","purple","fuchsia","purple","fuchsia"]



import json

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


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
    print(learner_names)
    #"computing pseudo regret:"
    for learner in (learner_names) :
        pseudo_regret_of_learner = []
        for arm_values_run in learner_arm_value_dict[learner]:
            pseudo_regret_of_learner.append(calculate_cumulative_pseudo_regret(arm_values_run,oracle_value))
        learner_pseudo_regret_dict[learner] = pseudo_regret_of_learner
        learner_average_pseudo_regret[learner] = np.mean(pseudo_regret_of_learner,axis = 0)
        learner_std[learner] = np.std(pseudo_regret_of_learner,axis = 0)
    i = 0
    for learner in learner_names:
        x = np.arange(len(learner_average_pseudo_regret[learner]))
        confidence_interval = 2*learner_std[learner]/np.sqrt(n_runs)
        (_, caps, _) =plt.errorbar(x,learner_average_pseudo_regret[learner],confidence_interval,elinewidth=0.9,capsize=3, errorevery=4000 ,label = learner,color=color_list[i])
        i = i+1
        for cap in caps:
            cap.set_markeredgewidth(1)
    plt.ylabel('Pseudo Regret')
    plt.title(str(file_name))
    plt.legend(loc='upper left')
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox = False, shadow = False)

    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time')
    plt.savefig("Results/"+ str(file_name)+" ANALYTICS.png", bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=600)
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


def compute_full_analytics(n_runs,time_horizon,oracle,_learners,experiment_name):
    N_RUNS = n_runs
    time = time_horizon
    Oracle_best_arm_value = oracle.best_arm.value  
    learner_names=[]
    for learner in _learners:
        learner_names.append(learner.name)

    exp_name = experiment_name
    step = ""
    print(learner_names)
    
    #dict learner-> list of list of played arms
    learner_arm_value_dict = {}
    for l in range(len(learner_names)):
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
            plt.plot(calculate_cumulative_pseudo_regret(learner_arm_value_dict[l][run_id],Oracle_best_arm_value),label = l)

        plt.ylabel('cumulative pseudo regret')
        plt.title('cumulative pseudo regret '+"exp_0_"+str(step)+"_run"+str(run_id))
        plt.legend(loc='lower left')
        plt.xlabel('Time t')
        plt_name = "Results/VERIFICA"+"experiment_4_"+str(step)+"_run_"+str(run_id)+".png"
        plt.savefig(plt_name, bbox_inches='tight')
        plt.clf()

    compute_experiment_pseudo_regret(learner_names=learner_names,learner_arm_value_dict=learner_arm_value_dict,oracle_value=Oracle_best_arm_value,n_runs=N_RUNS,file_name=exp_name)
    print("DONE")


def compute_full_analytics_from_files(n_runs,time_horizon,Oracle_best_arm_value,learner_names,experiment_name):
    N_RUNS = n_runs
    time = time_horizon
    Oracle_best_arm_value
    exp_name = experiment_name
    step = ""
    print(learner_names)
    
    #dict learner-> list of list of played arms
    learner_arm_value_dict = {}
    for l in range(len(learner_names)):
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
            plt.plot(calculate_cumulative_pseudo_regret(learner_arm_value_dict[l][run_id],Oracle_best_arm_value),label = l)


        plt.ylabel('cumulative pseudo regret')
        plt.title('cumulative pseudo regret '+"exp_0_"+str(step)+"_run"+str(run_id))
        plt.legend(loc='lower left')
        plt.xlabel('Time t')
        plt_name = "Results/VERIFICA"+"experiment_4_"+str(step)+"_run_"+str(run_id)+".png"
        plt.savefig(plt_name, bbox_inches='tight')
        plt.clf()

    compute_experiment_pseudo_regret(learner_names=learner_names,learner_arm_value_dict=learner_arm_value_dict,oracle_value=Oracle_best_arm_value,n_runs=N_RUNS,file_name=exp_name)
    print("DONE")

ln = ["Idea2_zeros","Idea2_ones","Baseline_myopic","Baseline_farsighted","Bound1_myopic","Bound1_farsighted","Thompson_baseline_myopic","Thompson_baseline_farsighted","BayesUCBPersistentfarsighted","BayesUCBPersistentmyopic"]
compute_full_analytics_from_files(n_runs=50,time_horizon=20000,Oracle_best_arm_value=30.500000000002828,learner_names=ln,experiment_name="experiment_B")