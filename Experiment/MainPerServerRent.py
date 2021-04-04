import sys
sys.path.append("Experiment")
sys.path.append("Experiment/RentScenario")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
from RentScenario.RentArm import RentArm
from RentScenario.RentEnv import RentEnv
from RentScenario.RentBucket import RentBucket
from RentScenario.RentUCBLearner_single import RentUCBLearner_single_m,RentUCBLearner_single_f
from RentScenario.RentUCBLearner_double import RentUCBLearner_double
from RentScenario.RentBound1Learner import RentBound1Learner
from RentScenario.RentBound1Learner import RentBound1LearnerPartial
from RentScenario.Idea2Rent import Idea2_rent
from RentScenario.TSLearnerRent import ThompsonBaselineRent,BayesUCBPersistentRent
from tqdm import tqdm






#%% DATASET EXTRACION

CONVERSIONE_DAY_MONTH = 30.4
#CONVERSIONE_DAY_MONTH = 1
df = pd.read_csv("Datasets/BERTIERI/BertieriSel.csv",sep=",")
df['clu_datainiziocontratto'] =pd.to_datetime(df.clu_datainiziocontratto,dayfirst=True)
df["durata_affitto"] =pd.to_timedelta(df.durata_affitto)
canoni_distinct = df["clu_canonemensile"].unique().tolist()
canoni_distinct.sort()
print(canoni_distinct)
sfitto_dict = {}
for c in canoni_distinct :
    sfitto_dict[str(c)] =[]

posto_letto_distinct = df["id_postoletto"].unique().tolist()
for p in posto_letto_distinct:
    df_temp = df[df["id_postoletto"]==p]
    df_temp = df_temp.sort_values(by = ["clu_datainiziocontratto"])
    canoni_temp = df_temp["clu_canonemensile"].tolist()
    date_inizio_temp = df_temp["clu_datainiziocontratto"].tolist()
    durata_temp =df_temp["durata_affitto"].tolist()

    
    for i in range(1,len(canoni_temp)):
        canone_key = canoni_temp[i]
        sfitto = (date_inizio_temp[i]-date_inizio_temp[i-1]).days - pd.to_timedelta(durata_temp[i-1]).days
        sfitto_dict[str(canone_key)].append(int(sfitto/CONVERSIONE_DAY_MONTH))


    
avg_sfitto = []
std_sfitto = []
num_sfitto = []
for c in canoni_distinct:
    avg_sfitto.append(np.average(np.array(sfitto_dict[str(c)])))
    std_sfitto.append(np.std(np.array(sfitto_dict[str(c)])))
    num_sfitto.append(len(sfitto_dict[str(c)]))

confidence_interval = (2*np.array(std_sfitto))/np.sqrt(num_sfitto)



#contratto dict
contratto_dict = {}
for c in canoni_distinct :
    contratto_dict[str(c)] =[]
canoni_temp = df["clu_canonemensile"].tolist()

durata_temp = df["durata_affitto"].tolist()



for i in range(len(canoni_temp)):
    contratto_dict[str(canoni_temp[i])].append(int(durata_temp[i].days/CONVERSIONE_DAY_MONTH))

print(contratto_dict)
#TODO REMOVE #########

l = np.array(contratto_dict["610.0"])  * 1
contratto_dict["610.0"] = l.tolist() 

#print(contratto_dict)

#print(sfitto_dict)

#####################


avg_sfitti_dict = {}
avg_contratti_dict = {}
for c in canoni_distinct:
        avg_contratti_dict[str(c)] = np.average(np.array(contratto_dict[str(c)]))
        avg_sfitti_dict[str(c)] = np.average(np.array(sfitto_dict[str(c)]))


#%% FUNCTIONS DEF

def get_arm_list(c_list,tmax, avg_sfitti_dict,avg_contratti_dict):
        arm_list = []
        i = 0
        for c in c_list:
                k_a_s = avg_contratti_dict[str(c)]/(avg_sfitti_dict[str(c)]+avg_contratti_dict[str(c)])
                arm_list.append(RentArm(canone=c,k_a_s=k_a_s,id_code=i))
                i =i+1        
        return arm_list

def get_oracle(c_list,tmax, avg_sfitti_dict,avg_contratti_dict):
        arms = get_arm_list(c_list,tmax, avg_sfitti_dict,avg_contratti_dict)
        best = -1
        for a in arms:
                if a.value > best :
                        oracle = RentArm(canone = a.canone, k_a_s= a.k_a_s)
                        best = a.value
        return oracle

#%%PLOT 



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
        plt.errorbar(x,learner_average_pseudo_regret[learner],confidence_interval,elinewidth=0.9,capsize=2,errorevery=1000 ,label = learner)

    plt.ylabel('cumulative pseudo regret')
    plt.title('cumulative pseudo regret '+ str(file_name))
    plt.legend(loc='upper left')
    plt.xlabel('Time t')
    plt.savefig("Results/"+ str(file_name)+"FULLANALYTICS.png",dpi=600)
    plt.show()
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





#%% TIME COSTANT DEF
TMAX_a = 0
TMAX_s = 0

for c in canoni_distinct:
        TMAX_a = np.maximum(TMAX_a,max(contratto_dict[str(c)]))
        TMAX_s = np.maximum(TMAX_s,max(sfitto_dict[str(c)]))

print(TMAX_a)
print(TMAX_s)

T_GLOBAL = TMAX_a +TMAX_s


#%%RUN
from RentScenario.RentBound1Learner import RentBound1LearnerAdaptive
from RentScenario.RentBound1Learner import RentPersistentSingleExpl


n_runs = 50
experiment_name = "affitti-def"
T_HORIZON = 50000
for run in range(n_runs):
        learners = []
        rent_env = RentEnv(tmax_a=TMAX_a,tmax_s=TMAX_s)
        arms = get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict = avg_sfitti_dict,avg_contratti_dict =avg_contratti_dict)
        for a in arms:
                print("canone:"+str(a.canone)+" x/x+s:"+str(a.k_a_s)+" value: "+str(a.value))
        oracle = get_oracle(canoni_distinct,T_GLOBAL,avg_sfitti_dict = avg_sfitti_dict,avg_contratti_dict =avg_contratti_dict)
        print("ORACLE VALUE:"+str(oracle.value))
        learners.append(RentUCBLearner_single_m(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict = avg_sfitti_dict,avg_contratti_dict =avg_contratti_dict),T_GLOBAL))
        learners.append(RentUCBLearner_single_f(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict = avg_sfitti_dict,avg_contratti_dict =avg_contratti_dict),T_GLOBAL))

        #learners.append(Idea2_rent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL))
        learners.append(ThompsonBaselineRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL))        
        learners.append(ThompsonBaselineRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=True))


        

        

        

        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=False,param=1))
        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=True,param=1))

        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=False,param=2))
        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=True,param=2))

        learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL))
        learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=True))

        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=False,param=4))
        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=True,param=4))

        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=False,param=5))
        #learners.append(BayesUCBPersistentRent(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,farsighted=True,param=5))
        

        #learners.append(RentUCBLearner_double(len(arms),get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),T_GLOBAL,TMAX_a,TMAX_s))
        #learners.append(RentPersistentSingleExpl(n_arms=len(arms),arms=get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),tmax_a=TMAX_a,tmax_s=TMAX_s,t_global=T_GLOBAL))
        #learners.append(RentBound1LearnerAdaptive(n_arms=len(arms),arms=get_arm_list(canoni_distinct,T_GLOBAL,avg_sfitti_dict,avg_contratti_dict),tmax_a=TMAX_a,tmax_s=TMAX_s,t_global=T_GLOBAL))

        for i in tqdm(range(T_HORIZON)):    
                for learner in learners:
                        arm = learner.pull_arm()
                        bucket = rent_env.round(arm,contratto_dict,sfitto_dict)
                        learner.update_observations(arm,bucket)

                       
                        
                       
                
             
                        

        
        #SAVE CSV
        save_run_results(learners=learners ,run_id=run,exp_name=experiment_name)




