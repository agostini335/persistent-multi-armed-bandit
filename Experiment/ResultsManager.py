import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Parser.ArmSet import ArmSet
from Parser.ExperimentDescription import ExperimentDescription
import os
import json

class ResultsManager():

    """
    class used to handle the results of an experiment
    ....
    
    Attributes
    ----------
    

    Methods
    -------
    calculate_cumulative_pseudo_regret(collected_arm_values,oracle_arm_value):
        return the cumulative pseudo regret
   
    """
    def __init__(self):
        pass

    def calculate_cumulative_pseudo_regret(self,collected_arm_values,oracle_arm_value):
        istant_regret = oracle_arm_value - collected_arm_values
        cumulative_regret = np.zeros(len(collected_arm_values))
        cumulative_regret[0]=istant_regret[0]
        for i in range(1,len(collected_arm_values)):
            cumulative_regret[i]=cumulative_regret[i-1]+istant_regret[i]
        return cumulative_regret
    
    def plot_cumulative_pseudo_regret(self,collected_arm_values,oracle_arm_value):
        cml_rgrt= self.calculate_cumulative_pseudo_regret(collected_arm_values,oracle_arm_value)
        x1 = np.arange(len(cml_rgrt))
        plt.ylabel('cumulative pseudo regret')
        plt.title('cumulative pseudo regret')
        plt.plot(x1,cml_rgrt, label = "comulative pseudo regret")
        plt.xlabel('Time t')
        plt.show()
    
    def compare_pseudo_regret(self,b1_m,b1_f,bl_m,bl_f,oracle_arm_value):
        b1_m_rgrt = self.calculate_cumulative_pseudo_regret(b1_m,oracle_arm_value)
        b1_f_rgrt = self.calculate_cumulative_pseudo_regret(b1_f,oracle_arm_value)
        bl_m_rgrt = self.calculate_cumulative_pseudo_regret(bl_m,oracle_arm_value)
        bl_f_rgrt = self.calculate_cumulative_pseudo_regret(bl_f,oracle_arm_value)
        x1 = np.arange(len(b1_m_rgrt))


        plt.plot(b1_m_rgrt, label="bound1_myopic")
        plt.plot(b1_f_rgrt, label="bound1_farsighted")
        plt.plot(bl_m_rgrt, label="baseline_myopic")
        plt.plot(bl_f_rgrt, label="baseline_farsighted")

        plt.ylabel('cumulative pseudo regret')
        plt.title('cumulative pseudo regret')
        plt.legend(loc='lower left')
        #plt.plot(x1,cml_rgrt, label = "comulative pseudo regret")
        plt.xlabel('Time t')
        plt.show()


    def create_result_folder(self):
        #CREO LA CARTELLA
        self.dirName = "Experiment/Results/"+self.config_manager.name.split(".")[0]
         # Create target Directory if don't exist
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)
            print("Directory " , self.dirName ,  " Created ")
        else:    
            assert(False) #mi blocco perchè significa che sto sovrascrivendo

    def save_experiment_description(self):
        #converting list of arms in arm set extended
        cnfg_m = self.config_manager
        arms = cnfg_m.get_arm_list()
        arm_set_list = []
        # creo una lista estesa di arm rispetto al file di configurazione
        # in questo modo ho tutti gli id corretti
        for arm in arms:
            a_set =ArmSet(arm.alpha,arm.beta,arm.reward,quantity = 1,increment = 0, id_code = arm.id_code,arm_value = arm.value)
            arm_set_list.append(a_set)
        experiment_desc = ExperimentDescription(cnfg_m.get_tmax(),cnfg_m.get_time(),arm_set_list,cnfg_m.get_n_runs())

        #saving
        name = str(self.dirName+"/EXT"+cnfg_m.name)
        with open(name, 'w', encoding='utf-8') as f:
            json.dump(experiment_desc,   f, default=lambda o: o.__dict__,ensure_ascii=False, indent=4)

    def save_run_results(self,learners,run_id):
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
        name = self.dirName+"/id_results_run_"+str(run_id)+".csv"
        df.to_csv(name, index= True)

        #faccio lo stesso ma con gli arm values
        results = []
        names = []
        for l in learners:
            results.append(l.collected_arm_values)
            names.append(l.name)
        #creo le tuple per pandas
        results_tuples_list = list(zip(*results))
        # Converting lists of tuples into 
        # pandas Dataframe. 
        df = pd.DataFrame(results_tuples_list, columns = names)
        name = self.dirName+"/arm_value_results_run_"+str(run_id)+".csv"
        df.to_csv(name, index= True)
    
    

  

 






