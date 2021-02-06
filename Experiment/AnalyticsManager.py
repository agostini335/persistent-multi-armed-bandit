import sys
sys.path.append("Experiment")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Environment.Arm import Arm
from Parser.ArmSet import ArmSet
from Parser.ExperimentDescription import ExperimentDescription
import os
import json
from tqdm import tqdm
from Learners.Oracle import Oracle

class AnalyticsManager():

    """
    class used to produce the Analytics of an experiment
    ....
    
    Attributes
    ----------
    

    Methods
    -------
   
    """
    def __init__(self,expeirment_path,expeirment_name,fixed_tmax=-1,fixed = False):
        self.path = expeirment_path
        self.name = expeirment_name
        self.fixed = fixed
        self.fixed_tmax = fixed_tmax
        self.figure_counter = 0
    
    def load_experiment_description(self):
         #load json
        self.config_file_name=str(self.path +"/"+ self.name + "/"+"EXT"+self.name+".json")
        with open(self.config_file_name, 'r', encoding='utf-8') as f:
            d = json.load(f)        
        self.decoded_e = ExperimentDescription.from_json(d)
        print("loaded description: "+str(self.name))
        print("n_runs : " + str(self.decoded_e.n_runs))

        #build arms of the experiment
        self.arm_list = [] 
        for armSet in self.decoded_e.arm_sets:
            reward = armSet['starting_reward']
            self.arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,self.decoded_e.tmax,armSet['id_code']))
        
        #create oracle
        o_arm_list = [] 
        for armSet in self.decoded_e.arm_sets:
            reward = armSet['starting_reward']
            o_arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,self.decoded_e.tmax,armSet['id_code']))
        self.oracle = Oracle(len(self.arm_list),o_arm_list,self.decoded_e.tmax)


    
    def load_experiment_data(self):
        #extract learners
        data = pd.read_csv(self.path +"/"+ self.name + "/"+ "results_run_0" +".csv")
        self.learners = data.columns.values.tolist()[1:] #la prima colonna che salvo è quella del tempo quindi la rimuovo
        print(self.learners)

        #dict learner-> list of list of played arms
        self.learner_data_dict = {}
        self.learner_arm_value_dict = {}
      
        for l in range(len(self.learners)):
            results_list_of_l = [] #it must have n_runs element
            for i in range(0,self.decoded_e.n_runs):
                full_name = self.path +"/"+ self.name + "/"+ "results_run_" + str(i) +".csv"
                data = pd.read_csv(full_name)
                arm_played_by_l_on_run_i = data[self.learners[l]]
                assert(len(arm_played_by_l_on_run_i) == self.decoded_e.time)

                results_list_of_l.append(arm_played_by_l_on_run_i)
            
            assert(len(results_list_of_l) == self.decoded_e.n_runs)       
            self.learner_data_dict[self.learners[l]] = results_list_of_l
        
        #change id_code with the value of the arm
        "loading arm value of each play:"
        for learner in tqdm(self.learners) :
            learner_arm_values = []
            for run_data in self.learner_data_dict[learner]:
                current_run_arm_values =[]
                for i in range(self.decoded_e.time):
                    #from id_code to arm value
                    current_run_arm_values.append([arm for arm in self.arm_list if arm.id_code == run_data[i]][0].value)                
                    assert[len([arm for arm in self.arm_list if arm.id_code == run_data[i]])==1]
                learner_arm_values.append(current_run_arm_values)
            self.learner_arm_value_dict[learner] = learner_arm_values
        
    
    
    def calculate_cumulative_pseudo_regret(self,collected_arm_values,oracle_arm_value):
        istant_regret = ( collected_arm_values - oracle_arm_value ) * (-1)
        cumulative_regret = np.zeros(len(collected_arm_values))
        cumulative_regret[0]=istant_regret[0]
        for i in range(1,len(collected_arm_values)):
            cumulative_regret[i]=cumulative_regret[i-1]+istant_regret[i]
        return cumulative_regret
           

    def load_experiment(self, with_arm_values = False):
        if not with_arm_values :
            self.load_experiment_description()
            self.load_experiment_data()
        else:
            self.load_experiment_description_with_arm_values()
            self.load_experiment_data_with_arm_values()
            

   


    def compute_experiment_pseudo_regret(self):
        self.learner_pseudo_regret_dict = {}
        self.learner_average_pseudo_regret = {}
        self.learner_std = {}
        "computing pseudo regret:"
        for learner in tqdm(self.learners) :
            pseudo_regret_of_learner = []
            for arm_values_run in self.learner_arm_value_dict[learner]:
                pseudo_regret_of_learner.append(self.calculate_cumulative_pseudo_regret(arm_values_run,self.oracle.best_arm.value))
            
            self.learner_pseudo_regret_dict[learner] = pseudo_regret_of_learner
            self.learner_average_pseudo_regret[learner] = np.mean(pseudo_regret_of_learner,axis = 0)
            self.learner_std[learner] = np.std(pseudo_regret_of_learner,axis = 0)

        #save results
        results = []
        names = []
        for l in self.learners:
            results.append(self.learner_average_pseudo_regret[l])
            names.append(l)
        #creo le tuple per pandas
        results_tuples_list = list(zip(*results))
        # Converting lists of tuples into 
        # pandas Dataframe. 
        df = pd.DataFrame(results_tuples_list, columns = names)
        name = self.path+'/'+self.name+"/cumulated_average_pseudo_regret_"+ str(self.name)+".csv"
        df.to_csv(name, index= True)

        #save results
        results = []
        names = []
        for l in self.learners:
            results.append(self.learner_std[l])
            names.append(l)
        #creo le tuple per pandas
        results_tuples_list = list(zip(*results))
        # Converting lists of tuples into 
        # pandas Dataframe. 
        df = pd.DataFrame(results_tuples_list, columns = names)
        name = self.path+'/'+self.name+"/std_pseudo_regret"+ str(self.name)+".csv"
        df.to_csv(name, index= True)


     

    
    
    def plot_cumulative_pseudo_regret(self):        
        for learner in self.learners:
            x = np.arange(len(self.learner_average_pseudo_regret[learner]))

            confidence_interval = 2*self.learner_std[learner]/np.sqrt(self.decoded_e.n_runs)
            plt.errorbar(x,self.learner_average_pseudo_regret[learner],confidence_interval,elinewidth=0.9,capsize=2,errorevery=7000 ,label = learner)

        plt.ylabel('cumulative pseudo regret')
        plt.title('cumulative pseudo regret '+ str(self.name))
        plt.legend(loc='upper left')
        plt.xlabel('Time t')
        plt.savefig(self.path+'/'+self.name+"/cumulative_pseudo_regret_"+ str(self.name)+".png",dpi=600)
        plt.clf()
    
    def plot_cumulative_pseudo_regret_of_a_run(self,run_id):
        for learner in self.learners:
            plt.plot(self.learner_pseudo_regret_dict[learner][run_id], label = learner)
        plt.ylabel('cumulative pseudo regret run:'+str(run_id))
        plt.title('cumulative pseudo regret')
        plt.legend(loc='lower left')
        plt.xlabel('Time t')        
        plt.show()
    
    def load_experiment_description_with_arm_values(self):
        if self.fixed == False:
            #load json
            self.config_file_name=str(self.path +"/"+ self.name + "/"+"EXT"+self.name+".json")
            with open(self.config_file_name, 'r', encoding='utf-8') as f:
                d = json.load(f)        
            self.decoded_e = ExperimentDescription.from_json(d)
            print("loaded description: "+str(self.name))
            print("n_runs : " + str(self.decoded_e.n_runs))

            #build arms of the experiment
            self.arm_list = [] 
            for armSet in self.decoded_e.arm_sets:
                reward = armSet['starting_reward']
                self.arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,self.decoded_e.tmax,armSet['id_code']))
            
            #create fake oracle
            oracle_value = 0
            for armSet in self.decoded_e.arm_sets:
                value= armSet['arm_value']
                if(value>oracle_value):
                    oracle_value=value
            self.oracle = Oracle(0,[],0)
            fake_best_arm = Arm(0,0,0,0,-1)
            fake_best_arm.value=oracle_value
            self.oracle.best_arm=fake_best_arm
        
        else:
            #load json
            self.config_file_name=str(self.path +"/"+ self.name + "/"+"EXT"+self.name+".json")
            with open(self.config_file_name, 'r', encoding='utf-8') as f:
                d = json.load(f)        
            self.decoded_e = ExperimentDescription.from_json(d)
            print("loaded description: "+str(self.name))
            print("n_runs : " + str(self.decoded_e.n_runs))

            #build arms of the experiment
            self.arm_list = [] 
            for armSet in self.decoded_e.arm_sets:
                reward = armSet['starting_reward']
                self.arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,self.fixed_tmax,armSet['id_code']))
            
            #create fake oracle
            oracle_value = 0
            for armSet in self.decoded_e.arm_sets:
                value= armSet['arm_value']
                if(value>oracle_value):
                    oracle_value=value
            self.oracle = Oracle(0,[],0)
            fake_best_arm = Arm(0,0,0,0,-1)
            fake_best_arm.value=oracle_value
            self.oracle.best_arm=fake_best_arm

        print(self.oracle.best_arm.value)
        
    
    def load_experiment_data_with_arm_values(self):
        #extract learners
        data = pd.read_csv(self.path +"/"+ self.name + "/"+ "arm_value_results_run_0" +".csv")
        self.learners = data.columns.values.tolist()[1:] #la prima colonna che salvo è quella del tempo quindi la rimuovo
        print(self.learners)

        #dict learner-> list of list of played arms
        self.learner_data_dict = {}
        self.learner_arm_value_dict = {}
      
        for l in range(len(self.learners)):
            results_list_of_l = [] #it must have n_runs element
            for i in range(0,self.decoded_e.n_runs):
                full_name = self.path +"/"+ self.name + "/"+ "arm_value_results_run_" + str(i) +".csv"
                data = pd.read_csv(full_name)
                arm_values_of_l_on_run_i = data[self.learners[l]]
                assert(len(arm_values_of_l_on_run_i) == self.decoded_e.time)

                results_list_of_l.append(arm_values_of_l_on_run_i)
            
            assert(len(results_list_of_l) == self.decoded_e.n_runs)       
            self.learner_arm_value_dict[self.learners[l]] = results_list_of_l
    
   

        
    







        
     



    


   

  

 



