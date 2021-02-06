import sys
sys.path.append("Experiment")
import json
from Parser.ExperimentDescription import ExperimentDescription
from Environment.Arm import Arm

class ConfigManager():
    
    def __init__(self,path, name):
        self.path = path
        self.name =  name
        self.config_file_name=str(path +"/"+ name)
        with open(self.config_file_name, 'r', encoding='utf-8') as f:
            d = json.load(f)        
        self.decoded_e = ExperimentDescription.from_json(d)
    
    def get_arm_list(self):
        arm_list = [] 
        current_arm_id = 0
        for armSet in self.decoded_e.arm_sets:
            quantity = armSet['quantity']
            reward = armSet['starting_reward']
            for i in range(0,quantity):
                arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,self.decoded_e.tmax,current_arm_id))
                reward += armSet['reward_incrment_step']
                current_arm_id +=1
        return arm_list
    
    def get_arm_list_incremented(self,arm_id,increment):
        arm_list = [] 
        current_arm_id = 0
        for armSet in self.decoded_e.arm_sets:
            quantity = armSet['quantity']
            reward = armSet['starting_reward']
            for i in range(0,quantity):
                if(current_arm_id==arm_id):
                    arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward+increment,self.decoded_e.tmax,current_arm_id))
                else:
                    arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,self.decoded_e.tmax,current_arm_id))
                reward += armSet['reward_incrment_step']
                current_arm_id +=1
        return arm_list



    
    def get_arm_list_fixed_tmax(self, fixed_tmax):
        arm_list = [] 
        current_arm_id = 0
        for armSet in self.decoded_e.arm_sets:
            quantity = armSet['quantity']
            reward = armSet['starting_reward']
            for i in range(0,quantity):
                arm_list.append(Arm(armSet['alpha'],armSet['beta'],reward,fixed_tmax,current_arm_id))
                reward += armSet['reward_incrment_step']
                current_arm_id +=1
        return arm_list
       
        

    
    def get_n_arms(self):
        return len(self.get_arm_list())
    
    def get_time(self):
        return int(self.decoded_e.time)
    
    def get_tmax(self):
        return int(self.decoded_e.tmax)
    
    def get_tmin(self):
        return 0
    
    def get_n_runs(self):
        return int(self.decoded_e.n_runs)
    
    


        
    
