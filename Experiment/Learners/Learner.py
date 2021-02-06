import numpy as np

class Learner:
    """
    Super class used to represent a generic Learner
    ....
    
    Attributes
    ----------
    n_arms : int
        number of arms
    t : int
        global time of the experiment
    collected_arm_values: double[]
        list of arm_value collected during the experiment (arm_value = reward*beta(a,b).mean())
    arms: arm[]
        list of arms of the learner
    name: string
        name of the learner, used to reference the learner in the export of the results
    played_arms_id = int[]
        list that stores sequentially the id_code of the arms pulled during the experiment

    

    Methods
    -------
    update_observations(self,pulled_arm, bucket) :
        -incrment the time of the expiriment
        -set the starting time of the bucket collected
        -update the buckets of the pulled_arm
        -update the collected_buckets of the learner
        -update the collected_arm_values of the learner
    
    pull_arm(self):
        -must be implemented in the subclass

    """
    def __init__(self,n_arms,arms,name):
        self.name=name
        self.n_arms = n_arms
        self.t = 0
        self.arms = arms
        self.collected_arm_values = []
        self.played_arms_id = []
    
    def pull_arm(self):
        pass

    def update_observations(self,pulled_arm, bucket):
        self.t += 1
        bucket.t_start=self.t
        pulled_arm.buckets.append(bucket)
        self.collected_arm_values.append(pulled_arm.value)
        self.played_arms_id.append(pulled_arm.id_code)

        