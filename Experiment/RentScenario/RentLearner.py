import numpy as np

class RentLearner:
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
