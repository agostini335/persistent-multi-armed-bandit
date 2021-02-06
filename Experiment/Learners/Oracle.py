import sys
sys.path.append("Experiment")
from Learners.Learner import Learner
import numpy as np

class Oracle(Learner):
    """
    class used to represent the oracle 
    ....
    
    Attributes
    ----------
    tmax: int
        lenght of a bucket
    best_arm: arm
        arm with the maximum value

   
    Methods
    -------
    pull_arm(self):
        -return best arm
    update_observations(self, pulled_arm, bucket):
        -tables update
        -criterion update 
 
    """

    def __init__(self, n_arms, arms, tmax):
        super().__init__(n_arms,arms,"Oracle")
        self.tmax=tmax
        max = 0
        for arm in arms:
            if(arm.value>=max):
                max=arm.value
                self.best_arm=arm
                
    def pull_arm(self):
        return self.best_arm
        
    def update_observations(self, pulled_arm, bucket):
        pass
        