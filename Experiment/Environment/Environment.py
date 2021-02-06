import numpy as np
from Environment.Bucket import Bucket

class Environment():
    """
    class used to represent the Environment
    ....
    
    Attributes
    ----------
    tmax: int
        lenght of a bucket
    n_arms: int
        number of arms of the environment

    Methods
    -------
    round(self, pulled_arm):
        return the bucket for the current round
    
    generate_bucket(self,a,b):
        generate a bucket sampling from a beta distribution of parameters a and b
    
 
    """
    def __init__(self, n_arms,tmax, fixed_tmax = -1):
        self.fixed_tmax = fixed_tmax
        self.tmax=tmax
        self.n_arms = n_arms
        if fixed_tmax == -1:
            self.fixed_tmax = tmax



    def round(self, pulled_arm):
        bucket = self.generate_bucket(pulled_arm.alpha,pulled_arm.beta)        
        return bucket

    
    def generate_bucket(self,a,b):
        result = np.random.beta(a,b)
        ones = np.ones(round(result*self.fixed_tmax))
        zeros = np.zeros(self.tmax-len(ones))
        bucket = Bucket(np.concatenate((ones,zeros)))
        return bucket

    