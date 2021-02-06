import numpy as np
from scipy.stats import beta as bt


class Arm():
    """
    A class used to represent an Arm
    ....
    
    Attributes
    ----------
    alpha : double
        alpha parameter of the beta-distribution associated to the arm 
    beta : double
        beta parameter of the beta-distribution associated to the arm
    reward: int
        istantaneous reward associated to the arm
    buckets: bucket[]
        list of collected buckets that are still active
    non_active_buckets: bucket[]
        list of non-active buckets that are no longer considered by the learner (not used in the current implementation)
    betamean: double 
        mean of the beta distribution associated to the arm
    value: double
        betamean*reward
    id: int
        identifier of the arm
    
    Methods
    -------
        
    """
    def __init__(self, alpha, beta, reward, tmax, id_code ):
        self.alpha = alpha
        self.beta = beta
        self.id_code = id_code
        self.reward = reward # R - reward istantaneo
        self.betamean = bt(alpha,beta).mean()
        self.value = self.compute_value(alpha,beta,reward,tmax) # expected reward
        self.buckets = []
        self.non_active_buckets = []
    
    def compute_value(self,a,b,reward,tmax):
        value = 0
        for i in range (1,tmax+1):
            value += i * (bt.cdf(i/tmax,a,b)-bt.cdf((i-1)/tmax,a,b))
        return value * reward




        
