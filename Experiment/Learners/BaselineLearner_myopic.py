from Learners.Learner import Learner
import numpy as np

class BaselineLearner_myopic (Learner):
    """
    class used to represent the Baseline Learner myopic
    ....
    
    Attributes
    ----------
    
    criterion: double((n_arms))
        criterion of each arm

    tmax: int
        lenght of a bucket

    Methods
    -------
    pull_arm(self):
        -init phase
        -return the next arm to pull according to the criterion
    update_observations(self, pulled_arm, bucket):
        -update at the end of a bucket
        -criterion update 
    """

    def __init__(self, n_arms, arms, tmax, tmin):
        super().__init__(n_arms,arms,"Baseline_myopic")
        self.criterion = np.ones(n_arms) * np.inf
        self.expected_payoffs = np.zeros(n_arms)
        self.number_of_completed_buckets = np.zeros(n_arms)
        self.tmax = tmax
        self.tmin = tmin


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)

        for arm in range(self.n_arms):
            active_buckets = []
            for b in self.arms[arm].buckets:
                # m is the relative time of a bucket
                m = int(self.t - b.t_start)
                if m  == self.tmax-1 :
                    #expected payoff update
                    self.number_of_completed_buckets[arm] = self.number_of_completed_buckets[arm] + 1
                    
                    new_observed_reward = np.sum(b.values)*self.arms[arm].reward

                    self.expected_payoffs[arm] = (self.expected_payoffs[arm] * (self.number_of_completed_buckets[arm]
                        - 1.0) + new_observed_reward) / self.number_of_completed_buckets[arm]
                else:                
                    active_buckets.append(b)                    
    
            self.arms[arm].buckets = active_buckets
            
            #criterion update
            if(self.number_of_completed_buckets[arm]>0):
                support = (self.tmax*self.arms[arm].reward - self.tmin*self.arms[arm].reward)
                self.criterion[arm] = self.expected_payoffs[arm] + support*np.sqrt(2* np.log(self.t) / self.number_of_completed_buckets[arm])


        

        