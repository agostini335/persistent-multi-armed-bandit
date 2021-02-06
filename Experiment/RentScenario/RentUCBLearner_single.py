from RentLearner import RentLearner
import numpy as np


class RentUCBLearner_single(RentLearner):
    def __init__(self, n_arms, arms, t_global):
        super().__init__(n_arms,arms,"RentUCBLearner_single")
        self.criterion = np.ones(n_arms) * np.inf
        self.expected_payoffs = np.zeros(n_arms)
        self.number_of_completed_buckets = np.zeros(n_arms)
        self.t_global = t_global


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
                if m  == self.t_global -1 :
                    #expected payoff update
                    self.number_of_completed_buckets[arm] = self.number_of_completed_buckets[arm] + 1

                    new_observed_reward = b.ones_contratto/(b.zeros_sfitto+b.ones_contratto)

                    self.expected_payoffs[arm] = (self.expected_payoffs[arm] * (self.number_of_completed_buckets[arm]
                        - 1.0) + new_observed_reward) / self.number_of_completed_buckets[arm]
                else:                
                    active_buckets.append(b)                    
    
            self.arms[arm].buckets = active_buckets
            
            #criterion update
            if(self.number_of_completed_buckets[arm]>0):
                self.criterion[arm] = self.arms[arm].canone * (self.expected_payoffs[arm] + np.sqrt((2* np.log(self.t)) / self.number_of_completed_buckets[arm]))