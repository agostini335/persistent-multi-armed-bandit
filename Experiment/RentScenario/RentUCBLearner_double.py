from RentLearner import RentLearner
import numpy as np


class RentUCBLearner_double(RentLearner):
    def __init__(self, n_arms, arms, t_global, t_max_contratto, t_max_sfitto):
        super().__init__(n_arms,arms,"RentUCBLearner_double")
        self.criterion = np.ones(n_arms) * np.inf
        self.expected_contract_payoffs = np.zeros(n_arms)
        self.expected_vacant_payoffs = np.zeros(n_arms)
        self.number_of_completed_buckets = np.zeros(n_arms)
        self.t_global = t_global
        self.t_max_contratto = t_max_contratto
        self.t_max_sfitto = t_max_sfitto


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

                    new_observed_reward_contract = b.ones_contratto

                    new_observed_reward_vacant = b.zeros_sfitto

                    self.expected_contract_payoffs[arm] = (self.expected_contract_payoffs[arm] * (self.number_of_completed_buckets[arm]
                        - 1.0) + new_observed_reward_contract) / self.number_of_completed_buckets[arm]

                    self.expected_vacant_payoffs[arm] = (self.expected_vacant_payoffs[arm] * (self.number_of_completed_buckets[arm]
                        - 1.0) + new_observed_reward_vacant) / self.number_of_completed_buckets[arm]
                else:                
                    active_buckets.append(b)                    
    
            self.arms[arm].buckets = active_buckets
            
            #criterion update
            if(self.number_of_completed_buckets[arm]>0):

                exploration_term_c = self.t_max_contratto * np.sqrt((2* np.log(self.t)) / self.number_of_completed_buckets[arm])

                exploration_term_s = self.t_max_sfitto * np.sqrt((2* np.log(self.t)) / self.number_of_completed_buckets[arm])

                
                
                num = (self.expected_contract_payoffs[arm] + exploration_term_c)

                den = max (1,self.expected_vacant_payoffs[arm] - exploration_term_s + self.expected_contract_payoffs[arm] -exploration_term_c)
                
                self.criterion[arm] = self.arms[arm].canone * min(1, num / den)