from Learners.Learner import Learner
import numpy as np

class Baseline_Thompson_myopic (Learner):

    def __init__(self, n_arms, arms, tmax, tmin):
        super().__init__(n_arms,arms,"Baseline_Thompson_myopic")
        self.success_vector = np.zeros(n_arms)
        self.failure_vector = np.zeros(n_arms)
        self.alpha_vector = np.ones(n_arms)
        self.beta_vector = np.ones(n_arms)
        self.tmax = tmax
        self.tmin = tmin


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        #sampling
        theta = np.random.beta(self.success_vector + self.alpha_vector, self.failure_vector + self.beta_vector)
        pulled_arm_id = np.random.choice(np.argmax(theta))
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
