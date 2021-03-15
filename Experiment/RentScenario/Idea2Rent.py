from RentLearner import RentLearner
import numpy as np


class Idea2_rent(RentLearner):
    def __init__(self, n_arms, arms, t_global):
        super().__init__(n_arms,arms,"Idea2_rent")
        self.criterion = np.ones(n_arms) * np.inf
        self.expected_payoffs = np.zeros(n_arms)
        self.number_of_completed_buckets = np.zeros(n_arms)
        self.t_global = t_global
        self.old_buckets_sum = np.zeros(n_arms)
        self.active_buckets_sum = np.zeros(n_arms)
        self.number_of_visits = np.zeros(n_arms)
        self.number_of_pulls = np.zeros(n_arms)


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)
        self.number_of_pulls[pulled_arm.id_code] = self.number_of_pulls[pulled_arm.id_code] + 1
        

        for arm in range(self.n_arms):

            assert( arm == self.arms[arm].id_code)
            active_buckets = []
            self.active_buckets_sum[arm] = 0


            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                
                if m  < self.t_global :
                    #expected payoff update
                    active_buckets.append(b)

                    self.active_buckets_sum[arm] = self.active_buckets_sum[arm] + sum(b.values[0:m+1])
                    if b.values[m]>0:
                        b.was_active = True #TODO ottimizzare
        
                    if b.values[m] == 0 and b.was_active == False:
                        self.number_of_visits[arm] = self.number_of_visits[arm] + 1
                else:                
                    self.old_buckets_sum[arm] = self.old_buckets_sum[arm] + sum(b.values)          
            self.arms[arm].buckets = active_buckets


            if(self.number_of_pulls[arm]>0):
                first_term = np.sqrt((2*self.t_global*np.log(self.t))/ self.number_of_pulls[arm])
                second_term = (self.t_global*(self.t_global-1))/(2*self.number_of_pulls[arm])
                c = first_term + second_term
            else:
                c = np.inf 
            self.criterion[arm] = self.arms[arm].canone*((self.old_buckets_sum[arm] + self.active_buckets_sum[arm])/self.number_of_visits[arm] + c)
            
        