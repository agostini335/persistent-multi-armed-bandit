import sys
sys.path.append("Experiment")
from Learners.Learner import Learner
import numpy as np

class Idea2Complete(Learner):
    def __init__(self, n_arms, arms, tmax):
        super().__init__(n_arms,arms,"Idea2_zeros_c")
        self.number_of_pulls = np.zeros(n_arms)
        self.criterion = np.zeros(n_arms)
        self.old_buckets_sum = np.zeros(n_arms)
        self.active_buckets_sum = np.zeros(n_arms)
        self.tmax=tmax


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]        
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    

    def update_observations(self, pulled_arm, bucket):
        #print("----------------------------------")
        #print(self.number_of_pulls)
        #print(self.criterion)
        #print(self.old_buckets_sum)
        #print(self.active_buckets_sum)
        super().update_observations(pulled_arm, bucket)
        

        for arm in range(self.n_arms):

            assert( arm == self.arms[arm].id_code)
            active_buckets = []
            self.active_buckets_sum[arm] = 0

            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                if m < self.tmax:
                    active_buckets.append(b)
                    self.active_buckets_sum[arm] = self.active_buckets_sum[arm] + sum(b.values[0:m+1])
                else:
                    assert(m == self.tmax)
                    self.number_of_pulls[pulled_arm.id_code] = self.number_of_pulls[pulled_arm.id_code] + 1
                    self.old_buckets_sum[arm] = self.old_buckets_sum[arm] + sum(b.values)
            self.arms[arm].buckets = active_buckets            

            if(self.number_of_pulls[arm]>0):
                first_term = np.sqrt((2*self.tmax*np.log(self.t))/ self.number_of_pulls[arm])
                second_term = (self.tmax*(self.tmax-1))/(2*self.number_of_pulls[arm])
                c = first_term + second_term
            else:
                c = np.inf 
            self.criterion[arm] = self.arms[arm].reward*((self.old_buckets_sum[arm] + self.active_buckets_sum[arm])/self.number_of_pulls[arm] + c)

                

class Idea2PositiveComplete(Learner):
    def __init__(self, n_arms, arms, tmax):
        super().__init__(n_arms,arms,"Idea2_ones_c")
        self.number_of_pulls = np.zeros(n_arms)
        self.criterion = np.zeros(n_arms)
        self.old_buckets_sum = np.zeros(n_arms)
        self.active_buckets_sum = np.zeros(n_arms)
        self.tmax=tmax


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    

    def update_observations(self, pulled_arm, bucket):
        #print("----------------------------------")
        #print(self.number_of_pulls)
        #print(self.criterion)
        #print(self.old_buckets_sum)
        #print(self.active_buckets_sum)
        super().update_observations(pulled_arm, bucket)
        

        for arm in range(self.n_arms):

            assert( arm == self.arms[arm].id_code)
            active_buckets = []
            self.active_buckets_sum[arm] = 0

            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                if m < self.tmax:
                    active_buckets.append(b)
                    self.active_buckets_sum[arm] = self.active_buckets_sum[arm] + sum(b.values[0:m+1]) + (self.tmax-(m+1))
                else:
                    assert(m == self.tmax)
                    self.number_of_pulls[pulled_arm.id_code] = self.number_of_pulls[pulled_arm.id_code] + 1
                    self.old_buckets_sum[arm] = self.old_buckets_sum[arm] + sum(b.values)
            self.arms[arm].buckets = active_buckets            

            if(self.number_of_pulls[arm]>0):
                first_term = np.sqrt((2*self.tmax*np.log(self.t))/ self.number_of_pulls[arm])
                second_term = (self.tmax*(self.tmax-1))/(2*self.number_of_pulls[arm])
                c = first_term + second_term
            else:
                c = np.inf 
            self.criterion[arm] = self.arms[arm].reward*((self.old_buckets_sum[arm] + self.active_buckets_sum[arm])/self.number_of_pulls[arm] + c)
    

class Idea2Spotify(Learner):
    def __init__(self, n_arms, arms, tmax):
        super().__init__(n_arms,arms,"Idea2_zeros")
        self.number_of_pulls = np.zeros(n_arms)
        self.criterion = np.zeros(n_arms)
        self.old_buckets_sum = np.zeros(n_arms)
        self.active_buckets_sum = np.zeros(n_arms)
        self.tmax=tmax

        #assign_numeric_id
        for id_arm in range(n_arms):
            self.arms[id_arm].numeric_id = id_arm        


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]        
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    

    def update_observations(self, pulled_arm, bucket):
        #print("----------------------------------")
        #print(self.number_of_pulls)
        #print(self.criterion)
        #print(self.old_buckets_sum)
        #print(self.active_buckets_sum)
        super().update_observations(pulled_arm, bucket)
        self.number_of_pulls[pulled_arm.numeric_id] = self.number_of_pulls[pulled_arm.numeric_id] + 1

        for arm in range(self.n_arms):

            assert( arm == self.arms[arm].numeric_id)
            active_buckets = []
            self.active_buckets_sum[arm] = 0

            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                if m < self.tmax:
                    active_buckets.append(b)
                    self.active_buckets_sum[arm] = self.active_buckets_sum[arm] + sum(b.values[0:m+1])
                else:
                    assert(m == self.tmax)
                    self.old_buckets_sum[arm] = self.old_buckets_sum[arm] + sum(b.values)
            self.arms[arm].buckets = active_buckets            

            if(self.number_of_pulls[arm]>0):
                first_term = np.sqrt((2*self.tmax*np.log(self.t))/ self.number_of_pulls[arm])
                second_term = (self.tmax*(self.tmax-1))/(2*self.number_of_pulls[arm])
                c = first_term + second_term
            else:
                c = np.inf 
            self.criterion[arm] = self.arms[arm].reward*((self.old_buckets_sum[arm] + self.active_buckets_sum[arm])/self.number_of_pulls[arm] + c)

                

class Idea2PositiveSpotify(Learner):
    def __init__(self, n_arms, arms, tmax):
        super().__init__(n_arms,arms,"Idea2_ones")
        self.number_of_pulls = np.zeros(n_arms)
        self.criterion = np.zeros(n_arms)
        self.old_buckets_sum = np.zeros(n_arms)
        self.active_buckets_sum = np.zeros(n_arms)
        self.tmax=tmax

        #assign_numeric_id
        for id_arm in range(n_arms):
            self.arms[id_arm].numeric_id = id_arm        



    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    

    def update_observations(self, pulled_arm, bucket):
        #print("----------------------------------")
        #print(self.number_of_pulls)
        #print(self.criterion)
        #print(self.old_buckets_sum)
        #print(self.active_buckets_sum)
        super().update_observations(pulled_arm, bucket)
        self.number_of_pulls[pulled_arm.numeric_id] = self.number_of_pulls[pulled_arm.numeric_id] + 1

        for arm in range(self.n_arms):

            assert( arm == self.arms[arm].numeric_id)
            active_buckets = []
            self.active_buckets_sum[arm] = 0

            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                if m < self.tmax:
                    active_buckets.append(b)
                    self.active_buckets_sum[arm] = self.active_buckets_sum[arm] + sum(b.values[0:m+1]) + (self.tmax-(m+1))
                else:
                    assert(m == self.tmax)
                    self.old_buckets_sum[arm] = self.old_buckets_sum[arm] + sum(b.values)
            self.arms[arm].buckets = active_buckets            

            if(self.number_of_pulls[arm]>0):
                first_term = np.sqrt((2*self.tmax*np.log(self.t))/ self.number_of_pulls[arm])
                second_term = (self.tmax*(self.tmax-1))/(2*self.number_of_pulls[arm])
                c = first_term + second_term
            else:
                c = np.inf 
            self.criterion[arm] = self.arms[arm].reward*((self.old_buckets_sum[arm] + self.active_buckets_sum[arm])/self.number_of_pulls[arm] + c)