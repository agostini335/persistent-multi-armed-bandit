from Learners.Learner import Learner
import numpy as np

class BaselineLearner_farsighted_adaptive(Learner):


    def __init__(self, n_arms, arms, tmax, tmin):
        super().__init__(n_arms,arms,"Baseline_farsighted_adp")
        self.tmax=tmax
        self.criterion = np.ones(n_arms) * np.inf
        self.tmin = tmin
        self.tmax_seen = 0
        
        self.total_reward_old = np.zeros(n_arms)
        self.number_of_completed_buckets_old = np.zeros(n_arms)


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        idxs = np.argwhere(self.criterion == self.criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    

    def update_observations(self, pulled_arm, bucket):

        super().update_observations(pulled_arm, bucket)

        #tmax_seen update
        for arm in range(self.n_arms):
            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                if(m< self.tmax and b.values[m]>0):
                    self.tmax_seen = max(self.tmax_seen,m+1)
                    break
       
        for arm in range(self.n_arms):
            #check correct list of arms
            assert(self.arms[arm].id_code == arm)
            completed = 0
            total_reward = 0
            active_buckets = []

            for b in self.arms[arm].buckets:
                # m is the relative time of a bucket
                m = int(self.t - b.t_start)
                # if m == tmax : the bucket is ended
                if m == self.tmax-1:
                    #bucket concluso definitivamente
                    self.number_of_completed_buckets_old[arm] +=1
                    new_observed_reward = np.sum(b.values)*self.arms[arm].reward                    
                    self.total_reward_old[arm] +=new_observed_reward
                else:
                    if m<self.tmax-1:
                        active_buckets.append(b)
                    if m == self.tmax_seen-1 or b.values[m]==0 :
                        completed += 1
                        total_reward += np.sum(b.values[0:self.tmax_seen])*self.arms[arm].reward
            
            self.arms[arm].buckets=active_buckets
           
           
            #criterion update
            if(completed > 0 or self.number_of_completed_buckets_old[arm]>0):
                support = (self.tmax_seen*self.arms[arm].reward - self.tmin*self.arms[arm].reward)
                self.criterion[arm] = (total_reward+self.total_reward_old[arm])/(completed+self.number_of_completed_buckets_old[arm]) + support*np.sqrt(2* np.log(self.t) / (completed+self.number_of_completed_buckets_old[arm]))
