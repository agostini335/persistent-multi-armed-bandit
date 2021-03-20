from RentLearner import RentLearner
import numpy as np
import random
from scipy.stats import beta as scipybeta
class ThompsonBaselineRent(RentLearner):
 
    def __init__(self, n_arms, arms, tmax, optimistic = False, farsighted = False):
        name = "Thompson_baseline_rent"                
        if optimistic == True:
            name += "_optim"
        if farsighted == True:
            name += "_farsighted"
        else:
            name += "_myopic"

        super().__init__(n_arms,arms,name)
      
        self.alpha_v = np.ones(n_arms)
        self.beta_v = np.ones(n_arms)
        self.tmax = tmax
        self.arms_r_vector = np.zeros(n_arms)        
        self.optimistic = optimistic
        self.farsighted = farsighted
        for id_arm in range(n_arms):      
            self.arms_r_vector[id_arm] = self.arms[id_arm].canone

    def optimistic_function(self,t,avg):
            if t > avg:
                return t
            else:
                return avg
    
    def get_optimistic_theta(self,theta,A,B,alpha_init,beta_init):
        avg = (A+alpha_init)/(A+alpha_init+B+beta_init)        
        new_theta = self.optimistic_function(theta,avg)
        return new_theta
       

    def pull_arm(self):
        #myopic static
        # draw THETAmi according to the Beta             
        THETA = np.random.beta(self.alpha_v , self.beta_v)
        if(self.optimistic):
            print("not implemented")          
        # chose armi
        pulled_arm_id = np.argmax(THETA * self.arms_r_vector)
        assert(self.arms[pulled_arm_id].id_code == pulled_arm_id)            
        return self.arms[pulled_arm_id]        

    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)
        if( self.farsighted == False):
            #myopic
            for arm in range(self.n_arms):
                active_buckets = []
                for b in self.arms[arm].buckets:
                    m = int(self.t - b.t_start)
                    if m < self.tmax:
                        active_buckets.append(b)
                    else:
                        value = sum(b.values)/(b.ones_contratto+b.zeros_sfitto)
                        assert(value>=0)
                        assert(value<=1)
                        if(random.random() <= value):
                            self.alpha_v[arm] = self.alpha_v[arm]+1
                        else:
                            self.beta_v[arm] = self.beta_v[arm]+1
                self.arms[arm].buckets = active_buckets
        else:
            #farsighted
            for arm in range(self.n_arms):
                active_buckets = []
                for b in self.arms[arm].buckets:
                    m = int(self.t - b.t_start)
                    if m < b.zeros_sfitto + b.ones_contratto:
                        active_buckets.append(b)
                    else:
                        value = sum(b.values)/(b.ones_contratto+b.zeros_sfitto)
                        assert(value>=0)
                        assert(value<=1)
                        if(random.random() <= value):
                            self.alpha_v[arm] = self.alpha_v[arm]+1
                        else:
                            self.beta_v[arm] = self.beta_v[arm]+1
                self.arms[arm].buckets = active_buckets


class BayesUCBPersistentRent(RentLearner):
 
    def __init__(self, n_arms, arms, tmax, farsighted=False, param = 3):
        name = "BayesUCBPersistent_"
        name += str(param) 
        if farsighted:
            name += "_farsighted"
        else:
            name += "_myopic"  
        super().__init__(n_arms,arms,name)
      
        self.hits_table = np.zeros((tmax,n_arms))
        self.sfitto_table = np.zeros((tmax,n_arms))
        self.visits_table = np.zeros((tmax,n_arms))
        self.init_prior_alpha = np.ones((tmax,n_arms))
        self.init_prior_beta = np.ones((tmax,n_arms))
        self.tmax = tmax
        self.arms_r_vector = np.zeros(n_arms)
        self.farsighted = farsighted
        self.param = param       
               
        #assign_numeric_id
        for id_arm in range(n_arms):      
            self.arms_r_vector[id_arm] = self.arms[id_arm].canone
                   
        

    def pull_arm(self):
          
        S_a = self.hits_table
        F_a = self.visits_table - self.hits_table
        S_s = self.sfitto_table
        F_s = self.visits_table - self.sfitto_table

        THETA_RESULTS = 1
        if(self.t>0):
            THETA_a_num = scipybeta.ppf((1-(1/(self.param*self.t))),S_a + self.init_prior_alpha, F_a + self.init_prior_beta)
            THETA_s_den = scipybeta.ppf(((1/(self.param*self.t))),S_s + self.init_prior_alpha, F_s + self.init_prior_beta)
            THETA_a_den = scipybeta.ppf(((1/(self.param*self.t))),S_s + self.init_prior_alpha, F_s + self.init_prior_beta)
      
            THETA_a_num = THETA_a_num.sum(axis = 0)
            THETA_s_den = THETA_s_den.sum(axis = 0)
            THETA_a_den = THETA_a_den.sum(axis = 0)


            THETA_RESULTS = THETA_a_num /(THETA_s_den + THETA_s_den)
    
        # chose armi

        THETA_RESULTS = THETA_RESULTS*self.arms_r_vector
        idxs = np.argwhere(THETA_RESULTS.max() == THETA_RESULTS).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        assert(self.arms[pulled_arm_id].id_code == pulled_arm_id)  
        #print(pulled_arm_id)          
        return self.arms[pulled_arm_id]

    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)

        if self.farsighted == False:
        
            for arm in range(self.n_arms):
                active_buckets = []
                for b in self.arms[arm].buckets:
                    # m is the relative time of a bucket
                    m = int(self.t - b.t_start)
                    # if m == tmax : the bucket is ended
                    if m < self.tmax:
                        active_buckets.append(b)
                        #update visite
                        self.visits_table[m][arm] = self.visits_table[m][arm]+1
                        #update dei successi
                        if b.values[m] == 0 and b.was_active== False:
                            self.sfitto_table[m][arm] = self.sfitto_table[m][arm]+1

                        if b.values[m]>0:
                                b.was_active = True
                                self.hits_table[m][arm] = self.hits_table[m][arm]+1

                    self.arms[arm].buckets = active_buckets
        
        else:

            for arm in range(self.n_arms):
                active_buckets = []
                for b in self.arms[arm].buckets:
                    # m is the relative time of a bucket
                    m = int(self.t - b.t_start)
                    # if m == tmax : the bucket is ended
                    if m < b.zeros_sfitto + b.ones_contratto:
                        active_buckets.append(b)
                        #update visite
                        self.visits_table[m][arm] = self.visits_table[m][arm]+1
                        #update dei successi
                        if b.values[m] == 0 and b.was_active== False:
                            self.sfitto_table[m][arm] = self.sfitto_table[m][arm]+1

                        if b.values[m]>0:
                                b.was_active = True
                                self.hits_table[m][arm] = self.hits_table[m][arm]+1
                    else:
                        for x in range(m,self.tmax):
                            self.visits_table[x][arm] = self.visits_table[x][arm]+1


                    self.arms[arm].buckets = active_buckets
        
       
