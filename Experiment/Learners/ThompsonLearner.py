from Learners.Learner import Learner
import numpy as np
from scipy.stats import beta as scipybeta
import Utils
import random

class ThompsonLearner(Learner):
    """
    class used to represent the Thompson Learner
    ....
    
    Attributes
    ----------
    hits_table : int((tmax,n_arms))
        matrix representing the number of hits obtained by an arm i at a step j (relative time of a bucket)
        it must be updated according with the time of the experiment
    visits_table: int((tmax,n_arms))
        matrix representing the number of visits done by an arm i at a step j (relative time of a bucket)
        it must be updated according with the time of the experiment
    tmax: int
        lenght of a bucket

    Methods
    -------
    pull_arm(self):
        -init phase
        -return the next arm to pull
    update_observations(self, pulled_arm, bucket):
        -tables update
    """

    def __init__(self, n_arms, arms, tmax, farsighted, adaptive, monotonic, optimistic = False):
        name = "Thompson"        
        if farsighted == True:
            name += "_farsighted"
        else:
            name += "_myopic"        
        if adaptive == True:
            name += "_adp"
            self.tmax_seen = 0
        if monotonic == True:
            name += "_mono"
        if optimistic == True:
            name += "_optim"

        super().__init__(n_arms,arms,name)
        self.farsighted = farsighted
        self.adaptive = adaptive
        self.hits_table = np.zeros((tmax,n_arms))
        self.visits_table = np.zeros((tmax,n_arms))
        self.init_prior_alpha = np.ones((tmax,n_arms))
        self.init_prior_beta = np.ones((tmax,n_arms))
        self.tmax = tmax
        self.arms_r_vector = np.zeros(n_arms)
        self.monotonic = monotonic
        self.optimistic = optimistic
        self.vectorized_optimistic_function = np.vectorize(self.optimistic_function)
       
        #populating arms_r_vector
        for id_arm in range(n_arms):
            if ("play" not in str(self.arms[id_arm].id_code)):
                self.arms_r_vector[self.arms[id_arm].id_code] = self.arms[id_arm].reward
            else:
                #TODO sostituire
                pass



    def make_monotonic(self, x ):
        for i in range(0,self.tmax-1):
            if (x[i+1]>x[i]):
                x[i+1]=x[i]

    def get_monotonic_theta(self,theta):
        np.apply_along_axis( self.make_monotonic, axis=0, arr=theta )   

    

    def optimistic_function(self,t,avg):
            if t > avg:
                return t
            else:
                return avg
    
    def get_optimistic_theta(self,theta,S,F,alphas,betas):
        avg = (S+alphas)/(S+alphas+F+betas)        
        new_theta = self.vectorized_optimistic_function(theta,avg)
        return new_theta

        


    def update_tmax_seen(self):
        #tmax_seen update
        for arm in range(self.n_arms):
            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                assert(m<=self.tmax)
                if(m< self.tmax and b.values[m]>0):
                    self.tmax_seen = max(self.tmax_seen,m+1)
                    break



                   
        

    def pull_arm(self):
        if self.farsighted == True and self.adaptive == True:
            #farsighted adaptive
            self.update_tmax_seen()
            S = self.hits_table
            F = self.visits_table - self.hits_table
            THETA = np.random.beta(S + self.init_prior_alpha, F + self.init_prior_beta)
            if(self.monotonic):
                self.get_monotonic_theta(THETA)
            pulled_arm_id = np.argmax((THETA[:self.tmax_seen,:].sum(axis = 0)*self.arms_r_vector))
            '''

            print("SUCCESS:")
            print(S)
            print("FAIL:")
            print(F)
            print("VISIT:")
            print(self.visits_table)
            print("THETA:")
            print(THETA)
            print("THETA SUM:")
            print(THETA.sum(axis = 0))
            '''


            assert(self.arms[pulled_arm_id].id_code == pulled_arm_id)
            return self.arms[pulled_arm_id]

        if self.farsighted == True and self.adaptive == False:
            #farsighted static
            S = self.hits_table
            F = self.visits_table - self.hits_table
            THETA = np.random.beta(S + self.init_prior_alpha, F + self.init_prior_beta)
            if(self.monotonic):
                self.get_monotonic_theta(THETA)
            pulled_arm_id = np.argmax(THETA.sum(axis = 0)*self.arms_r_vector)
            assert(self.arms[pulled_arm_id].id_code == pulled_arm_id)            
            return self.arms[pulled_arm_id]

        if self.farsighted == False and self.adaptive == True:
            #myopic adaptive
            self.update_tmax_seen()
            S = self.hits_table
            F = self.visits_table - self.hits_table
            THETA = np.random.beta(S + self.init_prior_alpha, F + self.init_prior_beta)
            if(self.monotonic):
                self.get_monotonic_theta(THETA)           
            pulled_arm_id = np.argmax((THETA[:self.tmax_seen,:].sum(axis = 0)*self.arms_r_vector))
            assert(self.arms[pulled_arm_id].id_code == pulled_arm_id)
            return self.arms[pulled_arm_id]

        if self.farsighted == False and self.adaptive == False:
            #myopic static
            # draw THETAmi according to the Beta             
            S = self.hits_table
            F = self.visits_table - self.hits_table
            THETA = np.random.beta(S + self.init_prior_alpha, F + self.init_prior_beta)
            if(self.monotonic):
                self.get_monotonic_theta(THETA)
            if(self.optimistic):
                THETA = self.get_optimistic_theta(THETA,S,F,self.init_prior_alpha,self.init_prior_beta)

            '''
            print("SUCCESS:")
            print(S)
            print("FAIL:")
            print(F)
            print("VISIT:")
            print(self.visits_table)
            print("THETA:")
            print(THETA)
            print("THETA SUM:")
            print(THETA.sum(axis = 0))
            '''
            # chose armi
            pulled_arm_id = np.argmax(THETA.sum(axis = 0)*self.arms_r_vector)
            assert(self.arms[pulled_arm_id].id_code == pulled_arm_id)            
            return self.arms[pulled_arm_id]
        
        

    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)

        if self.farsighted == True and self.adaptive == True:
            #farsighted adaptive
            #check correct list of arms
            #tables update
            for arm in range(self.n_arms):
                #check correct list of arms
                assert(self.arms[arm].id_code == arm)
                active_buckets = []
                for b in self.arms[arm].buckets:
                    # m is the relative time of a bucket
                    m = int(self.t - b.t_start)
                    # if m == tmax : the bucket is ended
                    if m < self.tmax :
                        if b.values[m]>0:
                            #update visite
                            self.visits_table[m][arm] = self.visits_table[m][arm]+1
                            #update dei successi
                            
                            self.hits_table[m][arm] = self.hits_table[m][arm]+1
                            active_buckets.append(b)
                        else:
                            #full update visite
                            for k in range (m,self.tmax):
                                self.visits_table[k][arm] = self.visits_table[k][arm]+1        
                self.arms[arm].buckets = active_buckets
           

            
        if self.farsighted == True and self.adaptive == False:
            #farsighted static
            #check correct list of arms
            #tables update
            for arm in range(self.n_arms):
                #check correct list of arms
                assert(self.arms[arm].id_code == arm)
                active_buckets = []
                for b in self.arms[arm].buckets:
                    # m is the relative time of a bucket
                    m = int(self.t - b.t_start)
                    # if m == tmax : the bucket is ended
                    if m < self.tmax :
                        if b.values[m]>0:
                            #update visite
                            self.visits_table[m][arm] = self.visits_table[m][arm]+1
                            #update dei successi
                            
                            self.hits_table[m][arm] = self.hits_table[m][arm]+1
                            active_buckets.append(b)
                        else:
                            #full update visite
                            for k in range (m,self.tmax):
                                self.visits_table[k][arm] = self.visits_table[k][arm]+1        
                self.arms[arm].buckets = active_buckets
           
            
            

        if self.farsighted == False and self.adaptive == True:
            #myopic adaptive
            for arm in range(self.n_arms):
                #check correct list of arms
                assert(self.arms[arm].id_code == arm)
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
                        if b.values[m]>0:
                            self.hits_table[m][arm] = self.hits_table[m][arm]+1                       
                self.arms[arm].buckets = active_buckets
            
         
        if self.farsighted == False and self.adaptive == False:
            #myopic static
            #tables update
            for arm in range(self.n_arms):
                #check correct list of arms
                assert(self.arms[arm].id_code == arm)
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
                        if b.values[m]>0:
                            self.hits_table[m][arm] = self.hits_table[m][arm]+1
                self.arms[arm].buckets = active_buckets



class ThompsonLearnerSpotify(Learner):
 
    def __init__(self, n_arms, arms, tmax, optimistic = False):
        name = "Thompson"        
        name += "_myopic"        
        if optimistic == True:
            name += "_optim"

        super().__init__(n_arms,arms,name)
      
        self.hits_table = np.zeros((tmax,n_arms))
        self.visits_table = np.zeros((tmax,n_arms))
        self.init_prior_alpha = np.ones((tmax,n_arms))
        self.init_prior_beta = np.ones((tmax,n_arms))
        self.tmax = tmax
        self.arms_r_vector = np.zeros(n_arms)        
        self.optimistic = optimistic
        self.vectorized_optimistic_function = np.vectorize(self.optimistic_function)
       
        #assign_numeric_id
        for id_arm in range(n_arms):
            self.arms[id_arm].numeric_id = id_arm        
            self.arms_r_vector[id_arm] = self.arms[id_arm].reward


    def optimistic_function(self,t,avg):
            if t > avg:
                return t
            else:
                return avg
    
    def get_optimistic_theta(self,theta,S,F,alphas,betas):
        avg = (S+alphas)/(S+alphas+F+betas)        
        new_theta = self.vectorized_optimistic_function(theta,avg)
        return new_theta

        


    def update_tmax_seen(self):
        #tmax_seen update
        for arm in range(self.n_arms):
            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                assert(m<=self.tmax)
                if(m< self.tmax and b.values[m]>0):
                    self.tmax_seen = max(self.tmax_seen,m+1)
                    break
                   
        

    def pull_arm(self):
        #myopic static
        # draw THETAmi according to the Beta             
        S = self.hits_table
        F = self.visits_table - self.hits_table
        THETA = np.random.beta(S + self.init_prior_alpha, F + self.init_prior_beta)
        if(self.optimistic):
           THETA = self.get_optimistic_theta(THETA,S,F,self.init_prior_alpha,self.init_prior_beta)
        # chose armi
        pulled_arm_id = np.argmax(THETA.sum(axis = 0)*self.arms_r_vector)
        assert(self.arms[pulled_arm_id].numeric_id == pulled_arm_id)            
        return self.arms[pulled_arm_id]
        
        

    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)
        for arm in range(self.n_arms):
            #check correct list of arms
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
                    if b.values[m]>0:
                        self.hits_table[m][arm] = self.hits_table[m][arm]+1
            self.arms[arm].buckets = active_buckets

class ThompsonBaselineSpotify(Learner):
 
    def __init__(self, n_arms, arms, tmax, optimistic = False):
        name = "Thompson_baseline"                
        if optimistic == True:
            name += "_optim"

        super().__init__(n_arms,arms,name)
      
        self.alpha_v = np.ones(n_arms)
        self.beta_v = np.ones(n_arms)
        self.tmax = tmax
        self.arms_r_vector = np.zeros(n_arms)        
        self.optimistic = optimistic
       
        #assign_numeric_id
        for id_arm in range(n_arms):
            self.arms[id_arm].numeric_id = id_arm        
            self.arms_r_vector[id_arm] = self.arms[id_arm].reward


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
        assert(self.arms[pulled_arm_id].numeric_id == pulled_arm_id)            
        return self.arms[pulled_arm_id]
        
        

    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)
        for arm in range(self.n_arms):
            #check correct list of arms
            active_buckets = []
            for b in self.arms[arm].buckets:
                # m is the relative time of a bucket
                m = int(self.t - b.t_start)
                # if m == tmax : the bucket is ended
                if m < self.tmax:
                    active_buckets.append(b)
                else:
                    value = sum(b.values)
                    assert(value>=0)
                    assert(value<=self.tmax)
                    normalized_value = value/self.tmax
                    if(random.random() <= normalized_value):
                        self.alpha_v[arm] = self.alpha_v[arm]+1
                    else:
                        self.beta_v[arm] = self.beta_v[arm]+1

            self.arms[arm].buckets = active_buckets


class ThompsonLearnerExplorerSpotify(Learner):
 
    def __init__(self, n_arms, arms, tmax, exploration_factor, optimistic = False):
        name = "Thompson"        
        name += "_expl_"
        name += str(exploration_factor)        
        if optimistic == True:
            name += "_optim"

        super().__init__(n_arms,arms,name)
      
        self.hits_table = np.zeros((tmax,n_arms))
        self.visits_table = np.zeros((tmax,n_arms))
        self.init_prior_alpha = np.ones((tmax,n_arms))
        self.init_prior_beta = np.ones((tmax,n_arms))
        self.tmax = tmax
        self.K_EXP = exploration_factor
        self.arms_r_vector = np.zeros(n_arms)        
        self.optimistic = optimistic
        self.exploration_counter = np.zeros(n_arms)
        self.vectorized_optimistic_function = np.vectorize(self.optimistic_function)
       
        #assign_numeric_id
        for id_arm in range(n_arms):
            self.arms[id_arm].numeric_id = id_arm        
            self.arms_r_vector[id_arm] = self.arms[id_arm].reward


    def optimistic_function(self,t,avg):
            if t > avg:
                return t
            else:
                return avg
    
    def get_optimistic_theta(self,theta,S,F,alphas,betas):
        avg = (S+alphas)/(S+alphas+F+betas)        
        new_theta = self.vectorized_optimistic_function(theta,avg)
        return new_theta

        


    def update_tmax_seen(self):
        #tmax_seen update
        for arm in range(self.n_arms):
            for b in self.arms[arm].buckets:
                m = int(self.t - b.t_start)
                assert(m<=self.tmax)
                if(m< self.tmax and b.values[m]>0):
                    self.tmax_seen = max(self.tmax_seen,m+1)
                    break
                   
        

    def pull_arm(self):
        #myopic static
        # draw THETAmi according to the Beta 
        self.exploration_counter +=1            
        
        S = self.hits_table
        F = self.visits_table - self.hits_table

        THETA = np.random.beta(S + self.init_prior_alpha, F + self.init_prior_beta)
        if(self.optimistic):
           THETA = self.get_optimistic_theta(THETA,S,F,self.init_prior_alpha,self.init_prior_beta)
        
        #exploration bonus
        exp_bonus = self.exploration_counter*self.K_EXP
        THETA_summed = THETA.sum(axis = 0)
        THETA_altered = THETA_summed + exp_bonus
        
        pulled_arm_id = np.argmax(THETA_altered*self.arms_r_vector)
        assert(self.arms[pulled_arm_id].numeric_id == pulled_arm_id)
        self.exploration_counter[pulled_arm_id]=0           
        return self.arms[pulled_arm_id]
        
        

    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)
        for arm in range(self.n_arms):
            #check correct list of arms
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
                    if b.values[m]>0:
                        self.hits_table[m][arm] = self.hits_table[m][arm]+1
            self.arms[arm].buckets = active_buckets


class BayesUCBPersistentSpotify(Learner):
 
    def __init__(self, n_arms, arms, tmax):
        name = "BayesUCBPersistent"        
        super().__init__(n_arms,arms,name)
      
        self.hits_table = np.zeros((tmax,n_arms))
        self.visits_table = np.zeros((tmax,n_arms))
        self.init_prior_alpha = np.ones((tmax,n_arms))
        self.init_prior_beta = np.ones((tmax,n_arms))
        self.tmax = tmax
        self.arms_r_vector = np.zeros(n_arms)        
               
        #assign_numeric_id
        for id_arm in range(n_arms):
            self.arms[id_arm].numeric_id = id_arm        
            self.arms_r_vector[id_arm] = self.arms[id_arm].reward
                   
        

    def pull_arm(self):
        #myopic static
        # draw THETAmi according to the Beta             
        S = self.hits_table
        F = self.visits_table - self.hits_table
        THETA = scipybeta.ppf((1-(1/(1+self.t))),S + self.init_prior_alpha, F + self.init_prior_beta)
        # chose armi
        pulled_arm_id = np.argmax(THETA.sum(axis = 0)*self.arms_r_vector)
        assert(self.arms[pulled_arm_id].numeric_id == pulled_arm_id)            
        return self.arms[pulled_arm_id]
    
    def update_observations(self, pulled_arm, bucket):
        super().update_observations(pulled_arm, bucket)
        for arm in range(self.n_arms):
            #check correct list of arms
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
                    if b.values[m]>0:
                        self.hits_table[m][arm] = self.hits_table[m][arm]+1
            self.arms[arm].buckets = active_buckets