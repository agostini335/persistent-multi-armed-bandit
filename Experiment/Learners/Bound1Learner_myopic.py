from Learners.Learner import Learner
import numpy as np


class Bound1Learner_myopic(Learner):
    """
    class used to represent the Bound1 Learner myopic: the learner is unable to see the full bucket after
        the first zero.
    ....
    
    Attributes
    ----------
    hits_table : int((tmax,n_arms))
        matrix representing the number of hits obtained by an arm i at a step j (relative time of a bucket)
        it must be updated according with the time of the experiment
    visits_table: int((tmax,n_arms))
        matrix representing the number of visits done by an arm i at a step j (relative time of a bucket)
        it must be updated according with the time of the experiment
    bound1_criterion: double((n_arms))
        criterion of each arm
    tmax: int
        lenght of a bucket

    Methods
    -------
    pull_arm(self):
        -init phase
        -return the next arm to pull according to the criterion
    update_observations(self, pulled_arm, bucket):
        -tables update
        -criterion update 
    """

    def __init__(self, n_arms, arms, tmax):
        super().__init__(n_arms,arms,"Bound1_myopic")
        self.hits_table = np.zeros((tmax,n_arms))
        self.visits_table = np.zeros((tmax,n_arms))
        self.bound1_criterion = np.zeros(n_arms)
        self.tmax=tmax


    def pull_arm(self):
        #init phase
        if self.t < self.n_arms:
            return self.arms[self.t]
        idxs = np.argwhere(self.bound1_criterion == self.bound1_criterion.max()).reshape(-1)
        pulled_arm_id = np.random.choice(idxs)
        return self.arms[pulled_arm_id]
    

    def update_observations(self, pulled_arm, bucket):

        super().update_observations(pulled_arm, bucket)
        

        #tables update
        for arm in range(self.n_arms):
            #check correct list of arms
            #assert(self.arms[arm].id_code == arm)
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
        
                
        #criterion update
        for arm in range(self.n_arms):

            criterion = 0
            for i in range(self.tmax):                
                x_i = c_i = 1                
                if self.visits_table[i][arm]>0 :
                    x_i = self.hits_table[i][arm]/self.visits_table[i][arm]
                    c_i = np.sqrt(2 * np.log( self.t*np.power(self.tmax,1/4))/ self.visits_table[i][arm])  
                
                criterion += min(1,x_i+c_i)              
    
            self.bound1_criterion[arm]= self.arms[arm].reward * criterion





