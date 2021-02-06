from RentLearner import RentLearner
import numpy as np

class RentBound1LearnerPartial(RentLearner):

    def __init__(self, n_arms, arms, tmax_a, tmax_s, t_global):
        super().__init__(n_arms,arms,"RentBound1LearnerPartial")

        self.hits_table = np.zeros((t_global,n_arms))
        self.visits_table = np.zeros((t_global,n_arms))
        self.sfitto_table = np.zeros((t_global,n_arms))

        self.bound1_criterion = np.zeros(n_arms)
        self.noexp_crit = np.zeros(n_arms)
        self.tmax_a=tmax_a
        self.tmax_s=tmax_s
        self.t_global = t_global


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
            assert(self.arms[arm].id_code == arm)
            active_buckets = []

            for b in self.arms[arm].buckets:
                # m is the relative time of a bucket
                m = int(self.t - b.t_start)
                # if m == tmax : the bucket is ended
                if m < self.t_global:
                    active_buckets.append(b)
                    #update visite
                    self.visits_table[m][arm] = self.visits_table[m][arm]+1
                    #update dei successi
                    if b.values[m]>0:
                        self.hits_table[m][arm] = self.hits_table[m][arm]+1
                        b.was_active = True #TODO ottimizzare
                    
                    if b.values[m] == 0 and b.was_active == False:
                        self.sfitto_table[m][arm] = self.sfitto_table[m][arm]+1
                    

            self.arms[arm].buckets = active_buckets
        
                
        #criterion update
        for arm in range(self.n_arms):
            num = 0
            den = 0
            num_no_exp = 0
            den_no_exp = 0
            print("arm")
            print(arm)
            print("hits")
            print(self.hits_table)
            print("vistis")
            print(self.visits_table)
            print("sfitto")
            print(self.sfitto_table)
            print("t")
            print(self.t)
            print("c")
            print(np.sqrt((2 * np.log( self.t))/    self.visits_table[0][arm]))

            for i in range(self.t_global):                
                x_i = s_i = c_i = 0       

                if self.visits_table[i][arm]>0 :
                    x_i = self.hits_table[i][arm] / self.visits_table[i][arm]
                    s_i = self.sfitto_table[i][arm] / self.visits_table[i][arm] 
                    c_i = np.sqrt((2 * np.log( self.t))/    self.visits_table[i][arm])

                assert(c_i>=0)
                assert(x_i>=0 and x_i<=1)
                assert(s_i>=0 and s_i <=1)
                num +=  (x_i + c_i)
                den +=  (s_i + x_i - c_i - c_i)
                num_no_exp +=x_i
                den_no_exp +=max(1,(s_i+x_i))
            
            assert(num>=num_no_exp)
            assert(den<=den_no_exp)
            assert((num/den))>=(num_no_exp/max(1,den_no_exp))
            criterion =  self.arms[arm].canone * (num/max(1,den))            
            #criterion =  self.arms[arm].canone * ((num_no_exp+c_i)/max(1,(den_no_exp-2*c_i)))
            print("criterion")
            print(criterion)
            print("full criterium")
            print(self.bound1_criterion)
            self.bound1_criterion[arm]= criterion           

            if(den_no_exp>0):
                self.noexp_crit[arm] = self.arms[arm].canone * num_no_exp/den_no_exp



class RentBound1Learner(RentLearner):

    def __init__(self, n_arms, arms, tmax_a, tmax_s, t_global):
        super().__init__(n_arms,arms,"RentBound1Learner")

        self.hits_table = np.zeros((t_global,n_arms))
        self.visits_table = np.zeros((t_global,n_arms))
        self.sfitto_table = np.zeros((t_global,n_arms))

        self.bound1_criterion = np.zeros(n_arms)
        self.noexp_crit = np.zeros(n_arms)
        self.tmax_a=tmax_a
        self.tmax_s=tmax_s
        self.t_global = t_global


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
            assert(self.arms[arm].id_code == arm)
            active_buckets = []

            for b in self.arms[arm].buckets:
                # m is the relative time of a bucket
                m = int(self.t - b.t_start)
                # if m == tmax : the bucket is ended
                if m < self.t_global:
                    active_buckets.append(b)
                    #update visite
                    self.visits_table[m][arm] = self.visits_table[m][arm]+1
                    #update dei successi
                    if b.values[m]>0:
                        self.hits_table[m][arm] = self.hits_table[m][arm]+1
                        b.was_active = True #TODO ottimizzare
                    
                    if b.values[m] == 0 and b.was_active == False:
                        self.sfitto_table[m][arm] = self.sfitto_table[m][arm]+1
                    

            self.arms[arm].buckets = active_buckets
        
                
        #criterion update
        for arm in range(self.n_arms):
            num = 0
            den = 0
            num_no_exp = 0
            den_no_exp = 0

            for i in range(self.t_global):                
                x_i = s_i = c_i = 0       
                if self.visits_table[i][arm]>0 :
                    x_i = self.hits_table[i][arm] 
                    s_i = self.sfitto_table[i][arm]
                                
                num +=  (x_i)
                den +=  (s_i + x_i)
                num_no_exp +=x_i
                den_no_exp +=(s_i+x_i)
            c_i = np.sqrt((2 * np.log( self.t))/    self.visits_table[0][arm])
            assert(num>=num_no_exp)
            assert(den<=den_no_exp)
            assert((num/max(1,den))>=(num_no_exp/max(1,den_no_exp)))
            criterion =  self.arms[arm].canone * (num/max(1,den +c_i))            
            #criterion =  self.arms[arm].canone * ((num_no_exp+c_i)/max(1,(den_no_exp-2*c_i)))
            self.bound1_criterion[arm]= criterion
            if(den_no_exp>0):
                self.noexp_crit[arm] = self.arms[arm].canone * num_no_exp/max(1,den_no_exp)


class RentBound1LearnerAdaptive(RentLearner):

    def __init__(self, n_arms, arms, tmax_a, tmax_s, t_global):
        super().__init__(n_arms,arms,"RentBound1LearnerAdapt")

        self.hits_table = np.zeros((t_global,n_arms))
        self.visits_table = np.zeros((t_global,n_arms))
        self.sfitto_table = np.zeros((t_global,n_arms))

        self.t_affitto_seen = np.zeros(n_arms)
        self.t_sfitto_seen = np.zeros(n_arms)
        self.first_hit = np.zeros(n_arms)

        self.bound1_criterion = np.zeros(n_arms)
        self.noexp_crit = np.zeros(n_arms)
        self.tmax_a=tmax_a
        self.tmax_s=tmax_s
        self.t_global = t_global


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
            assert(self.arms[arm].id_code == arm)
            active_buckets = []

            for b in self.arms[arm].buckets:
                # m is the relative time of a bucket
                m = int(self.t - b.t_start)
                # if m == tmax : the bucket is ended
                if m < self.t_global:
                    active_buckets.append(b)
                    #update visite
                    self.visits_table[m][arm] = self.visits_table[m][arm]+1
                    #update dei successi
                    if b.values[m]>0:
                        self.hits_table[m][arm] = self.hits_table[m][arm]+1
                        b.was_active = True #TODO ottimizzare
                    
                    if b.values[m] == 0 and b.was_active == False:
                        self.sfitto_table[m][arm] = self.sfitto_table[m][arm]+1
            
            self.arms[arm].buckets = active_buckets

            #first_hit

            #print(arm)
            where = np.where(self.hits_table[:,arm]>0)[0]            
            if(len(where)>0):
                self.first_hit[arm] = where[0]
            #t_affitto_seen
            where = np.where(self.hits_table[:,arm]>0)[0]
            if(len(where)>0):
                self.t_affitto_seen[arm] = where[len(where)-1]+1
            #t_sfitto_seen
            where = np.where(self.sfitto_table[:,arm]>0)[0]
            if(len(where)>0):
                self.t_sfitto_seen[arm] = where[len(where)-1]+1
            
        
                
        #criterion update
        for arm in range(self.n_arms):
            num = 0
            den = 0

            #NUMERATORE
            for i in range(int(self.first_hit[arm]),int(self.t_affitto_seen[arm])):
                x_i = self.hits_table[i][arm] / self.visits_table[i][arm]
                c_i = np.sqrt((2 * np.log( self.t))/    self.visits_table[i][arm])
                num += x_i + c_i

            #DENOMINATORE SFITTO
            for i in range(0,int(self.t_sfitto_seen[arm])):
                s_i = self.sfitto_table[i][arm] / self.visits_table[i][arm]
                c_i = np.sqrt((2 * np.log( self.t))/    self.visits_table[i][arm])
                den += max(0,s_i - c_i)
            #DENOMINATORE AFFITTO
            for i in range(int(self.first_hit[arm]),int(self.t_affitto_seen[arm])):
                x_i = self.hits_table[i][arm] / self.visits_table[i][arm]
                c_i = np.sqrt((2 * np.log( self.t))/    self.visits_table[i][arm])
                den += max(0,x_i - c_i)
            


            criterion =  self.arms[arm].canone * (num/max(1,den))    
            if(den<1):
                print("-------------------------------------------------------------FAIL")        
            self.bound1_criterion[arm]= criterion
