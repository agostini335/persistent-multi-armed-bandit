import numpy as np
from RentBucket import RentBucket
import random

class RentEnv():

    def __init__(self,tmax_a, tmax_s):        
        self.tmax_a = tmax_a
        self.tmax_s = tmax_s

    def round(self, pulled_arm, contratto_dict, sfitto_dict):
        bucket = self.generate_bucket_sfitto(pulled_arm,contratto_dict,sfitto_dict)        
        return bucket

    
    def generate_bucket_sfitto(self,pulled_arm,contratto_dict,sfitto_dict):
        c_key = pulled_arm.canone
        durata_affitto = random.choice(contratto_dict[str(c_key)])
        durata_sfitto = random.choice(sfitto_dict[str(c_key)])

        sfitto_zeros = np.zeros(durata_sfitto)
        affitto_ones = np.ones(durata_affitto)
        rest_zeros = np.zeros(self.tmax_a+self.tmax_s-durata_affitto-durata_sfitto)
        
    
        bucket = RentBucket(values = np.concatenate((sfitto_zeros,affitto_ones,rest_zeros)), zeros_sfitto=durata_sfitto,ones_contratto=durata_affitto)
       
        return bucket
    