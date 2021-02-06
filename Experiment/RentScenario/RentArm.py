import numpy as np
from scipy.stats import beta as bt


class RentArm():

    def __init__(self, canone, k_a_s = -1,id_code=-1 ):
        self.id_code = id_code
        self.canone = canone
        self.k_a_s = k_a_s
        self.value = self.k_a_s*self.canone
        self.buckets = []
        self.non_active_buckets = []
        