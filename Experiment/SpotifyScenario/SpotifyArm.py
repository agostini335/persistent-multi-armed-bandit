import numpy as np
from scipy.stats import beta as bt


class SpotifyArm():
    
  
    def __init__(self, value, id_code ):
        self.value = value
        self.id_code = id_code
        self.buckets = []
        self.non_active_buckets = []
        self.reward = 1
        self.numeric_id = -1
    




