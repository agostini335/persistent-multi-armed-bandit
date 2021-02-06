import json
class ExperimentDescription():

    def __init__(self,tmax,time,arm_sets,n_runs = 1):
        self.tmax = tmax
        self.time = time
        self.n_runs = n_runs
        self.arm_sets = arm_sets
    
    @classmethod
    def from_json(cls, json_data: dict):
        return cls(**json_data)
       
