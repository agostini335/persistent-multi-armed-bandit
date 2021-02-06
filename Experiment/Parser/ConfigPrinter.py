from ExperimentDescription import *
from ArmSet import *
import json


a1= ArmSet(alpha=1,beta=1,starting_reward=10,quantity=10,increment=0)
a2= ArmSet(alpha=1,beta=3,starting_reward=1,quantity=10,increment=0.5)
a3= ArmSet(alpha=1,beta=5,starting_reward=1,quantity=10,increment=1.5)

experiment_desc = ExperimentDescription(tmax=100,time=100,arm_sets=[a1,a2,a3])

#saving
with open('ConfigFiles/experiment_1.json', 'w', encoding='utf-8') as f:
    json.dump(experiment_desc,   f, default=lambda o: o.__dict__,ensure_ascii=False, indent=4)

