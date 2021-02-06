
class ArmSet():
    def __init__(self,alpha,beta,starting_reward,quantity,increment = 0, id_code = -1, arm_value = -1):
        self.alpha = alpha
        self.beta = beta
        self.starting_reward = starting_reward
        self.quantity = quantity
        self.reward_incrment_step = increment
        self.id_code = id_code
        self.arm_value = arm_value

    
    @classmethod
    def from_json(cls, data):
        return cls(**data)
        


        