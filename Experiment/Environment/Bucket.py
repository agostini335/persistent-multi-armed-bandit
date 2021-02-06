
class Bucket():
    """
    class used to represent a single Bucket 
    ....
    
    Attributes
    ----------
    t_start : int
        global time istant when the bucket is generated 
    values : int[]
        array of the realizations
    Methods
    -------
        
    """
    def __init__(self, values, t_start = None, is_active = True):
        self.t_start = t_start
        self.values = values
        self.is_active = True

        
