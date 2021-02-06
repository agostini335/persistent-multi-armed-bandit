class RentBucket():

    def __init__(self, values, zeros_sfitto, ones_contratto, t_start = None, is_active = True):
        self.t_start = t_start
        self.values = values
        self.zeros_sfitto = zeros_sfitto
        self.ones_contratto = ones_contratto
        self.is_active = True
        self.was_active = False
