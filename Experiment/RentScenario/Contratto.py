class Contratto():
    def __init__(self,nome, canone, durata, data_inizio, data_distetta,data_scadenza, sfitto = -1):
        self.nome = nome
        self.canone = canone
        self.durata = durata
        self.sfitto = sfitto
        self.data_inizio = data_inizio
        self.data_disdetta = data_distetta
        self.data_scadenza = data_scadenza
    