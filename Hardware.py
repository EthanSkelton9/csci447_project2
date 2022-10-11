from k_nearest import K_Nearest
from IBL import IBL

class Hardware (IBL):
    def __init__(self):
        #list of feature names(excluding class)
        features = [ #Class at end
            "Vendor Name",
            "Model Name",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "ERP"
            ]

        #initiate glass test set
        super().__init__(file='machine.csv', features = features, name = 'Hardware', classLoc = 8, replaceValue = None, classification =
                         False)