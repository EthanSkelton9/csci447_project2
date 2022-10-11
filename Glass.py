from k_nearest import K_Nearest
from IBL import IBL

class Glass (IBL):
    def __init__(self):
        #list of feature names(excluding class)
        features = [ #Class at end
            "Id number: 1 to 214",
            "RI: refractive index",
            "Na: Sodium",
            "Mg: Magnesium",
            "Al: Aluminum",
            "Si: Silicon",
            "K: Potassium",
            "Ca: Calcium",
            "Ba: Barium",
            "Fe: Iron"
            ]

        #initiate glass test set
        super().__init__(file='glass.csv', features = features, name = 'Glass', classLoc = 'end', replaceValue = None, classification =
                         True)