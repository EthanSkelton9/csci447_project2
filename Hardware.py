from k_nearest import K_Nearest

class Hardware (K_Nearest):
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
            "Ba: Barium"
            ]

        #initiate glass test set
        super().__init__(file='machine.csv', features = features, name = 'Hardware', classLoc = 'end', replaceValue = None, classification =
                         False)