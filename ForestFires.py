from k_nearest import K_Nearest

class ForestFires (K_Nearest):
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
            "a",
            "b",
            "c"
            ]

        #initiate forestfires test set
        super().__init__(file='forestfires.csv', features = features, name = 'ForestFires', classLoc = 'end', classification = False)