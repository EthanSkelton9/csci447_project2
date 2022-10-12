from IBL import IBL

class ForestFires (IBL):
    def __init__(self):
        #list of feature names(excluding class)
        features = [ #Class at end
            
            ]

        #initiate forestfires test set
        super().__init__(file='forestfires.csv', features = None, name = 'ForestFires', classLoc = 'end', replaceValue = None, classification = False)