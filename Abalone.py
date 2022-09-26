from IBL import IBL

class Abalone (IBL):
    def __init__(self):
        # list of feature names(excluding class)
        features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight',
                    'Shell Weight']

        # initiate soybeans test set
        super().__init__(file='abalone.csv', features=features, name="Abalone", classLoc='end', classification = False)