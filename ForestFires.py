import IBL

class Glass (IBL):
    def __init__(self):

        #initiate forestfires test set
        super().__init__(file='forestfires.csv', name = 'Fires', classLoc = 'end')