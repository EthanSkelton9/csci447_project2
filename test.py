
'''
Will predict values in the test set of data and determine if the hyperparameters need to be changed
'''
class Test:
    
    '''
    Initiate class Test 
    
    @param name - name of file 
    
    '''
    def __init__(self, name):
        self.name = name
     
     
    def printName(self):
        print(self.name)