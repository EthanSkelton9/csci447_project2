from re import A
from k_nearest import K_Nearest
from SoyBean import SoyBean
from Abalone import Abalone
from Glass import Glass
import os

'''
    Driver for k Nearest Neighbor    
'''
def main_Ethan():

    #hyperparameters:
    #a
    #b
    datasets = []
    datasets.append(Glass().Learning())

    for ds in datasets:
        ds.split()
        #for loop that goes through split dataset testing on each 10%
        k = K_Nearest()
        train = k.TrainFile(ds)
        test = k.Test_file(ds, train)
        #change hyperparameters for next dataset
        a = test.a
    
def main_Ian():
    D = Abalone()


if __name__ == '__main__':
    main_Ian()
