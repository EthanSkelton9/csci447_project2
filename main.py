from re import A
from Hardware import Hardware
from SoyBean import SoyBean
from Abalone import Abalone
from Glass import Glass
from ForestFires import ForestFires
from BreastCancer import BreastCancer
import os
import pandas as pd

'''
    Driver for k Nearest Neighbor    
'''
def main_Ethan():

    #Read in datasets
    F = ForestFires()
    G = Glass()
    H = Hardware()
    B = BreastCancer()
    A = Abalone()
    S = SoyBean()

    A.test(k_space = range(4, 9), sigma_space = range(4, 14), appendCount = 30, seed = 2)
    


    # #hyperparameters:
    # #a
    # #b
    # datasets = []
    # datasets.append(Glass().Learning())

    # for ds in datasets:
    #     ds.split()
    #     #for loop that goes through split dataset testing on each 10%
    #     k = K_Nearest()
    #     train = k.TrainFile(ds)
    #     test = k.Test_file(ds, train)
    #     #change hyperparameters for next dataset
    #     a = test.a


    
def main_Ian():
    D = Abalone()
    D.Ian_test(k_space = range(4, 9), sigma_space = range(4, 14), appendCount = 20, seed = 7)
    # NN
    #-----------------
    # SoyBean Complete
    # Glass Complete
    # BreastCancer Complete
    # Hardware Complete
    # ForestFires Complete




if __name__ == '__main__':
    main_Ian()
    #main_Ethan()
