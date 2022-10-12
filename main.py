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

    #G.test(k_space = range(4, 9), appendCount = 30, seed = 2)
    A    .test(k_space = range(4, 9), sigma_space = range(4,14), appendCount = 30, seed = 2)

    
def main_Ian():
    D = Abalone()
    D.Ian_test(k_space = range(4, 9), sigma_space = range(4, 14), appendCount = 3, seed = 7)
    # NN
    #-----------------
    # SoyBean Complete
    # Glass Complete
    # BreastCancer Complete
    # Hardware Complete
    # ForestFires Complete




if __name__ == '__main__':
    #main_Ian()
    main_Ethan()
