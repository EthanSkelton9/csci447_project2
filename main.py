from re import A
from Hardware import Hardware
from SoyBean import SoyBean
from Abalone import Abalone
from Glass import Glass
from ForestFires import ForestFires
from BreastCancer import BreastCancer
import os
import pandas as pd
import VideoScripts

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

    B.test(k_space = range(4, 9), appendCount = 30, seed = 2)
    #H.test(k_space = range(4, 9), sigma_space = range(4,14), appendCount = 30, seed = 2)

    
def main_Ian():
    D = Abalone()
    # D.test(k_space = range(4, 9), head = 100, seed = 2)
    D.latex_display()
    #D.Ian_test(k_space = range(4, 9), head = 100, sigma_space = [4, 13], epsilon_space = range(2, 7), appendCount = 50, seed = 7)
    # NN
    #-----------------
    # SoyBean Complete
    # Glass Complete
    # BreastCancer Complete
    # Hardware Complete
    # ForestFires Complete


def main_video():
    D = Abalone()

    #VideoScripts.show_partitions(D)
<<<<<<< HEAD
    #VideoScripts.show_distance_and_kernel(D)
    #VideoScripts.nn_classification(D)
    VideoScripts.display(D)
=======
    #VideoScripts.show_distance(D)
    #VideoScripts.show_kernel(D,4)
    VideoScripts.cluster(D)
>>>>>>> 11b6bb003572fea2ba3d6184cac394e3e60a7992


if __name__ == '__main__':
    #main_Ian()
    #main_Ethan()
    main_video()
