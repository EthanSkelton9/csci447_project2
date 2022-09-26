from k_nearest import K_Nearest
from SoyBean import SoyBean
from Abalone import Abalone
import os

'''
    Driver for k Nearest Neighbor    
'''
def main_Ethan():
    ds = K_Nearest("hello")
    ds.TrainFile()  #train file and return with new hyperparameters to input

def main_Ian():
    D = Abalone()


if __name__ == '__main__':
    main_Ian()
