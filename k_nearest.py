#libraries


#import other classes
from train import Train
from test import Test


from IanClass import IanClass
from EthanClass import EthanClass
from asyncio.windows_events import NULL
import pandas as pd
import os
import math
import random
from ConfusionMatrix import ConfusionMatrix
import numpy as np
    
class K_Nearest(IanClass, EthanClass):
    
    
    '''
    Initiate class K_Nearest 
    
    @param name - name of file 
    
    '''
    def __init__(self, file, features, name, classLoc, replaceValue , classification = True):
        super().__init__(file, features, name, classLoc, replaceValue, classification)
        
    
    '''
    finds the missing values in the dataset and replaces them so that we dont have to worry about it
    
    @param file - file that we are checking for missing values
    
    @return - return the file without any missing values
    '''
    def missingValues(self, file):
        
        #replace missing values in the file
        
        
        return file
    
    '''
    Split the file into a test set
    
    @parameter file - initial file that the code has
    
    @return the part we are going to test the file on
    '''
    def test_file(file):
        return file
    
    
    
    '''
    Split the file into a train set
    
    @parameter file - initial file that the code has
    
    @return the part we are going to train the file on
    '''
    def train_file(file):
        return file
    
    
    '''
    Train the file that is sent in
    
    @param file - file that the program will be training on
    '''    
    def TrainFile(self, file):
        
        file = self.missingValues(file) #get rid of missing values in file
        
    #partition the file here
        
        trainf = self.test_file(file) #split data into training set
        testf = self.train_file(file) #split into testing set
        
        train = Train(trainf)
        test = Test(testf)
        
        
        
        
        
        

