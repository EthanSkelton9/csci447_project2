from IanClass import IanClass
from EthanClass import EthanClass
from asyncio.windows_events import NULL
import pandas as pd
import os
import math
import random
from ConfusionMatrix import ConfusionMatrix
import numpy as np


class IBL (IanClass, EthanClass):
    def __init__(self, file, features, name, classLoc, replaceValue = None):
        df = pd.read_csv(os.getcwd() + r'\Raw Data' + '\\' + file)
        self.df = df  # dataframe
        self.features = features
        self.name = name
        self.addColumnNames(classLoc)  # add column names to correct spot
        self.classes = list(set(self.df['Class']))
        if replaceValue: self.findMissing(replaceValue)  # replace missing values
        self.df.to_csv(
            os.getcwd() + '\\' + str(self) + '\\' + "{}_w_colnames.csv".format(str(self)))  # create csv of file
        self.seed = random.random()



