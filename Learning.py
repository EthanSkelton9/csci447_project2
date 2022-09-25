import pandas as pd
import os
import math
import random

class Learning:
    def __init__(self, file, features, name, classLoc, replaceValue = None):
        df = pd.read_csv(os.getcwd() + r'\Raw Data' + '\\' + file)
        self.df = df  # dataframe
        self.features = features
        self.name = name
        self.addColumnNames(classLoc)  # add column names to correct spot
        self.classes = list(set(self.df['Class']))
        self.df.to_csv(
            os.getcwd() + '\\' + str(self) + '\\' + "{}_w_colnames.csv".format(str(self)))  # create csv of file
        self.seed = random.random()

    def __str__(self):
        return self.name

    def addColumnNames(self, classLoc):
        if (classLoc == 'beginning'):  # if the class column is at the beginning
            self.df.columns = ['Class'] + self.features
            # shift the class column to the last column
            last_column = self.df.pop('Class')
            self.df.insert(len(self.df.columns), 'Class', last_column)
        elif (classLoc == 'end'):  # if the class column is at the end -> continue as normal
            self.df.columns = self.features + ['Class']
        else:
            print('Not sure where to place Class column')