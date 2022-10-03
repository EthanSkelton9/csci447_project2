from numpy import int64
import pandas as pd
import os
import math
import random

class Learning:
    def __init__(self, file, features, name, classLoc, replaceValue = None, classification = True):
        
        
        if(features != None):
            df = pd.read_csv(os.getcwd() + r'\Raw Data' + '\\' + file, header=None)
            self.features = features
        else:
            df = pd.read_csv(os.getcwd() + r'\Raw Data' + '\\' + file)
            self.features = df.keys().to_list()
            if(classLoc == 'beginning'):
                self.features.pop(0)
            if(classLoc == 'end'):
                self.features.pop()
        self.df = df  # dataframe
        self.name = name
        if replaceValue: self.findMissing(replaceValue)   

        print(type(df.iloc[24,6]))

        self.classification = classification
        self.addColumnNames(classLoc, classification)  # add column names to correct spot
        self.one_hot_encoding()
        self.z_score_normalization()
        if classification: self.classes = list(set(self.df['Target']))
        self.df.to_csv(
            os.getcwd() + '\\' + str(self) + '\\' + "{}_w_colnames.csv".format(str(self)))  # create csv of file
        self.seed = random.random()


        # function to find Missing data from dataset
    def findMissing(self, replaceValue):
        colToChange = None
        for col_name in self.df.columns:
            self.df[col_name] = self.df[col_name].replace(['?'], [int64(replaceValue)])
            colToChange = col_name
        self.df[colToChange] = pd.to_numeric(self.df[colToChange])
        

    def __str__(self):
        return self.name

    def addColumnNames(self, classLoc, classification):
        if (classLoc == 'beginning'):  # if the class column is at the beginning
            self.df.columns = ['Target'] + self.features
            # shift the class column to the last column
            last_column = self.df.pop('Target')
            self.df.insert(len(self.df.columns), 'Target', last_column)
        elif (classLoc == 'end'):  # if the class column is at the end -> continue as normal
            self.df.columns = self.features + ['Target']
        else:
            print('Not sure where to place Class column')

    def one_hot_encoding(self):
        (features_numerical, features_categorical) = ([], [])
        features_categorical_ohe = []
        for f in self.features:
            try:
                self.df[f].apply(pd.to_numeric)
                features_numerical.append(f)
            except:
                features_categorical.append(f)
                categories = set(self.df[f])
                for cat in categories:
                    features_categorical_ohe.append("{}_{}".format(f, cat))
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        self.df = pd.get_dummies(self.df, columns=self.features_categorical)
        self.features_ohe = features_numerical + features_categorical_ohe
        target_column = self.df.pop('Target')
        self.df.insert(len(self.df.columns), 'Target', target_column)

    def z_score_normalization(self):
        for col in self.features_ohe:
            std = self.df[col].std()
            if std != 0:
                self.df[col] = (self.df[col] - self.df[col].mean()) / std

    # return value of a ceratin feature
    def value(self, df, i):
        return df.loc[i, self.features_ohe]

