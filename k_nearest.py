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
        
    def tuners(self):
        tuner_index = random.sample(self.index, k=math.ceil(len(self.index) * .1))
        self.tuners = self.df.filter(items = tuner_index, axis=0)
        self.learning_set = self.df.drop(tuner_index,  axis=0)

    def stratified_partition(self, k):
        p = [[] for i in range(k)]
        if self.classification:
            def class_partition(df, p, c):
                n = df.shape[0]
                (q, r) = (n // k, n % k)
                j = 0
                for i in range(k):
                    z = (i + c) % k
                    p[z] = p[z] + [df.at[x, 'index'] for x in range(j, j + q + int(i < r))]
                    j += q + int(i < r)
                return (p, c + r)
            c = 0
            for cl in self.classes:
                df = self.df[self.df['Target'] == cl].reset_index()
                (p, c) = class_partition(df, p, c)
        else:
            df = self.df.sort_values(by=['Target']).reset_index()
            n = df.shape[0]
            (q, r) = (n // k, n % k)
            for i in range(k):
                p[i] = p[i] + [df.at[i + c * k, 'index'] for c in range(q + int(i < r))]
        return p

        # separate into training and test sets
    def training_test_sets(self, j, df, partition=None):
        if partition is None: partition = self.stratified_partition(10)
        train = []
        for i in range(len(partition)):
            if j != i:
                train += partition[i]
            else:
                test = partition[i]
        self.train_set = df.filter(items=train, axis=0).reset_index()
        self.test_set = df.filter(items=test, axis=0).reset_index()

    def norm_2_distance(self, x1, x2):
        d = 0
        for f_num in self.features_ohe:
            d += math.pow(x1[f_num] - x2[f_num], 2)
        return math.sqrt(d)


    def naiveEstimator(self, x, h):
        def P(self, x, cl, h):
            def kernel(u):
                return int(abs(u) < 1 / 2)
            p = 0
            for t in self.train_set[self.train_set['Target'] == cl].index:
                p += kernel(self.norm_2_distance(x, self.value(self.train_set, t)) / h)
            return p
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = P(x, cl, h)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax

    def kernelEstimator(self, x, h):
        def P(x, cl):
            def kernel(u):
                return math.exp(-math.pow(u, 2)/2) / math.sqrt(2 * math.pi)
            p = 0
            for t in self.train_set[self.train_set['Target'] == cl].index:
                p += kernel(self.norm_2_distance(x, self.value(self.train_set, t)) / h)
            return p
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = P(x, cl)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax

    def nearestneighbors_naive(self, x, k):
        distances = self.train_set.index.map(lambda i: self.norm_2_distance(x, self.value(self.train_set, i)))
        (_, indices) = distances.sort_values(return_indexer = True)
        return indices.take(range(k))

    def nearestneighborEstimator(self, x, k):
        nn = self.nearestneighbors_naive(x, k)
        nn = self.train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
        print(type(nn))
        def P(x, cl):
            if cl in nn.index:
                return nn.at[cl] / k
            else:
                return 0
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = P(x, cl)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax


    def split(file):
        
        return 



    def getLearning(self):
        return self.learning_set

    def getTune(self):
        return self.tuners

    '''
    Train the file that is sent in
    
    @param file - file that the program will be training on
    '''    
    def TrainFile(self):
        df = self
        df.tuners()
        tune = df.getTune()
        print(tune)
        train = df.getLearning()
        p = train.stratified_partition(10)
        df.training_test_sets(train)   
        
    
        
        
        
        
        
        

