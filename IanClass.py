import pandas as pd
import os
import math
import random
from Learning import Learning

class IanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue = None):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue = replaceValue)

    def stratified_partition(self, k):
        def partition(df, p, c):
            n = df.shape[0]
            (q, r) = (n // k, n % k)
            j = 0
            for i in range(k):
                z = (i + c) % k
                p[z] = p[z] + [df.at[x, 'index'] for x in range(j, j + q + int(i < r))]
                j += q + int(i < r)
            return (p, c + r)
        p = [[] for i in range(k)]
        c = 0
        for cl in self.classes:
            df = self.df[self.df['Class'] == cl].reset_index()
            (p, c) = partition(df, p, c)
        return p




    # separate into training and test sets
    def training_test_sets(self, j, df, partition=None):
        if partition is None: partition = self.partition(10)
        train = []
        for i in range(len(partition)):
            if j != i:
                train += partition[i]
            else:
                test = partition[i]
        self.train_set = df.filter(items=train, axis=0)
        self.test_set = df.filter(items=test, axis=0)