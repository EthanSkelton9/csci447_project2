import pandas as pd
import os
import math
import random
from Learning import Learning

class IanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue = None, classification = True):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue = replaceValue,
                         classification = classification)

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
        self.train_set = df.filter(items=train, axis=0)
        self.test_set = df.filter(items=test, axis=0)

    def norm_2_distance(self, x1, x2):
        d = 0
        for f_num in self.features_ohe:
            d += math.pow(x1[f_num] - x2[f_num], 2)
        return math.sqrt(d)

    def P(self, x, cl, h):
        def kernel(x):
            return int(abs(x) < 1/2)
        p = 0
        for t in self.train_set[self.train_set['Target'] == cl].index:
            p += kernel(self.norm_2_distance(x, self.value(self.train_set, t)) / h)
        return p

    def predicted_class(self, x, h):
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = self.P(x, cl, h)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax

