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
        tuner_index = random.sample(self.df.index, k=math.ceil(len(self.index) * .1))
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
        return (df.filter(items=train, axis=0).reset_index(), df.filter(items=test, axis=0).reset_index())

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

    def nnEstimator(self, train_set, k, editedn = False):
        if editedn:
            neighbors = train_set.loc[train_set.index.map(lambda i:
                                                          self.nnEstimator(train_set.drop([i]), k)(
                                                              self.value(train_set, i))
                                                          == train_set.at[i, 'Target'])]
        else:
            neighbors = train_set
        def nearestneighbor(x):
            distances = neighbors.index.map(lambda i: self.norm_2_distance(x, self.value(train_set, i)))
            nn = train_set.filter(items =
                                  distances.sort_values(return_indexer=True)[1].take(range(k)), axis=0).groupby(by =
                                    ['Target'])['Target'].agg('count')
            (argmax, max_P) = (None, 0)
            for cl in self.classes:
                y = nn.at[cl] / k if cl in nn.index else 0
                if y > max_P:
                    argmax = cl
                    max_P = y
            return argmax
        return nearestneighbor

    def test(self):
        p = self.stratified_partition(10)
        pred_df = pd.DataFrame(self.df.to_dict())
        predicted_classes = pd.Series(self.df.shape[0] * [None])
        for i in range(len(p)):
            (train_set, test_set) = self.training_test_sets(i, self.df, p)
            nn = self.nnEstimator(train_set, 4, True)
            classes = pd.Series(p[i]).map(lambda j: nn(self.value(self.df, j)))
            predicted_classes.iloc[p[i]] = classes
        pred_df["Pred"] = predicted_classes
        pred_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))


