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

    def stratified_partition(self, k, df = None):
        if df is None: df = self.df
        p = [[] for i in range(k)]
        if self.classification:
            def class_partition(classdf, p, c):
                n = classdf.shape[0]
                (q, r) = (n // k, n % k)
                j = 0
                for i in range(k):
                    z = (i + c) % k
                    p[z] = p[z] + [classdf.at[x, 'index'] for x in range(j, j + q + int(i < r))]
                    j += q + int(i < r)
                return (p, c + r)
            c = 0
            for cl in self.classes:
                classdf = df[df['Target'] == cl].reset_index()
                (p, c) = class_partition(classdf, p, c)
        else:
            sorted_df = df.sort_values(by=['Target']).reset_index()
            n = sorted_df.shape[0]
            (q, r) = (n // k, n % k)
            for i in range(k):
                p[i] = p[i] + [sorted_df.at[i + c * k, 'index'] for c in range(q + int(i < r))]
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

    def zero_one_loss(self, predicted, actual):
        if len(predicted) != len(actual):
            print("Not Equal Lengths")
        else:
            zero_one_sum = 0
            for i in range(len(predicted)):
                zero_one_sum += int(predicted[i] == actual[i])
            return zero_one_sum / len(predicted)

    def nnEstimator(self, train_set, k, sigma = None, epsilon = None, edit = False, test_set = None):
        if edit:
            def correctly_classified(i):
                if self.classification:
                    return self.nnEstimator(train_set.drop([i]),
                                            k, sigma = sigma)(self.value(train_set, i)) == train_set.at[i, 'Target']
                else:
                    return abs(self.nnEstimator(train_set.drop([i]),
                                                k, sigma = sigma)(self.value(train_set, i)) -
                                                train_set.at[i, 'Target']) < epsilon

            edited_neighbors = train_set.loc[train_set.index.map(correctly_classified)]
            old_pred = test_set.index.map(lambda j: self.nnEstimator(train_set, k,
                                                                     sigma = sigma)(self.value(test_set, j)))
            new_pred = test_set.index.map(lambda j: self.nnEstimator(edited_neighbors, k,
                                                                     sigma = sigma)(self.value(test_set, j)))
            actual = test_set['Target'].to_list()
            if self.zero_one_loss(old_pred, actual) > self.zero_one_loss(new_pred, actual) or \
                    train_set.shape[0] == edited_neighbors.shape[0]:
                neighbors = train_set
            else:
                return self.nnEstimator(edited_neighbors, k, sigma = sigma, epsilon = None,
                                        edit = True, test_set = test_set)
        else:
            neighbors = train_set
        def nn_estimate(x):
            distances = neighbors.index.to_series().map(lambda i: self.norm_2_distance(x, self.value(train_set, i)))
            dist_sorted = distances.sort_values().take(range(k))
            nn = dist_sorted.index
            if self.classification:
                w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
                (argmax, max_P) = (None, 0)
                for cl in self.classes:
                    y = w.at[cl] if cl in w.index else 0
                    if y > max_P:
                        argmax = cl
                        max_P = y
                return argmax
            else:
                def kernel(u):
                    return math.exp(-math.pow(u, 2) / sigma)
                v = dist_sorted.take(range(k)).map(kernel).to_numpy()
                r = nn.map(lambda i: train_set.at[i, 'Target'])
                return (v.dot(r))/v.sum()
        return nn_estimate

    def test(self):
        pred_df = pd.DataFrame(self.df.filter(items=range(50), axis=0).to_dict())
        p = self.stratified_partition(10, df = pred_df)
        predicted_classes = pd.Series(pred_df.shape[0] * [None])
        for i in range(10):
            (train_set, test_set) = self.training_test_sets(i, pred_df, p)
            nne = self.nnEstimator(train_set, 5, sigma = 1, epsilon = 2, edit = True, test_set = test_set)
            classes = pd.Series(p[i]).map(lambda j: nne(self.value(self.df, j)))
            predicted_classes.iloc[p[i]] = classes
        pred_df["Pred"] = predicted_classes
        pred_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))


