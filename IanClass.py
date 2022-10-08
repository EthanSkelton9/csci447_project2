import pandas as pd
import os
import math
import random
from Learning import Learning
from ConfusionMatrix import ConfusionMatrix
import functools
from functools import partial as pf
from functools import reduce as rd

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
        train = partition[:j] + partition[j+1:]
        return (df.filter(items=train, axis=0).reset_index(), df.filter(items=partition[j], axis=0).reset_index())

    def norm_2_distance(self, x1, x2):
        return math.sqrt(pd.Series(self.features_ohe).map(lambda f: math.pow(x1[f] - x2[f], 2)).sum())

    def zero_one_loss(self, predicted, actual):
        return pd.Series(zip(predicted, actual)).map(lambda pair: int(pair[0] == pair[1])).sum() / len(predicted)

    def p_Macro(self, predicted, actual):
        CM = ConfusionMatrix(self.classes)
        for i in range(len(predicted)):
            CM.df.at[predicted[i], actual[i]] += int(predicted[i] == actual[i])
        return CM.pmacro()

    def avg_Eval(self, f, g):
        def eval(predicted, actual):
            return (f(predicted, actual) + g(predicted, actual)) / 2
        return eval

    def nnEstimator(self, train_set, k, sigma = None, epsilon = None, edit = False, test_set = None):
        def nn_estimate_by_value(x):
            distances = train_set.index.to_series().map(lambda i: self.norm_2_distance(x, self.value(train_set, i)))
            dist_sorted = distances.sort_values().take(range(k))
            nn = dist_sorted.index
            if self.classification:
                w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
                count = lambda cl: w.at[cl] if cl in w.index else 0
                return rd(lambda cl1, cl2: x if count(cl1) > count(cl2) else cl2, self.classes)
            else:
                def kernel(u):
                    return math.exp(-math.pow(u, 2) / sigma)
                v = dist_sorted.take(range(k)).map(kernel).to_numpy()
                r = nn.map(lambda i: train_set.at[i, 'Target'])
                return (v.dot(r))/v.sum()

        if edit:
            def correctly_classified(i):
                target = self.nnEstimator(train_set.drop([i]), k, sigma=sigma)(self.value(train_set, i))
                if self.classification:
                    return target == train_set.at[i, 'Target']
                else:
                    return abs(target - train_set.at[i, 'Target']) < epsilon

            edited_neighbors = train_set.loc[train_set.index.map(correctly_classified)]
            if train_set.shape[0] != edited_neighbors.shape[0]:
                pred_func = lambda set: self.comp(self.nnEstimator(set, k, sigma), pf(self.value, test_set))
                old_pred = test_set.index.map(pred_func(train_set))
                new_pred = test_set.index.map(pred_func(edited_neighbors))
                actual = test_set['Target'].to_list()
                evaluator = self.avg_Eval(self.zero_one_loss, self.p_Macro)
                if evaluator(old_pred, actual) <= evaluator(new_pred, actual):
                    return self.nnEstimator(edited_neighbors, k, sigma, epsilon, True, test_set)
                else:
                    return nn_estimate_by_value
            else:
                return nn_estimate_by_value
        else:
            return nn_estimate_by_value

    def test(self):
        pred_df = pd.DataFrame(self.df.filter(items=range(50), axis=0).to_dict())
        p = self.stratified_partition(10, df = pred_df)
        predicted_classes = pd.Series(pred_df.shape[0] * [None])
        for i in range(10):
            (train_set, test_set) = self.training_test_sets(i, pred_df, p)
            nne = self.nnEstimator(train_set, 5, sigma = 1, epsilon = 2, edit = True, test_set = test_set)
            classes = pd.Series(p[i]).map(self.comp(nne, pf(self.value, self.df)))
            predicted_classes.iloc[p[i]] = classes
        pred_df["Pred"] = predicted_classes
        pred_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))


