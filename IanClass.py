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

    def tuner_split(self, df):
        tuner_index = random.sample(list(df.index), k=math.ceil(len(df.index) * .1))
        return (df.drop(tuner_index,  axis=0), df.filter(items = tuner_index, axis=0))

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
        train = rd(lambda l1, l2: l1 + l2, partition[:j] + partition[j+1:])
        return (df.filter(items=train, axis=0), df.filter(items=partition[j], axis=0))

    def norm_2_distance(self, x1, x2):
        return math.sqrt(pd.Series(self.features_ohe).map(lambda f: math.pow(x1[f] - x2[f], 2)).sum())

    def zero_one_loss(self, predicted, actual):
        return pd.Series(zip(predicted, actual)).map(lambda pair: int(pair[0] == pair[1])).sum() / len(predicted)

    def p_Macro(self, predicted, actual):
        CM = ConfusionMatrix(self.classes)
        for i in range(len(predicted)):
            CM.addOne(predicted[i], actual[i])
        return CM.pmacro()

    def avg_Eval(self, f, g):
        def eval(predicted, actual):
            return (f(predicted, actual) + g(predicted, actual)) / 2
        return eval

    def classification_error(self, predicted, actual):
        return 1 - self.avg_Eval(self.zero_one_loss, self.p_Macro)(predicted, actual)

    def mean_squared_error(self, predicted, actual):
        return pd.Series(zip(predicted, actual)).map(lambda pair: math.pow(pair[0] - pair[1], 2)).sum() / len(predicted)

    '''
        nnEstimator returns a function that predicts the target of an example using k nearest neighbors
        @param train_set - the set that we will use for our neighbors
        @param k - the number of neighbors we will use to predict an example
        @param sigma - the band width only used in regression sets
        @param epsilon - the max tolerance used to determine if two regression examples have the same target for editing
        @param edit - determines whether to use edited nearest neighbors or not
        @param test_set - test set used to determine whether the edited neighbors improves performance
        @return function that takes @param example x and returns predicted class or target value
    '''
    def nnEstimator(self, train_set, k, sigma = None, epsilon = None, edit = False, test_set = None):
        def nn_estimate_by_value(x):
            distances = train_set.index.to_series().map(lambda i: self.norm_2_distance(x, self.value(train_set, i)))
            dist_sorted = distances.sort_values().take(range(k))
            nn = dist_sorted.index
            if self.classification:
                w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
                count = lambda cl: w.at[cl] if cl in w.index else 0
                return rd(lambda cl1, cl2: cl1 if count(cl1) > count(cl2) else cl2, self.classes)
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
                evaluator = self.classification_error if self.classification else self.mean_squared_error
                if evaluator(old_pred, actual) >= evaluator(new_pred, actual):
                    return self.nnEstimator(edited_neighbors, k, sigma, epsilon, True, test_set)
        return nn_estimate_by_value

    def test(self):
        df = pd.DataFrame(self.df.filter(items = range(40), axis=0).to_dict())
        (learning_set, tuner_set) = self.tuner_split(df)
        tuner_target = tuner_set['Target'].to_list()
        evaluator = self.classification_error if self.classification else self.mean_squared_error
        analysis_df = pd.DataFrame(columns = ["Best k", "Error"], index = range(10))
        p = self.stratified_partition(10, df = learning_set)
        predicted_classes = pd.Series(index = learning_set.index)
        k_space = pd.Index(range(5,10))
        for i in range(10):
            (train_set, test_set) = self.training_test_sets(i, learning_set, p)
            error = pd.Series(data = k_space, index = k_space, name = "Error")
            for k in k_space:
                nne_for_hp = self.nnEstimator(train_set, k, sigma = 1, epsilon = 2, edit = True, test_set = tuner_set)
                pred_for_hp = tuner_set.index.map(self.comp(nne_for_hp, pf(self.value, tuner_set)))
                error[k] = evaluator(pred_for_hp, tuner_target)
                error.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_Fold_{}.csv".format(str(self), i))
            best_hp = rd(lambda t1, t2: t1 if t1[1] <= t2[1] else t2, error.items())[0]
            nne = self.nnEstimator(train_set, best_hp, sigma = 1, epsilon = 2, edit = True, test_set = test_set)
            pred_for_fold = pd.Series(p[i]).map(self.comp(nne, pf(self.value, test_set)))
            test_target = test_set['Target'].to_list()
            predicted_classes.loc[p[i]] = pred_for_fold.values
            analysis_df.loc[[i], ["Best k", "Error"]] = [best_hp, evaluator(pred_for_fold, test_target)]
        learning_set["Pred"] = predicted_classes
        learning_set.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))
        analysis_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Analysis.csv".format(str(self)))


