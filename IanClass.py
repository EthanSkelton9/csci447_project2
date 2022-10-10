import pandas as pd
import os
import math
import random
from Learning import Learning
from ConfusionMatrix import ConfusionMatrix
import functools
from functools import partial as pf
from functools import reduce as rd
from itertools import product as prod

class IanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue = None, classification = True):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue = replaceValue,
                         classification = classification)
        self.evaluator = self.classification_error if self.classification else self.mean_squared_error

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
    def training_test_dicts(self, df, partition=None):
        if partition is None: partition = self.stratified_partition(10)
        train_dict = {}
        test_dict = {}
        for i in range(len(partition)):
            train_index = rd(lambda l1, l2: l1 + l2, partition[:i] + partition[i+1:])
            train_dict[i] = df.filter(items=train_index, axis=0)
            test_dict[i] = df.filter(items=partition[i], axis=0)
        return (train_dict, test_dict)

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
                if self.evaluator(old_pred, actual) >= self.evaluator(new_pred, actual):
                    return self.nnEstimator(edited_neighbors, k, sigma, epsilon, True, test_set)
        return nn_estimate_by_value


    def getErrorDf(self, tuner_set, train_dict, k_space, sigma_space, epsilon_space):
        def error(i):
            (f, k, sigma, epsilon) = my_space[i]
            nne_for_hp = self.nnEstimator(train_dict[f], k, sigma, epsilon, edit=True, test_set=tuner_set)
            pred_for_hp = tuner_set.index.map(self.comp(nne_for_hp, pf(self.value, tuner_set)))
            return self.evaluator(pred_for_hp, tuner_target)

        tuner_target = tuner_set['Target'].to_list()
        folds = pd.Index(range(10))
        my_space = pd.Series(prod(folds, k_space, sigma_space, epsilon_space))
        cols = list(zip(*my_space))
        col_titles = ["Fold", "k", "sigma", "epsilon"]
        data = zip(col_titles, cols)
        error_df = pd.DataFrame(index=range(len(my_space)))
        for (title, col) in data:
            error_df[title] = col
        error_df["Error"] = pd.Series(range(len(my_space))).map(error)
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error.csv".format(str(self)))
        return error_df

    def getAnalysisDf(self, learning_set, train_dict, test_dict, error_df):
        analysis_df = pd.DataFrame(columns=["k", "sigma", "epsilon", "Error"], index=range(10))
        predicted_classes = pd.Series(index=learning_set.index)
        for i in range(10):
            fold_df = error_df.loc[lambda df: df['Fold'] == i]
            best_row = fold_df.loc[lambda df: df['Error'] == fold_df["Error"].min()].iloc[0]
            (best_k, best_sigma, best_epsilon) = (int(best_row["k"]), best_row["sigma"], best_row["epsilon"])
            nne = self.nnEstimator(train_dict[i], best_k, best_sigma, best_epsilon, edit=True, test_set=test_dict[i])
            pred_for_fold = pd.Series(test_dict[i].index).map(self.comp(nne, pf(self.value, test_dict[i])))
            test_target = test_dict[i]['Target'].to_list()
            predicted_classes.loc[test_dict[i].index] = pred_for_fold.values
            analysis_df.loc[[i], ["k", "sigma", "epsilon", "Error"]] = \
                [best_k, best_sigma, best_epsilon, self.evaluator(pred_for_fold, test_target)]
        learning_set["Pred"] = predicted_classes
        learning_set.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))
        analysis_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Analysis.csv".format(str(self)))


    def test(self, k_space, head = None, sigma_space = [None], epsilon_space = [None]):
        if head is None: head = self.df.shape[0]
        df = pd.DataFrame(self.df.filter(items = range(head), axis=0).to_dict())
        (learning_set, tuner_set) = self.tuner_split(df)
        p = self.stratified_partition(10, df = learning_set)
        (train_dict, test_dict) = self.training_test_dicts(learning_set, p)
        error_df = self.getErrorDf(tuner_set, train_dict, k_space, sigma_space, epsilon_space)
        self.getAnalysisDf(learning_set, train_dict, test_dict, error_df)






