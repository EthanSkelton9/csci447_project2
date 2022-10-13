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
import numpy as np
import time

class IanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue = None, classification = True):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue = replaceValue,
                         classification = classification)
        self.evaluator = self.classification_error if self.classification else self.mean_squared_error

    def tuner_split(self, df):
        random.seed(self.seed)
        tuner_index = random.sample(list(df.index), k=math.ceil(len(df.index) * .1))
        return (df.drop(tuner_index,  axis=0), df.filter(items = tuner_index, axis=0))

    def stratified_partition_Ian(self, k, df = None):
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
        return math.sqrt((x1.to_numpy() - x2.to_numpy()).dot(x1.to_numpy() - x2.to_numpy()))

    def zero_one_loss(self, predicted, actual):
        return pd.Series(zip(predicted, actual)).map(lambda pair: int(pair[0] == pair[1])).sum() / len(predicted)

    def p_Macro(self, predicted, actual):
        p_list = predicted.to_list()
        a_list = actual.to_list()
        CM = ConfusionMatrix(self.classes)
        for i in range(len(predicted)):
            CM.addOne(p_list[i], a_list[i])
        return CM.pmacro()

    def avg_Eval(self, f, g):
        def eval(predicted, actual):
            return (f(predicted, actual) + g(predicted, actual)) / 2
        return eval

    def classification_error(self, predicted, actual):
        return 1 - self.avg_Eval(self.zero_one_loss, self.p_Macro)(predicted, actual)

    def mean_squared_error(self, predicted, actual):
        p_vec = predicted.to_numpy()
        a_vec = actual.to_numpy()
        diff_vec = p_vec - a_vec
        return diff_vec.dot(diff_vec) / len(predicted)

    def kernel(self, x1, x2, sigma):
        return math.exp(-math.pow(self.norm_2_distance(x1, x2), 2) / sigma)

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
    def nnEstimator(self, train_set, k, sigma = None, epsilon = None, edit = False, test_set = None, start_time = None):
        if start_time is not None:
            if time.time() - start_time > 60 * 5:
                raise RuntimeError("Time is past 5 minutes.")
        train_set_values = train_set.index.to_series().map(lambda i: self.value(train_set, i))
        def nn_estimate_by_value(x):
            # print("--------------------")
            #print("New x to estimate")
            x_vec = x.to_numpy()
            #print("Computing Distances:")
            distances = train_set_values.map(lambda y: math.sqrt((x_vec - y.to_numpy()).dot(x_vec-y.to_numpy())))
            #print("Distances Computed")
            # print("Number of distances is {}".format(len(distances)))
            if k < len(distances):
                #print("Distances Number: {}".format(len(distances)))
                dist_sorted = distances.sort_values()
                #print("Sorted Distances Number: {}".format(len(dist_sorted)))
                dist_sorted = dist_sorted.take(range(k))
            else:
                dist_sorted = distances.sort_values()
            #print("Sorted Distances")
            nn = dist_sorted.index
            if self.classification:
                #print("Creating Train Frame")
                w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
                count = lambda cl: w.at[cl] if cl in w.index else 0
                #print("Returning Class")
                return rd(lambda cl1, cl2: cl1 if count(cl1) > count(cl2) else cl2, self.classes)
            else:
                def kernel(u):
                    return math.exp(-math.pow(u, 2) / sigma)
                v = dist_sorted.map(kernel).to_numpy()
                r = nn.map(lambda i: train_set.at[i, 'Target'])
                return v.dot(r)/v.sum()

        if edit:
            def correctly_classified(i):
                #print("Testing Neighbors on Themselves")
                target = self.nnEstimator(train_set.drop([i]), k, sigma=sigma, start_time=start_time)(self.value(train_set, i))
                #print("Target is {}".format(target))
                if self.classification:
                    return target == train_set.at[i, 'Target']
                else:
                    return abs(target - train_set.at[i, 'Target']) < epsilon

            yes = train_set.index.map(correctly_classified)
            no = yes.map(lambda y: not y)

            edited_neighbors = train_set.loc[train_set.index.map(correctly_classified)]
            print("Edited Out: {}".format(train_set.loc[no]))
            #print("Found Edited Neighbors")
            if train_set.shape[0] != edited_neighbors.shape[0]:
                #print("It is a smaller set.")
                pred_func = lambda set: self.comp(self.nnEstimator(set, k, sigma, start_time=start_time), pf(self.value, test_set))
                #print("Old Predictions")
                old_pred = test_set.index.map(pred_func(train_set))
                #print("New Predictions")
                new_pred = test_set.index.map(pred_func(edited_neighbors))
                actual = test_set['Target']
                if self.evaluator(old_pred, actual) >= self.evaluator(new_pred, actual):
                    #print("Recursively Edit Again")
                    return self.nnEstimator(edited_neighbors, k, sigma, epsilon, True, test_set, start_time)
        return nn_estimate_by_value


    def enn(self, train_set, k, sigma = None, epsilon = None, test_set = None, editedOut = [[]]):
        train_set_values = train_set.index.to_series().map(lambda i: self.value(train_set, i))
        def nn_estimate_by_value(x):
            # print("--------------------")
            #print("New x to estimate")
            x_vec = x.to_numpy()
            #print("Computing Distances:")
            distances = train_set_values.map(lambda y: math.sqrt((x_vec - y.to_numpy()).dot(x_vec-y.to_numpy())))
            #print("Distances Computed")
            # print("Number of distances is {}".format(len(distances)))
            if k < len(distances):
                #print("Distances Number: {}".format(len(distances)))
                dist_sorted = distances.sort_values()
                #print("Sorted Distances Number: {}".format(len(dist_sorted)))
                dist_sorted = dist_sorted.take(range(k))
            else:
                dist_sorted = distances.sort_values()
            #print("Sorted Distances")
            nn = dist_sorted.index
            if self.classification:
                #print("Creating Train Frame")
                w = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
                count = lambda cl: w.at[cl] if cl in w.index else 0
                #print("Returning Class")
                return (editedOut, nn, rd(lambda cl1, cl2: cl1 if count(cl1) > count(cl2) else cl2, self.classes))
            else:
                def kernel(u):
                    return math.exp(-math.pow(u, 2) / sigma)
                v = dist_sorted.map(kernel).to_numpy()
                r = nn.map(lambda i: train_set.at[i, 'Target'])
                return (editedOut, nn, (v.dot(r))/v.sum())


        def correctly_classified(i):
            #print("Testing Neighbors on Themselves")
            target = self.enn(train_set.drop([i]), k, sigma=sigma)(self.value(train_set, i))
            if self.classification:
                return target == train_set.at[i, 'Target']
            else:
                return abs(target - train_set.at[i, 'Target']) < epsilon


        (edited_neighbors, out) = (train_set.loc[train_set.index.map(correctly_classified)], train_set.loc[train_set.index.map(not_correct)])
        newEdits = editedOut + [out.index.to_list()]
        #print("Found Edited Neighbors")
        if train_set.shape[0] != edited_neighbors.shape[0]:
            pred_func = lambda set: self.comp(self.nne(set, k, sigma), pf(self.value, test_set))
            old_pred = test_set.index.map(pred_func(train_set))
            new_pred = test_set.index.map(pred_func(edited_neighbors))
            actual = test_set['Target']
            if self.evaluator(old_pred, actual) >= self.evaluator(new_pred, actual):
                #print("Recursively Edit Again")
                return self.nnEstimator(edited_neighbors, k, sigma, epsilon, test_set)
        return nn_estimate_by_value


    def getErrorDf_NN(self, tuner_set, train_dict, k_space, sigma_space, epsilon_space, appendCount = None, csv = None):
        def error(i):
            (f, k, sigma, epsilon) = my_space[i]
            print("Fold is {}. k is {}. sigma is {}. epsilon is {}.".format(f, k, sigma, epsilon))
            nne_for_hp = self.nnEstimator(train_dict[f], k, sigma, epsilon, edit=False, test_set=tuner_set, start_time=start_time)
            #print("Created Estimator.")
            #print("Size of tuning set is {}.".format(len(tuner_set.index)))
            pred_for_hp = tuner_set.index.map(self.comp(nne_for_hp, pf(self.value, tuner_set)))
            return self.evaluator(pred_for_hp, tuner_target)

        start_time = time.time()
        tuner_target = tuner_set['Target']
        folds = pd.Index(range(10))
        my_space = pd.Series(prod(folds, k_space, sigma_space, epsilon_space))
        df_size = len(my_space)
        if csv is None:
            cols = list(zip(*my_space))
            col_titles = ["Fold", "k", "sigma", "epsilon"]
            data = zip(col_titles, cols)
            error_df = pd.DataFrame(index=range(len(my_space)))
            for (title, col) in data:
                error_df[title] = col
            error_df["Error"] = df_size * [None]
            start = 0
            print("Table Created")
        else:
            error_df = pd.read_csv(csv, index_col=0)
            filtered = error_df["Error"].loc[pd.isnull(error_df["Error"])]
            start = filtered.index[0]
        end = df_size if appendCount is None else min(start + appendCount, df_size)
        error_df["Error"][start:end] = pd.Series(range(start, end)).map(error).values
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_NN.csv".format(str(self)))
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_NN_From_{}_To_{}.csv".format(str(self), start, end))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df

    def getAnalysisDf_NN(self, learning_set, train_dict, test_dict, error_df):
        analysis_df = pd.DataFrame(columns=["k", "sigma", "epsilon", "Error"], index=range(10))
        predicted_classes = pd.Series(index=learning_set.index)
        for i in range(10):
            fold_df = error_df.loc[lambda df: df['Fold'] == i]
            best_row = fold_df.loc[lambda df: df['Error'] == fold_df["Error"].min()].iloc[0]
            (best_k, best_sigma, best_epsilon) = (int(best_row["k"]), best_row["sigma"], best_row["epsilon"])
            nne = self.nnEstimator(train_dict[i], best_k, best_sigma, best_epsilon, edit=False, test_set=test_dict[i])
            pred_for_fold = pd.Series(test_dict[i].index).map(self.comp(nne, pf(self.value, test_dict[i])))
            test_target = test_dict[i]['Target']
            predicted_classes.loc[test_dict[i].index] = pred_for_fold.values
            analysis_df.loc[[i], ["k", "sigma", "epsilon", "Error"]] = \
                [best_k, best_sigma, best_epsilon, self.evaluator(pred_for_fold, test_target)]
        learning_set["Pred"] = predicted_classes
        learning_set.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred_NN.csv".format(str(self)))
        analysis_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Analysis_NN.csv".format(str(self)))


    def getErrorDf_ENN(self, tuner_set, train_dict, k_space, sigma_space, epsilon_space, appendCount = None, csv = None):
        def error(i):
            (f, k, sigma, epsilon) = my_space[i]
            print("Fold is {}. k is {}. sigma is {}. epsilon is {}.".format(f, k, sigma, epsilon))
            nne_for_hp = self.nnEstimator(train_dict[f], k, sigma, epsilon, edit=True, test_set=tuner_set, start_time=start_time)
            #print("Created Estimator.")
            #print("Size of tuning set is {}.".format(len(tuner_set.index)))
            pred_for_hp = tuner_set.index.map(self.comp(nne_for_hp, pf(self.value, tuner_set)))
            return self.evaluator(pred_for_hp, tuner_target)

        start_time = time.time()
        tuner_target = tuner_set['Target']
        folds = pd.Index(range(10))
        my_space = pd.Series(prod(folds, k_space, sigma_space, epsilon_space))
        df_size = len(my_space)
        if csv is None:
            cols = list(zip(*my_space))
            col_titles = ["Fold", "k", "sigma", "epsilon"]
            data = zip(col_titles, cols)
            error_df = pd.DataFrame(index=range(len(my_space)))
            for (title, col) in data:
                error_df[title] = col
            error_df["Error"] = df_size * [None]
            start = 0
            print("Table Created")
        else:
            error_df = pd.read_csv(csv, index_col=0)
            filtered = error_df["Error"].loc[pd.isnull(error_df["Error"])]
            start = filtered.index[0]
        end = df_size if appendCount is None else min(start + appendCount, df_size)
        error_df["Error"][start:end] = pd.Series(range(start, end)).map(error).values
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_ENN.csv".format(str(self)))
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_ENN_From_{}_To_{}.csv".format(str(self), start, end))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df

    def getAnalysisDf_ENN(self, learning_set, train_dict, test_dict, error_df):
        analysis_df = pd.DataFrame(columns=["k", "sigma", "epsilon", "Error"], index=range(10))
        predicted_classes = pd.Series(index=learning_set.index)
        for i in range(10):
            fold_df = error_df.loc[lambda df: df['Fold'] == i]
            best_row = fold_df.loc[lambda df: df['Error'] == fold_df["Error"].min()].iloc[0]
            (best_k, best_sigma, best_epsilon) = (int(best_row["k"]), best_row["sigma"], best_row["epsilon"])
            nne = self.nnEstimator(train_dict[i], best_k, best_sigma, best_epsilon, edit=True, test_set=test_dict[i])
            pred_for_fold = pd.Series(test_dict[i].index).map(self.comp(nne, pf(self.value, test_dict[i])))
            test_target = test_dict[i]['Target']
            predicted_classes.loc[test_dict[i].index] = pred_for_fold.values
            analysis_df.loc[[i], ["k", "sigma", "epsilon", "Error"]] = \
                [best_k, best_sigma, best_epsilon, self.evaluator(pred_for_fold, test_target)]
        learning_set["Pred"] = predicted_classes
        learning_set.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred_ENN.csv".format(str(self)))
        analysis_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Analysis_ENN.csv".format(str(self)))


    def Ian_test(self, k_space, head = None, sigma_space = [None], epsilon_space = [None], appendCount = None, seed = None):
        if head is None: head = self.df.shape[0]
        if seed is not None: self.seed = seed
        df = pd.DataFrame(self.df.filter(items = range(head), axis=0).to_dict())
        (learning_set, tuner_set) = self.tuner_split(df)
        p = self.stratified_partition_Ian(10, learning_set)
        (train_dict, test_dict) = self.training_test_dicts(learning_set, p)
        csv = None
        csv = os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_ENN.csv".format(str(self))
        error_df = pd.read_csv(csv, index_col=0)
        # error_df = self.getErrorDf_ENN(tuner_set, train_dict, k_space, sigma_space, epsilon_space, appendCount, csv)
        self.getAnalysisDf_ENN(learning_set, train_dict, test_dict, error_df)

    def latex_display(self):
        csv = os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_ClusEst.csv".format(str(self))
        error_df = pd.read_csv(csv, index_col=0).head(5)
        print(error_df.to_latex())

    def summary(self):
        pred_df = pd.read_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred_ENN.csv".format(str(self)), index_col=0)
        print(self.evaluator(pred_df['Pred'], pred_df['Target']))





