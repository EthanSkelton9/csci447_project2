from IanClass import IanClass
from EthanClass import EthanClass
from asyncio.windows_events import NULL
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

#instance based learning
class IBL (IanClass, EthanClass):
    def __init__(self, file, features, name, classLoc,replaceValue = None, classification=True):
        super().__init__(file = file, features = features, name = name, classLoc = classLoc,
                         replaceValue = replaceValue, classification = classification)


    def clusterEstimator(self, train_set, k, sigma = None):
        def estimate(i):
            clusters = self.k_means(k)
            cluster_label = clusters[i]
            cluster = clusters.loc[lambda df: df == cluster_label].index
            index_intersection = pd.Index(set(cluster).intersection(set(train_set.index)))
            cluster_train_set = train_set.filter(items = index_intersection, axis=0)
            nearest_neighbor_k = len(index_intersection)
            nne = self.nnEstimator(train_set=cluster_train_set, k=nearest_neighbor_k, sigma = sigma)
            return nne(self.value(self.df, i))
        return estimate


    def getErrorDf_ClusEst(self, tuner_set, train_dict, k_space, sigma_space, appendCount = None, csv = None):
        def error(i):
            (f, k, sigma, _) = my_space[i]
            pred_for_hp = tuner_set.index.map(self.clusterEstimator(train_dict[f], k, sigma))
            return self.evaluator(pred_for_hp, tuner_target)

        tuner_target = tuner_set['Target'].to_list()
        folds = pd.Index(range(10))
        my_space = pd.Series(prod(folds, k_space, sigma_space))
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
        else:
            error_df = pd.read_csv(csv, index_col=0)
            print("Error Column: {}".format(error_df["Error"]))
            filtered = error_df["Error"].loc[pd.isnull(error_df["Error"])]
            start = filtered.index[0]
            print("Start: {}".format(start))
        end = df_size if appendCount is None else min(start + appendCount, df_size)
        error_df["Error"][start:end] = pd.Series(range(start, end)).map(error).values
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_ClusEst.csv".format(str(self)))
        error_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_From_ClusEst_{}_To_{}.csv".format(str(self), start, end))
        return error_df

    def getAnalysisDf_ClusEst(self, learning_set, train_dict, test_dict, error_df):
        analysis_df = pd.DataFrame(columns=["k", "sigma", "Error"], index=range(10))
        predicted_classes = pd.Series(index=learning_set.index)
        for i in range(10):
            fold_df = error_df.loc[lambda df: df['Fold'] == i]
            best_row = fold_df.loc[lambda df: df['Error'] == fold_df["Error"].min()].iloc[0]
            (best_k, best_sigma) = (int(best_row["k"]), best_row["sigma"])
            # nne = self.nnEstimator(train_dict[i], best_k, best_sigma, best_epsilon, edit=True, test_set=test_dict[i])
            # pred_for_fold = pd.Series(test_dict[i].index).map(self.comp(nne, pf(self.value, test_dict[i])))
            test_target = test_dict[i]['Target'].to_list()
            predicted_classes.loc[test_dict[i].index] = pred_for_fold.values
            analysis_df.loc[[i], ["k", "sigma", "epsilon", "Error"]] = \
                [best_k, best_sigma, best_epsilon, self.evaluator(pred_for_fold, test_target)]
        learning_set["Pred"] = predicted_classes
        learning_set.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))
        analysis_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Analysis.csv".format(str(self)))


    def test(self, k_space, head = None, sigma_space = [None], epsilon_space = [None], appendCount = None, seed = None):
        if head is None: head = self.df.shape[0]
        if seed is not None: self.seed = seed
        df = pd.DataFrame(self.df.filter(items = range(head), axis=0).to_dict())
        (learning_set, tuner_set) = self.tuner_split(df)
        p = self.stratified_partition(10, df = learning_set)
        (train_dict, test_dict) = self.training_test_dicts(learning_set, p)
        # csv = os.getcwd() + '\\' + str(self) + '\\' + "{}_Error_ClusEst.csv".format(str(self))
        error_df = self.getErrorDf(tuner_set, train_dict, k_space, sigma_space, epsilon_space, appendCount)
        # self.getAnalysisDf(learning_set, train_dict, test_dict, error_df)





