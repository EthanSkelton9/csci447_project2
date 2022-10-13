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
import IanClass
import EthanClass


def show_partitions(data):
<<<<<<< HEAD
    p = data.stratified_partition_Ian(10)
    df = data.df
    print("The size of the data set is {}".format(df.shape[0]))
    train_dict, test_dict = data.training_test_dicts(df, p)
    for i in range(len(p)):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("The indices for fold {} is:".format(i))
        print(p[i])
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("The test set for fold {} is:".format(i))
        print(test_dict[i])
        print("The size of the test set is: {}".format(test_dict[i].shape[0]))
        print("=======================================")
        print("The training set for fold {} is:".format(i))
        print(train_dict[i])
        print("The size of the training set is: {}".format(train_dict[i].shape[0]))
        print("++++")
        print("We see that test size {} + train size {} = total size {}".format(test_dict[i].shape[0], train_dict[i].shape[0],df.shape[0]))
        print("++++")


def show_distance_and_kernel(data):
    df = data.df
    for i in range(5):
        print("-----------------------------------------------------------------------------------------")
        x1 = data.value(df, random.sample(list(df.index), 1)[0])
        print("Printing first value of dataframe:")
        print(x1)
        x2 = data.value(df, random.sample(list(df.index), 1)[0])
        print("==================================================")
        print("Printing second value of dataframe:".format(x2))
        print(x2)
        print("+++++++++++++++++++++")
        print("Printing distance of two datapoints of dataframe:")
        dist = data.norm_2_distance(x1, x2)
        print(dist)
        print("~~~~~~~~~~~~~~~~~~~~~~")
        print("Using the radial basis kernel using that distance for the input:")
        for sigma in [4, 5, 6]:
            print("~~~~~~")
            print("For sigma = {}".format(sigma))
            print("The output of the kernel is:")
            print(data.kernel(x1, x2, sigma))
        print("+++++++++++++++++++++")

def nn_classification(data):
    p = data.stratified_partition_Ian(10)
    df = data.df
    train_dict, test_dict = data.training_test_dicts(df, p)
    for i in range(3):
        print("--------------------------------------------------------------------")
        print("For fold {}".format(i))
        print("My training set is:")
        print(train_dict[i])
        x = data.value(df, test_dict[i].index[0])
        print("===============================")
        print("A value from my test set is:")
        print(x)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for k in [5, 6]:
            print("******************************")
            print("Using k = {}".format(k))
            nn, pred = data.nnEstimator(train_dict[i], k)(x)
            print("The indices of my k nearest neighbors are:")
            print(nn)
            print("++++++++++++++")
            print("The predicted class is {}".format(pred))

def nn_regression(data):
    p = data.stratified_partition_Ian(10)
    df = data.df
    train_dict, test_dict = data.training_test_dicts(df, p)
    for i in range(3):
        print("--------------------------------------------------------------------")
        print("For fold {}".format(i))
        print("My training set is:")
        print(train_dict[i])
        x = data.value(df, test_dict[i].index[0])
        print("===============================")
        print("A value from my test set is:")
        print(x)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for k in [5, 6]:
            for sigma in [4, 5]:
                print("******************************")
                print("Using k = {} and sigma = {}".format(k, sigma))
                nn, pred = data.nnEstimator(train_dict[i], k, sigma=sigma)(x)
                print("The indices of my k nearest neighbors are:")
                print(nn)
                print("++++++++++++++")
                print("The predicted class is {}".format(pred))

def enn(data):
    p = data.stratified_partition_Ian(10)
    df = data.df.head(40)
    train_dict, test_dict = data.training_test_dicts(df, p)
    for i in range(1):
        print("--------------------------------------------------------------------")
        print("For fold {}".format(i))
        print("My training set is:")
        print(train_dict[i])
        x = data.value(df, test_dict[i].index[0])
        print("===============================")
        print("A value from my test set is:")
        print(x)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for k in [15]:
            print("******************************")
            print("Using k = {}".format(k))
            enn = data.nnEstimator(train_dict[i], k, sigma = 8, epsilon = 2, edit=True, test_set=test_dict[i])
            pred = enn(x)
            print("The indices of my k nearest neighbors are:")
            print("++++++++++++++")
            print("The predicted class is {}".format(pred))

def display(data):
    print("For NN, the average error is:")
    pred_df = pd.read_csv(os.getcwd() + '\\' + str(data) + '\\' + "{}_Pred_NN.csv".format(str(data)), index_col=0)
    print(data.evaluator(pred_df['Pred'], pred_df['Target']))

    print("For ENN, the average error is:")
    pred_df = pd.read_csv(os.getcwd() + '\\' + str(data) + '\\' + "{}_Pred_ENN.csv".format(str(data)), index_col=0)
    print(data.evaluator(pred_df['Pred'], pred_df['Target']))








# def show_regression():
=======
    d = data.stratified_partition_Ian(10)
    print(d)

def show_distance(data):
    x1 = data.df.iloc[1]
    x2 =data.df.iloc[2]
    print("Printing first value of dataframe:")
    print(x1)
    print("Printing second value of dataframe:")
    print(x2)
    print("Printing distance of two datapoints of dataframe:")
    dist = data.norm_2_distance(x1, x2)
    print(dist)
    print("--------------------------------------")

def show_kernel(data, sigma):
    x1 = data.df.iloc[1]
    x2 =data.df.iloc[2]
    dist = data.norm_2_distance(x1, x2)
    print("Second point would have a weight of:")
    print(math.exp(-math.pow(dist, 2) / sigma))
    print("Since it is:")
    print(dist)
    print("Away from the first point")
>>>>>>> 11b6bb003572fea2ba3d6184cac394e3e60a7992

#def show_regression(data):
   #x1 = data.df.iloc[1] 
# def show_classification():

# def editing():

def cluster(data):
    print(data.df.iloc[1])
    d = data.k_means_cluster(4)
    f = d.loc["Date"]
    print(f)
    print(d)




# def showPrefRegression():

# def showPrefClass():