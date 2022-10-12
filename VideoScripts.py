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


def show_partitions(data):
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
# def show_regression():

# def show_classification():

# def editing():

# def cluster():


# def showPrefRegression():

# def showPrefClass():