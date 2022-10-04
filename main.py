from re import A
from Hardware import Hardware
from k_nearest import K_Nearest
from SoyBean import SoyBean
from Abalone import Abalone
from Glass import Glass
from ForestFires import ForestFires
from BreastCancer import BreastCancer
import os
import pandas as pd

'''
    Driver for k Nearest Neighbor    
'''
def main_Ethan():

    #Read in datasets
    F = ForestFires()
    G = Glass()
    H = Hardware()
    B = BreastCancer()

    F.TrainFile()


    # #hyperparameters:
    # #a
    # #b
    # datasets = []
    # datasets.append(Glass().Learning())

    # for ds in datasets:
    #     ds.split()
    #     #for loop that goes through split dataset testing on each 10%
    #     k = K_Nearest()
    #     train = k.TrainFile(ds)
    #     test = k.Test_file(ds, train)
    #     #change hyperparameters for next dataset
    #     a = test.a


    
def main_Ian():
    def f1():
        D = SoyBean()
        p = D.stratified_partition(10)
        D.training_test_sets(0, D.df, p)
        x = D.value(D.df, 0)
        print(D.train_set.index)
        print(D.nearestneighbors_naive(x, 5))
    def f2():
        D = SoyBean()
        p = D.stratified_partition(10)
        D.training_test_sets(0, D.df, p)
        pred_df = pd.DataFrame(D.df.to_dict())
        predicted_classes = []
        for i in range(D.df.shape[0]):
            if i in p[0]:
                predicted = D.nearestneighborEstimator(D.value(D.df, i), 5)
                actual = D.df.at[i, 'Target']
                predicted_classes.append(predicted)
            else:
                predicted_classes.append(None)
        pred_df["Pred"] = predicted_classes
        pred_df.to_csv(os.getcwd() + '\\' + str(D) + '\\' + "{}_Pred.csv".format(str(D)))
    def f3():
        D = Abalone()
        x1 = D.value(D.df, 4)
        x2 = D.value(D.df, 7)
        print(D.norm_2_distance(x1, x2))
    f2()




if __name__ == '__main__':
    # main_Ian()
    main_Ethan()
