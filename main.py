from re import A
from Hardware import Hardware
from k_nearest import K_Nearest
from SoyBean import SoyBean
from Abalone import Abalone
from Glass import Glass
from ForestFires import ForestFires
from BreastCancer import BreastCancer
import os

'''
    Driver for k Nearest Neighbor    
'''
def main_Ethan():

    #Read in datasets
    F = ForestFires()
    G = Glass()
    H = Hardware()
    B = BreastCancer()

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
        D = Abalone()
        p = D.stratified_partition(10)
        for i in range(10):
            print(p[i])
            D.df.filter(items =  p[i], axis=0).to_csv(
                os.getcwd() + '\\' + str(D) + '\\' + "{}_{}.csv".format(str(D), i))
    def f2():
        D = Abalone()
        print("Numerical: {}".format(D.features_numerical))
        print("Categorical: {}".format(D.features_categorical))
        D.df.to_csv("d.csv")
    def f3():
        D = Abalone()
        x1 = D.value(D.df, 4)
        x2 = D.value(D.df, 7)
        print(D.norm_2_distance(x1, x2))
    f3()




if __name__ == '__main__':
    main_Ian()
    # main_Ethan()
