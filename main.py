from k_nearest import K_Nearest
from SoyBean import SoyBean
import os

'''
    Driver for k Nearest Neighbor    
'''
def main_Ethan():
    ds = K_Nearest("hello")
    ds.TrainFile()  #train file and return with new hyperparameters to input

def main_Ian():
    D = SoyBean()
    p = D.stratified_partition(10)
    for i in range(10):
        D.df.filter(items = p[i], axis = 0).to_csv(os.getcwd() +
                                                   '\\' + str(D) + '\\' + "{}_{}.csv".format(str(D), i))


if __name__ == '__main__':
    main_Ian()
