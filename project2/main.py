from k_nearest import K_Nearest

'''
    Driver for k Nearest Neighbor    
'''
def main():
    ds = K_Nearest("hello")
    ds.TrainFile()  #train file and return with new hyperparameters to input


main() 
