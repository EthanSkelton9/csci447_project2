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

class EthanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue = None, classification = True):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue = replaceValue,
                         classification = classification)

    def centroid(self, data, rand):
        avg = []
        if not data.empty:

        
            if rand:
                avg = data.sample().values.tolist()
                avg = avg[0]
            else:
                for col in data:
                    val = np.exp(data[col].sum() / len(data[col]))
                    avg.append(val)
            return avg
        else:
            return avg
        

    '''
    clusterSame returns whether two cluster list are the same
    @param nc - new cluster that was created from data
    @param c - cluster that we had

    @return True - if same
    @return False - if different
    '''
    def clusterSame(self, nc, c):
        nc.sort()
        c.sort()
        if nc == c :
            return True
        else:
            return False

    '''
    calcCluster - calculates new cluseter averages 

    '''

    def calcCluster(self, data, k):
        
        rand = False
        c = []
        for i in range(k):
            df = data.loc[data['cluster'] == i]
            c.append(self.centroid(df, rand)) #calculate the mean of the cluster
        return c

    def randCluster(self, data, k):
        c = []
        rand = True
        for i in range(k):
            c.append(self.centroid(data, rand))
        return c
    
    def d(self, data, centroid):
        distances = []
        for index, row in data.iterrows():
            r = []
            for i in range(len(row)):    
                r.append(row[i])
            d = []
            for r1, c1 in zip(r, centroid):
                item = r1 - c1
                d.append(item)
            distances.append(d)
        return distances


    def minimum(self, list, cData):
        c = 0
        min = 0
        for r in range(len(list[0])):
            for l in range(len(list)):
                total = 0
                for d in range(len(list[0][0])):
                    total += list[l][r][d]
                if abs(total) < abs(min) or min == 0:
                    min = total
                    c = l
            cData.iloc[r, -1] = c
        return cData

    '''
    k_means - calculates the clusters based on the mean 
    '''

    def k_means(self, k):
        data = self.df 
        cluster_same = True # initialize 

        new_data = data
        if "Target" in new_data:
            new_data.pop("Target")
        
        #initialize cluster list of centroids:
        cluster = []
        new_cluster = []
        cluster = self.randCluster(new_data, k)
     
        new_data['cluster'] = 0
        
        
        while cluster_same:
            
            if new_cluster != []: #set cluster to new set of cluster centers on a second go around
                cluster = new_cluster
            classList = []
            classA = []
            for u in cluster:
                classList.append(self.d(data,u))
            new_data = self.minimum(classList, new_data)
            # new_data['cluster'] = pd.Series(classA)
                
                
            #calculate new cluster
            new_cluster = self.calcCluster(new_data,k)
            cluster_same = self.clusterSame(new_cluster, cluster)
            
        
        return new_data['cluster']