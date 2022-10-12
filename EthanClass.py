import pandas as pd
import os
import math
import random
from Learning import Learning
import numpy as np

class EthanClass (Learning):
    def __init__(self, file, features, name, classLoc, replaceValue = None, classification = True):
        super().__init__(file=file, features=features, name=name, classLoc=classLoc, replaceValue = replaceValue,
                         classification = classification)

    def tuners(self):
        tuner_index = random.sample(self.df.index, k=math.ceil(len(self.index) * .1))
        self.tuners = self.df.filter(items = tuner_index, axis=0)
        self.learning_set = self.df.drop(tuner_index,  axis=0)

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
    def training_test_sets(self, j, df, partition=None):
        if partition is None: partition = self.stratified_partition(10)
        train = []
        for i in range(len(partition)):
            if j != i:
                train += partition[i]
            else:
                test = partition[i]
        return (df.filter(items=train, axis=0).reset_index(), df.filter(items=test, axis=0).reset_index())

    def norm_2_distance(self, x1, x2):
        d = 0
        for f_num in self.features_ohe:
            d += math.pow(x1[f_num] - x2[f_num], 2)
        return math.sqrt(d)


    def naiveEstimator(self, x, h):
        def P(self, x, cl, h):
            def kernel(u):
                return int(abs(u) < 1 / 2)
            p = 0
            for t in self.train_set[self.train_set['Target'] == cl].index:
                p += kernel(self.norm_2_distance(x, self.value(self.train_set, t)) / h)
            return p
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = P(x, cl, h)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax

    def kernelEstimator(self, x, h):
        def P(x, cl):
            def kernel(u):
                return math.exp(-math.pow(u, 2)/2) / math.sqrt(2 * math.pi)
            p = 0
            for t in self.train_set[self.train_set['Target'] == cl].index:
                p += kernel(self.norm_2_distance(x, self.value(self.train_set, t)) / h)
            return p
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = P(x, cl)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax



    def nearestneighborEstimator(self, train_set, x, k):
        def nearestneighbors_naive():
            distances = train_set.index.map(lambda i: self.norm_2_distance(x, self.value(train_set, i)))
            (_, indices) = distances.sort_values(return_indexer=True)
            return indices.take(range(k))
        nn = nearestneighbors_naive()
        nn = train_set.filter(items = nn, axis=0).groupby(by = ['Target'])['Target'].agg('count')
        def P(x, cl):
            if cl in nn.index:
                return nn.at[cl] / k
            else:
                return 0
        (argmax, max_P) = (None, 0)
        for cl in self.classes:
            y = P(x, cl)
            if y > max_P:
                argmax = cl
                max_P = y
        return argmax

    # def test(self):
    #     p = self.stratified_partition(10)
    #     pred_df = pd.DataFrame(self.df.to_dict())
    #     predicted_classes = pd.Series(self.df.shape[0] * [None])
    #     for i in range(len(p)):
    #         (train_set, test_set) = self.training_test_sets(i, self.df, p)
    #         classes = pd.Series(p[i]).map(lambda j: self.nearestneighborEstimator(train_set, self.value(self.df, j), 5))
    #         predicted_classes.iloc[p[i]] = classes
    #     pred_df["Pred"] = predicted_classes
    #     pred_df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_Pred.csv".format(str(self)))



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
    
    def dist(self, data, centroid):
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
                classList.append(self.dist(data,u))
            new_data = self.minimum(classList, new_data)
            # new_data['cluster'] = pd.Series(classA)
                
                
            #calculate new cluster
            new_cluster = self.calcCluster(new_data,k)
            cluster_same = self.clusterSame(new_cluster, cluster)
            
        
        return new_data['cluster']

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