import sys
import os

import numpy as np
import operator 
from operator import itemgetter
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1,28*28)


def euc_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    # init
    def __init__(self, K=3):
        self.K = K
        self.X_train = X_train
        self.y_train = y_train
    # fit    
    def fit(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train
            
    # predict
    def predict(self, X_test):
        predictions = [] 
        for i in range(len(X_test)):
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])
            dist_sorted = dist.argsort()[:self.K]
            neigh_count = {}
            for idx in dist_sorted:
                if self.y_train[idx] in neigh_count:
                    neigh_count[self.y_train[idx]] += 1
                else:
                    neigh_count[self.y_train[idx]] = 1
            sorted_neigh_count = sorted(neigh_count.items(),    
            key=operator.itemgetter(1), reverse=True)
            predictions.append(sorted_neigh_count[0][0]) 
        return predictions