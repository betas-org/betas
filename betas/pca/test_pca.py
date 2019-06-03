import os
import unittest

import numpy as np
import pca_evaluate
from sklearn.model_selection import train_test_split

class test_pca_eval(unittest.TestCase):
    def test_pca(self):
        """
        Testing the function to run the pca algo for various number of
        dimensions on the given dataset
        """
        #X1, X2 and X3 are data generated from different normal distributions and 
        #they're all appended to for the feature-set called 'X'
        X1 = np.random.normal(size=100)
        X1 = X1.reshape(len(X1),1)
        X = X1
        for i in range(2,11):
            X = np.concatenate((X, X1**i),axis=1)

        X2 = np.random.normal(loc=1,scale=0.1, size=100)
        X2 = X2.reshape(len(X2),1)
        for i in range(1,11):
            X = np.concatenate((X, X2**i),axis=1)
    
        X3 = np.random.normal(loc=2,scale=1.5, size=100)
        X3 = X3.reshape(len(X3),1)
        for i in range(1,11):
            X = np.concatenate((X, X3**i),axis=1)
        
        train_features, test_features, train_labels, test_labels = train_test_split(X, y, random_state=0)
        
        fig, optimal_dimensions = pca_evaluate.pca_viz_and_opt_dimensions(train_features, train_labels,\
                                                                          test_features, test_labels)
        self.assertAlmostEqual(1, optimal_dimensions)