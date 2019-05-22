"""
A python module that helps visuailize how the performance/evaluation metrics
differ as we vary the number of dimensions of the input data
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_misclassification_error(X, Y, Beta):
    Prediction_probabilities = sigmoid(X.dot(Beta))
    Y_pred = np.where(Prediction_probabilities>=0.5,1,-1)
    return 1 - accuracy_score(Y,Y_pred)


def ScaleFeatures(train_features, test_features):
    X_scaler = StandardScaler().fit(train_features)
    train_features_std = X_scaler.transform(train_features)
    test_features_std = X_scaler.transform(test_features)
    return train_features_std, test_features_std

def get_pca_and_cv_results(train_features_std, test_features_std, train_labels, dimension):
    """
    Function to run the PCA on the training data and then use it to find
    optimal lambda from the logistic regression cross validation function
    Inputs:
    - train_features_std: standardized training data
    - test_features_std: standardized test data
    - dimensions: number of components to run PCA on
    """
    pca = PCA(n_components=dimension)
    pca.fit(train_features_std)
    pca_components = pca.components_
    train_features_pca_std = train_features_std.dot(pca_components.T)
    test_features_pca_std = test_features_std.dot(pca_components.T)
    
    LR_CV = LogisticRegressionCV(penalty='l2', tol=0.0001, cv = 5,\
                             fit_intercept=False,intercept_scaling=0, \
                             solver='liblinear', max_iter=10000, multi_class='ovr')
    LR_CV_model = LR_CV.fit(train_features_pca_std, train_labels)
    Lambda_star = 1/(2*train_features_pca_std.shape[0]*LR_CV_model.C_[0])
    
    return train_features_pca_std, test_features_pca_std, Lambda_star



def Run_PCA_Across_Dimensions(train_features_std, train_labels, test_features_std, test_labels):
    dimensions = [2,4,8,16,32,64,128,256,512,1000]
    n, d = train_features_std.shape
    upper_limit = min(n, d)
    dimensions = []
    i = 2
    while i <= upper_limit:
        dimensions.append(i)
        i*=2

    missclassification_err_fast_images_train_opt = np.zeros(len(dimensions))
    missclassification_err_fast_images_test_opt = np.zeros(len(dimensions))
    i = 0
    for dimension in dimensions:
        train_features_pca_std, test_features_pca_std, Lambda_star = get_pca_and_cv_results(train_features_std,\
                                                                                           test_features_std,\
                                                                                           train_labels,
                                                                                           dimension)
        
        n = train_features_pca_std.shape[0]
        LR = LogisticRegression(
        penalty='l2', tol=0.001, C=1/(2*n*Lambda_star), fit_intercept=False,intercept_scaling=0, \
        solver='liblinear', max_iter=1000, multi_class='ovr')
        LR_model = LR.fit(train_features_pca_std, train_labels)

        Beta_Values_fast_images_opt_T = np.squeeze(np.asarray(LR_model.coef_))
        missclassification_err_fast_images_train_opt[i] = get_misclassification_error(train_features_pca_std,\
                                                                                      train_labels, \
                                                                                      Beta_Values_fast_images_opt_T)

        missclassification_err_fast_images_test_opt[i] =  get_misclassification_error(test_features_pca_std,\
                                                                                      test_labels, \
                                                                                      Beta_Values_fast_images_opt_T)
        i+=1
    return missclassification_err_fast_images_train_opt, missclassification_err_fast_images_test_opt, dimensions

def Plot_PCA_Errors(missclassification_err_fast_images_train_opt, missclassification_err_fast_images_test_opt, dimensions):
    fig, ax1 = plt.subplots(figsize=(35,12))
    color = 'tab:red'
    ax1.set_xlabel('PCA Dimensions', fontsize=30)
    ax1.set_xticks(dimensions)
    ax1.set_ylabel('Training Misclassification Error', fontsize=30,  color=color)
    ax1.plot(dimensions, missclassification_err_fast_images_train_opt, color=color)
    ax1.tick_params(labelsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Test Misclassification Error', fontsize=30, color=color)  # we already handled the x-label with ax1
    ax2.plot(dimensions, missclassification_err_fast_images_test_opt, color=color)
    ax2.tick_params(labelsize=15)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return fig

def Visualize_PCA(train_features, train_labels, test_features, test_labels):
    train_features_std, test_features_std = ScaleFeatures(train_features, test_features)
    error_train, error_test, dimensions = Run_PCA_Across_Dimensions(train_features_std, train_labels, test_features_std, test_labels)
    figure = Plot_PCA_Errors(error_train, error_test, dimensions)
    return figure









