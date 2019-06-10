"""
Module to retrieve the datasets and run some pre-processing.
Data is for binary classification.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def preprocess_data(X, y, test_size=0.25):
    """
    Pre-processes the input data by standardizing it and 
    splitting it into a train and test sets
    Input:
     X: input features/data to be pre-processed
     y: input labels
     test_size: fraction of the data to be split into the
                test set. 25% by default.
    Output:
     X_train: Standardized training feature set
     X_test: Standardized test feature set
     y_train: Training label set
     y_test: Test label set
    """
    X_std = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def get_spam_data():
    """
    Function to retrieve the spam dataset given in the ESL book's
    resources. First 57 columns are used as predictors, while the
    last/58th column is the binary response variable.
    """
    spam_data = pd.read_csv("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data",\
	                    delim_whitespace=True,header=None)
    y = spam_data.iloc[:,57]

    # This is done to ensure we have +1/-1 labels as that's 
    # what we use in the objective value function
    y = y.replace(0,-1)
    y = np.array(y)
    X = spam_data.iloc[:,0:57]
    return preprocess_data(X, y)


def get_two_class_data(X, y, class1, class2):
    """
    Helper function that retrieves only two classes, which are input
    from a multi-class dataset.
    Inputs:
     X: input features/data 
     y: input labels
     class1: one of the two classes to be retrieved
     class2: one of the two classes to be retrieved
    Outputs:
     X: input features/data with two classes
     y: input labels with two classes
    """
    selected_indices = np.where((y == class1)|(y == class2))
    y_curr = y[selected_indices]
    X_curr = X[selected_indices]
    y_curr[y_curr==class1] = -1
    y_curr[y_curr==class2] = 1
    return X_curr, y_curr

def get_digits_data_binary():
    """
    Function to retrieve the digits dataset provided by sklearn.
    Since the actual dataset has 10 classes, we call a helper 
    function to only retrieve two of those classes
    """
    digits = load_digits()
    # classes refers to the labels - we will only use two here for binary classification
    target = digits.target_names
    classes = np.random.randint(target.min(), target.max()+1, size=2)
    X, y = get_two_class_data(digits.data, digits.target, classes[0], classes[1])
    return preprocess_data(X, y)

