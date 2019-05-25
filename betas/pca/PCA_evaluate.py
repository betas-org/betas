"""
A python module that helps visuailize how the performance/evaluation metrics
differ as we vary the number of dimensions of the input data
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV


def sigmoid(features):
    """
    Sigmoid/Logistic function
    """
    return 1/(1+np.exp(-features))


def get_misclassification_error(features, labels, beta):
    """
    Function to compute the misclassification error
    from the predicted labels and the actual ones
    """
    prediction_probabilities = sigmoid(features.dot(beta))
    labels_pred = np.where(prediction_probabilities >= 0.5, 1, -1)
    return 1 - accuracy_score(labels, labels_pred)


def scale_features(train_features, test_features):
    """
    Function to scale the training as well as test data
    """
    x_scaler = StandardScaler().fit(train_features)
    train_features_std = x_scaler.transform(train_features)
    test_features_std = x_scaler.transform(test_features)
    return train_features_std, test_features_std


def get_pca_and_cv_results(train_features_std, test_features_std,
                           train_labels, dimension):
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

    lr_cv = LogisticRegressionCV(penalty='l2', tol=0.0001, cv=5,
                                 fit_intercept=False, intercept_scaling=0,
                                 solver='liblinear', max_iter=10000)
    lr_cv_model = lr_cv.fit(train_features_pca_std, train_labels)
    lambda_star = 1/(2*train_features_pca_std.shape[0]*lr_cv_model.C_[0])

    return train_features_pca_std, test_features_pca_std, lambda_star


def run_pca_across_dimensions(train_features_std, train_labels,
                              test_features_std, test_labels):
    """
    Function to run the PCA algorithms for various number of dimensions and
    evaluate each one based on the computed missclassification error values
    Inputs:
    - train_features_std: standardized training data
    - train_labels: training labels
    - test_features_std: standardized test data
    - test_labels: test labels
    """
    upper_limit = min(train_features_std.shape[0], train_features_std.shape[1])

    if upper_limit <= 1:
        raise "Data set is not worthy of PCA application due to limited size"

    dimensions = []
    i = 1
    while i <= upper_limit:
        dimensions.append(i)
        i *= 2

    missclassification_err_train = np.zeros(len(dimensions))
    missclassification_err_test = np.zeros(len(dimensions))
    i = 0
    for dimension in dimensions:
        train_features_pca_std, test_features_pca_std, lambda_star = \
            get_pca_and_cv_results(train_features_std, test_features_std,
                                   train_labels, dimension)

        lr_model = LogisticRegression(
            penalty='l2', tol=0.001,
            C=1/(2*train_features_pca_std.shape[0]*lambda_star),
            fit_intercept=False, intercept_scaling=0, solver='liblinear',
            max_iter=1000).fit(train_features_pca_std, train_labels)

        beta_values = np.squeeze(np.asarray(lr_model.coef_))
        missclassification_err_train[i] = \
            get_misclassification_error(train_features_pca_std,
                                        train_labels, beta_values)

        missclassification_err_test[i] = \
            get_misclassification_error(test_features_pca_std,
                                        test_labels, beta_values)
        i += 1
    return missclassification_err_train, missclassification_err_test,\
        dimensions


def plot_pca_errors(missclassification_err_train, missclassification_err_test,
                    dimensions):
    """
    Function to plot the misclassification error values for the various
    PCA runs for different dimensions
    Inputs:
    - missclassification_err_train: error values from training data
    - missclassification_err_test: error values from test data
    - dimensions: list of different dimensions
    """
    fig, ax1 = plt.subplots(figsize=(35, 12))
    color = 'tab:red'
    ax1.set_xlabel('PCA Dimensions', fontsize=30)
    ax1.set_xticks(dimensions)
    ax1.set_ylabel('Training Misclassification Error',
                   fontsize=30, color=color)
    ax1.plot(dimensions, missclassification_err_train, color=color)
    ax1.tick_params(labelsize=15)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Test Misclassification Error', fontsize=30, color=color)
    ax2.plot(dimensions, missclassification_err_test, color=color)
    ax2.tick_params(labelsize=15)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return fig


def visualize_pca(train_features, train_labels, test_features, test_labels):
    """
    Function that would be called eventually on the training and test data
    to generate the analysis of all the error values from different PCA
    dimensions in order to decide the most suited number of dimensions for the
    given dataset
    """
    train_features_std, test_features_std = \
        scale_features(train_features, test_features)
    error_train, error_test, dimensions = \
        run_pca_across_dimensions(train_features_std, train_labels,
                                  test_features_std, test_labels)
    figure = plot_pca_errors(error_train, error_test, dimensions)
    return figure
