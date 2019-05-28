'''
This module contains core functions of betas
'''

import numpy as np
import pandas as pd


def load_data():
    '''
    Create sample data to test the functionality of other implementations
    '''
    scores = np.concatenate((np.random.uniform(low=0.9, high=1, size=150),
                             np.random.uniform(low=0.8, high=0.9, size=75),
                             np.random.uniform(low=0.7, high=0.8, size=50),
                             np.random.uniform(low=0.5, high=0.7, size=25),
                             np.random.uniform(low=0, high=0.1, size=150),
                             np.random.uniform(low=0.1, high=0.2, size=75),
                             np.random.uniform(low=0.2, high=0.3, size=50),
                             np.random.uniform(low=0.3, high=0.5, size=25),
                             np.random.uniform(low=0.5, high=1, size=30),
                             np.random.uniform(low=0, high=0.5, size=30)))
    actual_label = np.concatenate((np.ones(300), np.zeros(300), np.zeros(30),
                                   np.ones(30)))
    return scores, actual_label


def classify(scores, actual_label, threshold=0.5):
    '''
    Convert model scores to groups based on threshold and actual labels
    Input:
        - 1D numpy array of model scores
        - 1D numpy array of actual labels
        - threshold to use
    '''
    pred_label = (scores > threshold) + 0
    cal_1 = pred_label + actual_label
    cal_2 = pred_label - actual_label
    result = pd.DataFrame({'scores': scores, 'actual_label': actual_label,
                           'pred_label': pred_label, 'group': ''})
    result.at[cal_1 == 2, 'group'] = 'TP'
    result.at[cal_1 == 0, 'group'] = 'TN'
    result.at[cal_2 == 1, 'group'] = 'FP'
    result.at[cal_2 == -1, 'group'] = 'FN'
    np.random.seed(0)
    noise = np.random.uniform(low=-0.3, high=0.3, size=len(result))
    result['position'] = result['actual_label'] + noise
    return result
