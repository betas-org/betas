'''
This module contains core functions of betas
'''

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def classify(scores, labels, threshold=0.5):
    '''
    Convert model scores to groups based on threshold and actual labels
    Input:
        - 1D numpy array of model scores
        - 1D numpy array of actual labels
        - threshold to use
    '''
    pred_label = (scores > threshold) + 0
    cal_1 = pred_label + labels
    cal_2 = pred_label - labels
    result = pd.DataFrame({'scores': scores, 'actual_label': labels,
                           'pred_label': pred_label, 'group': ''})
    result.at[cal_1 == 2, 'group'] = 'TP'
    result.at[cal_1 == 0, 'group'] = 'TN'
    result.at[cal_2 == 1, 'group'] = 'FP'
    result.at[cal_2 == -1, 'group'] = 'FN'
    np.random.seed(0)
    noise = np.random.uniform(low=-0.3, high=0.3, size=len(result))
    result['position'] = result['actual_label'] + noise
    return result
