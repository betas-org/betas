import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import random

def load_data():
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

    actual_label = np.concatenate((np.ones(300), np.zeros(300), np.zeros(30), np.ones(30)))
    return scores, actual_label

def classify(scores, actual_label, threshold=0.5, spread=0.3):
    pred_label = (scores > threshold) + 0
    cal_1 = pred_label + actual_label
    cal_2 = pred_label - actual_label
    df = pd.DataFrame({'scores': scores, 'actual_label': actual_label, 'pred_label': pred_label, 'group': ''})
    df.at[cal_1==2, 'group'] = 'TP'
    df.at[cal_1==0, 'group'] = 'TN'
    df.at[cal_2==1, 'group'] = 'FP'
    df.at[cal_2==-1, 'group'] = 'FN'
    np.random.seed(0)
    noise = np.random.uniform(low=-spread, high=spread, size=len(df))
    df['position'] = df['actual_label'] + noise
    return(df)
