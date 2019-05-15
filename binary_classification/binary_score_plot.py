import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class binary_score_plot(object):
    '''
    Class to construct plots to analyze
    performance of binary classifiers.
    Mainly acts as a wrapper for existing
    metrics and plotting functions.

    Input:
        - scores: 1D numpy array of model scores to plot (defaults to empty array)
        - labels: 1D numpy array of model labels to plot (defaults to empty array)
    '''

    def __init__(self, scores=np.array([]), labels=np.array([])):
        self._scores = scores
        self._labels = labels

    def get_scores(self):
        '''
        Get model scores to plot
        Output:
            - 1D numpy array of model scores
        '''

        return self._scores

    def get_labels(self):
        '''
        Get model labels to plot
        Output:
            - 1D numpy array of model labels
        '''

        return self._labels

    def set_scores(self, scores):
        '''
        Set model scores to plot
        Input:
            - 1D numpy array of model scores
        '''

        self._scores = scores

    def set_labels(self, labels):
        '''
        Set model labels to plot
        Input:
            - 1D numpy array of model labels
        '''

        self._labels = labels

    def plot_hist(self, bins=30):
        '''
        Plot two histograms: one of
        the actual binary labels
        and one of the model scores
        Input:
            - number of histogram bins to use
        '''

        labels = self.get_labels()
        scores = self.get_scores()

        plt.clf()
        plt.subplot(121)
        plt.hist(labels, range=(0, 1), bins=bins, histtype='step', lw=2, color='navy')
        plt.xlabel('Actual Labels')
        plt.ylabel('Frequency')

        plt.subplot(122)
        plt.hist(scores, range=(0, 1), bins=bins, histtype='step', lw=2, color='orange')
        plt.xlabel('Model Scores')

        plt.suptitle('Histograms of Actual Labels and Model Scores', fontsize=16)
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    def plot_pr_by_threshold(self):
        '''
        Plots model precision and recall
        by threshold, using matplotlib and seaborn
        allowing a user to visualize model performance
        at various thresholds
        '''

        labels = self.get_labels()
        scores = self.get_scores()

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        thresholds = np.append(thresholds, 1)

        plt.clf()
        plt.plot(thresholds, precision, color=sns.color_palette()[0])
        plt.plot(thresholds, recall, color=sns.color_palette()[1])

        leg = plt.legend(('Precision', 'Recall'), frameon=True)
        leg.get_frame().set_edgecolor('k')

        plt.xlabel('Threshold')
        plt.ylabel('Percent')
        plt.title('Precision and Recall by Model Threshold', fontsize=14)
        plt.show()

    def plot_roc(self):
        '''
        Plots the true positive rate vs. the false positive rate
        '''

        labels = self.get_labels()
        scores = self.get_scores()

        fpr, tpr, thresholds = roc_curve(labels, scores)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'fpr': pd.Series(fpr, index=i),
                            'tpr': pd.Series(tpr, index=i),
                            '1-fpr': pd.Series(1-fpr, index=i),
                            'tf': pd.Series(tpr - (1-fpr), index=i),
                            'thresholds': pd.Series(thresholds, index=i)})
        cutoff = float(roc.iloc[roc.tf.abs().argsort()[:1]]['thresholds'])
        roc_auc = auc(fpr, tpr)

        plt.clf()
        plt.plot(fpr, tpr, color='darkgreen', lw=2, label='AUC = %0.3f, Optimal Cutoff = %0.3f' %(roc_auc, cutoff))
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        plt.show()
