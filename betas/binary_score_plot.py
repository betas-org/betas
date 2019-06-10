"""
This module is designed for desmostraing binary score plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc


class BinaryScorePlot(object):
    """
    Class to construct plots to analyze performance of binary classifiers.
    Mainly acts as a wrapper for existing metrics and plotting functions.

    Input:
        - scores: 1D numpy array of model scores to plot (defaults to empty
          array)
        - labels: 1D numpy array of model labels to plot (defaults to empty
          array)
    """

    def __init__(self, scores=np.array([]), labels=np.array([]), threshold=.5):
        self._scores = scores
        self._labels = labels
        self._threshold = threshold

        pred_label = (scores > threshold) + 0
        cal_1 = pred_label + labels
        cal_2 = pred_label - labels
        df_curr = pd.DataFrame({'scores': scores, 'actual_label': labels,
                                'pred_label': pred_label, 'group': ''})
        df_curr.at[cal_1 == 2, 'group'] = 'TP'
        df_curr.at[cal_1 == 0, 'group'] = 'TN'
        df_curr.at[cal_2 == 1, 'group'] = 'FP'
        df_curr.at[cal_2 == -1, 'group'] = 'FN'
        df_curr['position'] = df_curr['actual_label'] + \
            np.random.uniform(low=-0.3, high=0.3, size=len(df_curr))
        self._df = df_curr
        sns.set_style("darkgrid")

    def get_scores(self):
        """
        Get model scores to plot
        Output:
            - 1D numpy array of model scores
        """

        return self._scores

    def get_labels(self):
        """
        Get model labels to plot
        Output:
            - 1D numpy array of model labels
        """

        return self._labels

    def get_df(self):
        """
        Get data for scatterplot
        Output:
            - pandas dataframe
        """

        return self._df

    def get_threshold(self):
        """
        Output:
            - threshold
        """

        return self._threshold

    def set_scores(self, scores):
        """
        Set model scores to plot
        Input:
            - 1D numpy array of model scores
        """

        self._scores = scores

    def set_labels(self, labels):
        """
        Set model labels to plot
        Input:
            - 1D numpy array of model labels
        """

        self._labels = labels

    def plot_hist(self, bins=30):
        """
        Plot two histograms: one of
        the actual binary labels
        and one of the model scores
        Input:
            - number of histogram bins to use
        """

        labels = self.get_labels()
        scores = self.get_scores()

        plt.clf()
        plt.subplot(221)
        sns.distplot(labels, kde=False)
        plt.xlabel('Actual Label', fontsize=12)
        plt.xticks([0, 1])
        plt.subplot(222)
        sns.distplot(scores, bins=bins, kde=False)
        plt.xlabel('Scores', fontsize=12)
        plt.xticks([0, 0.25, 0.5, 0.75, 1])
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(223)
        sns.distplot(scores[labels == 0], bins=bins, kde=False)
        plt.xlabel('Actual Label = 0', fontsize=12)
        plt.subplot(224)
        sns.distplot(scores[labels == 1], bins=bins, kde=False)
        plt.xlabel('Actual Label = 1', fontsize=12)
        plt.suptitle('Histograms of Model Scores by Actual Label', fontsize=16)
        plt.show()

    def plot_jitter(self):
        """
        Make jitter plot
        """

        df_curr = self.get_df()
        threshold = self.get_threshold()

        plt.clf()
        sns.scatterplot(x='position', y='scores', hue='group', s=10, alpha=0.8,
                        data=df_curr)
        plt.hlines(y=threshold, xmin=-0.3, xmax=1.8, color='red')
        plt.xticks([0, 1])
        plt.xlabel('Actual label', fontsize=14)
        plt.ylabel('Scores', fontsize=14)
        title = 'Scatterplot of Model Scores with Threshold = '
        title += str(threshold)
        plt.suptitle(title, fontsize=16)
        plt.show()

    def plot_pr_by_threshold(self):
        """
        Plots model precision and recall
        by threshold, using matplotlib and seaborn
        allowing a user to visualize model performance
        at various thresholds
        """

        labels = self.get_labels()
        scores = self.get_scores()

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        thresholds = np.append(thresholds, 1)

        plt.clf()
        plt.plot(thresholds, precision, color=sns.color_palette()[0])
        plt.plot(thresholds, recall, color=sns.color_palette()[1])

        leg = plt.legend(('Precision', 'Recall'), frameon=True)
        leg.get_frame().set_edgecolor('k')

        plt.xlabel('Threshold', fontsize=14)
        plt.ylabel('Percent', fontsize=14)
        plt.suptitle('Precision and Recall by Model Threshold', fontsize=16)

        plt.show()

    def plot_roc(self):
        """
        Plots the true positive rate vs. the false positive rate
        """

        labels = self.get_labels()
        scores = self.get_scores()
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.clf()
        plt.plot(fpr, tpr, color='darkgreen', lw=2,
                 label='AUC = %0.3f' % (roc_auc))
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.suptitle('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right")

        plt.show()

    def optimal_threshold(self, by_mode='roc'):
        """
        Calculate the optimal threshold by auc of either ROC or PR curve
        Input:
            - 1D numpy array of model scores
            - 1D numpy array of actual labels
            - Curve to use
        """

        labels = self.get_labels()
        scores = self.get_scores()

        if by_mode == 'roc':
            fpr, tpr, thresholds = roc_curve(labels, scores)
            tnr = 1 - fpr
            t_f = tpr - tnr
            optimal_threshold = round(thresholds[abs(t_f).argsort()[0]], 2)
        else:
            precision, recall, thresholds = precision_recall_curve(labels,
                                                                   scores)
            thresholds = np.append(thresholds, 1)
            t_f = precision - recall
            optimal_threshold = round(thresholds[abs(t_f).argsort()[0]], 2)
        return optimal_threshold
