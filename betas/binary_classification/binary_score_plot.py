import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class binary_score_plot(object):
    """
    Class to construct plots to analyze
    performance of binary classifiers.
    Mainly acts as a wrapper for existing
    metrics and plotting functions.

    Input:
        - scores: 1D numpy array of model scores to plot (defaults to empty
          array)
        - labels: 1D numpy array of model labels to plot (defaults to empty
          array)
    """

    def __init__(self, scores=np.array([]), labels=np.array([]), threshold=0.5):
        self._scores = scores
        self._labels = labels
        self._threshold = threshold

        pred_label = (scores > threshold) + 0
        cal_1 = pred_label + labels
        cal_2 = pred_label - labels
        df = pd.DataFrame({'scores': scores, 'actual_label': labels,
                           'pred_label': pred_label, 'group': ''})
        df.at[cal_1==2, 'group'] = 'TP'
        df.at[cal_1==0, 'group'] = 'TN'
        df.at[cal_2==1, 'group'] = 'FP'
        df.at[cal_2==-1, 'group'] = 'FN'
        df['position'] = df['actual_label'] + \
                         np.random.uniform(low=-0.3, high=0.3, size=len(df))
        self._df = df
        sns.set_style("white")

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
        plt.xlabel('Actual Label')
        plt.xticks([0,1])
        plt.subplot(222)
        sns.distplot(scores, bins=30, kde=False)
        plt.xlabel('Scores')
        plt.xticks([0,0.25,0.5,0.75,1])
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(223)
        sns.distplot(scores[labels==0], bins=30, kde=False)
        plt.xlabel('Actual Label = 0')
        plt.subplot(224)
        sns.distplot(scores[labels==1], bins=30, kde=False)
        plt.xlabel('Actual Label = 1')
        plt.suptitle('Histograms of Model Scores by Actual Label', fontsize=16)
        plt.show()

    def plot_jitter(self):
        """
        Make jitter plot
        """

        df = self.get_df()
        threshold = self.get_threshold()

        plt.clf()
        sns.scatterplot(x='position', y='scores', hue='group', s=10, alpha=0.8,
                        data=df)
        plt.hlines(y=threshold, xmin=-0.3, xmax=1.8, color='red')
        plt.xticks([0,1])
        plt.xlabel('Actual label')
        plt.ylabel('Scores')
        plt.suptitle('Scatterplot of Model Scores with Threshold = ' +
                      str(threshold), fontsize=16)
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

        plt.xlabel('Threshold')
        plt.ylabel('Percent')
        plt.suptitle('Precision and Recall by Model Threshold', fontsize=16)

        plt.show()

    def plot_roc(self):
        """
        Plots the true positive rate vs. the false positive rate
        """

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
        plt.plot(fpr, tpr, color='darkgreen', lw=2,
                 label='AUC = %0.3f' %(roc_auc))
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.suptitle('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right")

        plt.show()

    def optimal_threshold(self, by='roc'):
        """
        Calculate the optimal threshold by auc of either ROC or PR curve
        Input:
            - 1D numpy array of model scores
            - 1D numpy array of actual labels
            - Curve to use
        """

        labels = self.get_labels()
        scores = self.get_scores()

        if by == 'roc':
            fpr, tpr, thresholds = roc_curve(labels, scores)
            tnr = 1 - fpr
            tf = tpr - tnr
            optimal_threshold = round(thresholds[abs(tf).argsort()[0]], 2)
        else:
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            thresholds = np.append(thresholds, 1)
            tf = precision - recall
            optimal_threshold = round(thresholds[abs(tf).argsort()[0]], 2)
        return optimal_threshold
