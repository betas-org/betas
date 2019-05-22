import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class probability_plot(object):
    '''
    Class to construct plots to
    visualize the predicted probabilities
    of binary classifiers.
    Input:
        - scores: 1D numpy array of predicted probabilities to plot (defaults to empty array)
        - actual_label: 1D numpy array of actual labels (0 and 1), usually the y variable of test data
      (defaults to empty array)
    - threshold: a float to indicate at which point the probabilities should be seperated
      (defaults to 0.5)
    '''

    def __init__(self, scores=np.array([]), actual_label=np.array([]), threshold=0.5, spread=0.3):
        self._scores = scores
        self._actual_label = actual_label
        self._threshold = threshold
        self._spread = spread

        pred_label = (scores > threshold) + 0
        cal_1 = pred_label + actual_label
        cal_2 = pred_label - actual_label
        df = pd.DataFrame({'scores': scores, 'actual_label': actual_label,
                           'pred_label': pred_label, 'group': ''})
        df.at[cal_1==2, 'group'] = 'TP'
        df.at[cal_1==0, 'group'] = 'TN'
        df.at[cal_2==1, 'group'] = 'FP'
        df.at[cal_2==-1, 'group'] = 'FN'
        df['position'] = df['actual_label'] + np.random.uniform(low=-spread, high=spread, size=len(df))
        self._df = df

    def plot_jitter(self):
        '''
        Make jitter plot
        '''
        scores = self._scores
        actual_label = self._actual_label
        threshold = self._threshold
        spread = self._spread
        df = self._df

        sns.scatterplot(x='position', y='scores', hue='group', s=20, data=df)
        plt.hlines(y=threshold, xmin=0-spread, xmax=1+spread, color='red')
        plt.legend(bbox_to_anchor=(1.05, 1), borderaxespad=0)
        plt.xticks([0,1])
        plt.xlabel('Actual label')
        plt.ylabel('Scores')
        plt.title('Scatterplot of Model Scores (Cutoff Point: 0.5)', fontsize=16)
        plt.show()

    def plot_hist(self):
        '''
        Make histogram of model scores
        '''
        scores = self._scores
        actual_label = self._actual_label
        threshold = self._threshold
        spread = self._spread
        df = self._df

        sns.set_style("white")
        plt.subplot(221)
        sns.distplot(df.actual_label, kde=False)
        plt.xlabel('Actual Label')
        plt.xticks([0,1])
        plt.subplot(222)
        sns.distplot(df.scores, bins=30, kde=False)
        plt.xlabel('Scores')
        plt.xticks([0,0.25,0.5,0.75,1])
        plt.suptitle('Histograms of Actual Labels and Model Scores', fontsize=16)
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(223)
        sns.distplot(df.scores[df.actual_label==0], bins=30, kde=False)
        plt.xlabel('Actual Label = 0')
        plt.subplot(224)
        sns.distplot(df.scores[df.actual_label==1], bins=30, kde=False)
        plt.xlabel('Actual Label = 1')
        plt.suptitle('Histograms of Model Scores by Actual Label', fontsize=16)
        plt.show()
