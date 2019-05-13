import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class probability_plot(object):
	'''
	Class to construct plots to
	visualize the predicted probabilities
	of binary classifiers.
	Input:
        - probability: 1D numpy array of predicted probabilities to plot (defaults to empty array)
        - actual_label: 1D numpy array of actual labels (0 and 1), usually the y variable of test data 
		  (defaults to empty array)
		- threshold: a float to indicate at which point the probabilities should be seperated
		  (defaults to 0.5)
	'''

	def __init__(self, probability=np.array([]), actual_label=np.array([]), threshold=0.5):
		self._probability = probability
		self._actual_label = actual_label
		self._threshold = threshold

	def plot_jitter(self):
		'''
		Make jitter plot
		'''
		probability = self._probability
		actual_label = self._actual_label
		threshold = self._threshold

		pred_label = (probability > threshold) + 0
		cal_1 = pred_label + actual_label
		cal_2 = pred_label - actual_label
		df = pd.DataFrame({'probability': probability, 'actual_label': actual_label, 
						   'pred_label': pred_label, 'group': ''})
		df.at[cal_1==2, 'group'] = 'TP'
		df.at[cal_1==0, 'group'] = 'TN'
		df.at[cal_2==1, 'group'] = 'FP'
		df.at[cal_2==-1, 'group'] = 'FN'
		spread = 0.3
		df['position'] = df['actual_label'] + np.random.uniform(low=-spread, high=spread, size=len(df))
		TP = df[['probability','position']][df.group == 'TP']
		TN = df[['probability','position']][df.group == 'TN']
		FP = df[['probability','position']][df.group == 'FP']
		FN = df[['probability','position']][df.group == 'FN']

		plt.scatter(TP['position'], TP['probability'], label='TP', s=5)
		plt.scatter(TN['position'], TN['probability'], c='yellowgreen', label='TN', s=5)
		plt.scatter(FP['position'], FP['probability'], c='orchid', label='FP', s=5)
		plt.scatter(FN['position'], FN['probability'], c='red', label='FN', s=5)
		plt.hlines(y=0.5, xmin=-0.5, xmax=2, colors='red', linestyles='dashed')
		plt.legend(loc='upper right')
		plt.xticks([0,1])
		plt.xlabel('Actual label')
		plt.ylabel('Probability')
		plt.show()




















