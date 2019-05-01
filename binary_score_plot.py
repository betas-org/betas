import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve 

class binary_score_plot(object):
	'''
	Class to construct plots to analyze
	performance of binary classifiers

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

