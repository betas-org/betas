import numpy as np

from binary_score_plot import binary_score_plot


def main():
	'''
	Runs some sample labels and scores through
	the plot_pr_by_threshold function of the binary_score_plot
	'''

	labels = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1])
	scores = np.array([.78, 0.34, .98, 0.12, .77, .68, .65, .87, .13, .05, .03, .10, .14, .45, .78, .32, .99, .45])
	bsp = binary_score_plot(scores, labels)
	bsp.plot_pr_by_threshold()

if __name__ == '__main__':

	main()