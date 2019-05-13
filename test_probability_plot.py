import numpy as np
from probability_plot import probability_plot

def main():
	'''
	Runs some sample probabilities and labels through
	the plot_jitter() function.
	'''

	probabilities = np.concatenate((np.random.uniform(low=0.5, high=1, size=300),
                 		   np.random.uniform(low=0, high=0.5, size=300),
                 		   np.random.uniform(low=0.5, high=1, size=30),
                 		   np.random.uniform(low=0, high=0.5, size=30)))
	actual_label = np.concatenate((np.ones(300), np.zeros(300), 
								   np.zeros(30), np.ones(30)))
	threshold = 0.5

	pj = probability_plot(probability=probabilities, actual_label=actual_label, threshold=threshold)
	pj.plot_jitter()

if __name__ == '__main__':
	main()