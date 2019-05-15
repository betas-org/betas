import numpy as np
from binary_score_plot import binary_score_plot

def main():
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
	labels = np.concatenate((np.ones(300), np.zeros(300), 
								   np.zeros(30), np.ones(30)))

	bsp = binary_score_plot(scores, labels)
	bsp.plot_hist()
	bsp.plot_pr_by_threshold()
	bsp.plot_roc()

if __name__ == '__main__':
	main()