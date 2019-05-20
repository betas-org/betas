import numpy as np
from probability_plot import probability_plot

def main():
    '''
    Runs some sample probabilities and labels through
    the plot_jitter() function.
    '''

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
    threshold = 0.5
    spread = 0.3

    p = probability_plot(scores=scores, actual_label=actual_label, threshold=threshold, spread=spread)
    p.plot_jitter()
    p.plot_hist()

if __name__ == '__main__':
    main()
