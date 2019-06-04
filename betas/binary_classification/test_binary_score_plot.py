import numpy as np
import pandas as pd
from binary_score_plot import binary_score_plot


def main():
    '''
    Runs some sample labels and scores through
    the plot_pr_by_threshold() function,
    the plot_hist() function,
    and the plot_roc() function
    of the binary_score_plot class.
    '''

    df = pd.read_csv('spam_output.csv')
    scores = df.scores
    labels = df.actual_label
    threshold = 0.55

    bsp = binary_score_plot(scores, labels, threshold)
    bsp.plot_hist()
    bsp.plot_jitter()
    bsp.plot_pr_by_threshold()
    bsp.plot_roc()

if __name__ == '__main__':
    main()
