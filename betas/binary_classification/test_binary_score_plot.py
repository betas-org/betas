"""
Unit tests for binary_score_plot.py
"""

import numpy as np
import pandas as pd
import unittest
# package build
from binary_classification.binary_score_plot import binary_score_plot

# local
#from binary_score_plot import binary_score_plot

# Test scores and labels, size = 20
SCORES = np.array([0.54792232, 0.96933133, 0.99987806, 0.71340985, 0.71342704,
                   0.61219246, 0.72836524, 0.61848749, 0.99380689, 0.81295423,
                   0.88095189, 0.48289547, 0.54198686, 0.80857878, 0.9975807 ,
                   0.93301802, 0.4716791 , 0.96753287, 0.6641115 , 0.9989481 ])
LABELS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

class TestBinaryScorePlot(unittest.TestCase):
    """
    Testing the plots in the binary_score_plot module
    """

    def test_threshold_assignment(self):
        """
        Testing the threshold assignment in constructor
        """
        threshold = 0.55

        bsp = binary_score_plot(SCORES, LABELS, threshold)
        assigned_threshold = bsp._threshold
        self.assertTrue(threshold == assigned_threshold)
    
    def test_set_scores(self):
        """
        Test scores assignment
        """
        bsp = binary_score_plot(SCORES, LABELS)
        TESTSCORES = np.zeros(20)
        bsp.set_scores(TESTSCORES)
        self.assertEqual(bsp._scores.all(), TESTSCORES.all())
    
    def test_set_labels(self):
        """
        Test labels assignment
        """
        bsp = binary_score_plot(SCORES, LABELS)
        TESTLABELS = np.zeros(20)
        bsp.set_labels(TESTLABELS)
        self.assertEqual(bsp._labels.all(), TESTLABELS.all())
    
    def test_plots(self):
        """
        Test four plots
        """
        bsp = binary_score_plot(SCORES, LABELS)
        bsp.plot_hist()
        bsp.plot_jitter()
        bsp.plot_pr_by_threshold()
        bsp.plot_roc()

    def test_optimal_threshold_range(self):
        """
        Testing the optimal threshold range, default
        """
        bsp = binary_score_plot(SCORES, LABELS)
        opt_thresh = bsp.optimal_threshold()
        self.assertTrue(opt_thresh >= 0 and opt_thresh <= 1)

    def test_optimal_threshold_range_roc(self):
        """
        Testing the optimal threshold range by roc
        """
        bsp = binary_score_plot(SCORES, LABELS)
        opt_thresh = bsp.optimal_threshold(by='roc')
        self.assertTrue(opt_thresh >= 0 and opt_thresh <= 1)

if __name__ == "__main__":
    SUITE = unittest.TestLoader().loadTestsFromTestCase(TestBinaryScorePlot)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
