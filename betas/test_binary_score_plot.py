"""
Unit tests for binary_score_plot.py
"""

import unittest
import numpy as np
from binary_score_plot import BinaryScorePlot


# Test scores and labels, size = 20
SCORES = np.array([0.54792232, 0.96933133, 0.99987806, 0.71340985, 0.71342704,
                   0.61219246, 0.72836524, 0.61848749, 0.99380689, 0.81295423,
                   0.88095189, 0.48289547, 0.54198686, 0.80857878, 0.9975807,
                   0.93301802, 0.4716791, 0.96753287, 0.6641115, 0.9989481])
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

        bsp = BinaryScorePlot(SCORES, LABELS, threshold)
        assigned_threshold = bsp.get_threshold()
        self.assertTrue(threshold == assigned_threshold)

    def test_set_scores(self):
        """
        Test scores assignment
        """
        bsp = BinaryScorePlot(SCORES, LABELS)
        test_scores = np.zeros(20)
        bsp.set_scores(test_scores)
        self.assertEqual(bsp.get_scores().all(), test_scores.all())

    def test_set_labels(self):
        """
        Test labels assignment
        """
        bsp = BinaryScorePlot(SCORES, LABELS)
        test_labels = np.zeros(20)
        bsp.set_labels(test_labels)
        self.assertEqual(bsp.get_labels().all(), test_labels.all())

    def test_plots(self):
        """
        Test four plots
        """
        bsp = BinaryScorePlot(SCORES, LABELS)
        bsp.plot_hist()
        bsp.plot_jitter()
        bsp.plot_pr_by_threshold()
        bsp.plot_roc()
        self.assertEqual(1, 1)

    def test_optimal_threshold_range_roc(self):
        """
        Testing the optimal threshold range by roc (default)
        """
        bsp = BinaryScorePlot(SCORES, LABELS)
        opt_thresh = bsp.optimal_threshold(by_mode='roc')
        self.assertTrue(opt_thresh >= 0)
        self.assertTrue(opt_thresh <= 1)

    def test_optimal_threshold_range(self):
        """
            Testing the optimal threshold range not by roc
            """
        bsp = BinaryScorePlot(SCORES, LABELS)
        opt_thresh = bsp.optimal_threshold(by_mode='other')
        self.assertTrue(opt_thresh >= 0)
        self.assertTrue(opt_thresh <= 1)


if __name__ == "__main__":
    SUITE = unittest.TestLoader().loadTestsFromTestCase(TestBinaryScorePlot)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
