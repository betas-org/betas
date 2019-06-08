"""
Unit tests for binary_score_plot.py
"""

import numpy as np
import pandas as pd
import unittest
from binary_score_plot import binary_score_plot
class TestBinaryScorePlot(unittest.TestCase):
    """
    Testing the plots in the binary_score_plot module
    """

    def test_threshold_assignment(self):
        """
        Testing the threshold assignment
        """

        df = pd.read_csv('spam_output.csv')
        # In virtual envi: Failure
        # FileNotFoundError: [Errno 2] File b'spam_output.csv' does not exist: b'spam_output.csv'
        scores = df.scores
        labels = df.actual_label
        threshold = 0.55

        bsp = binary_score_plot(scores, labels, threshold)
        assigned_threshold = bsp._threshold

        self.assertTrue(threshold == assigned_threshold)

    def test_optimal_threshold_range(self):
        """
        Testing the optimal threshold range
        """

        df = pd.read_csv('spam_output.csv')
        # In virtual envi: Failure
        # FileNotFoundError: [Errno 2] File b'spam_output.csv' does not exist: b'spam_output.csv'
        scores = df.scores
        labels = df.actual_label

        bsp = binary_score_plot(scores, labels)
        opt_thresh = bsp.optimal_threshold(by='roc')

        self.assertTrue(opt_thresh >= 0 and opt_thresh <= 1)

