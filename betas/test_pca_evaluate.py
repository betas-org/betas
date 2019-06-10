"""
Unit tests for pca.py
"""

import unittest
import numpy as np
from sklearn.model_selection import train_test_split
import pca_evaluate


class TestPcaEval(unittest.TestCase):
    """
    Testing the function to run the pca algo for various number of
    dimensions on the given dataset
    """
    def test_pca_plot(self):
        """
        features_1, features_2 and features_3 are data generated from
        different normal distributions and they're all appended to for the
        feature-set called 'features'
        """
        features_1 = np.random.normal(size=100)
        features_1 = features_1.reshape(len(features_1), 1)
        features = features_1

        for i in range(2, 11):
            features = np.concatenate((features, features_1**i), axis=1)

        features_2 = np.random.normal(loc=1, scale=0.1, size=100)
        features_2 = features_2.reshape(len(features_2), 1)
        for i in range(1, 11):
            features = np.concatenate((features, features_2**i), axis=1)

        features_3 = np.random.normal(loc=2, scale=1.5, size=100)
        features_3 = features_3.reshape(len(features_3), 1)
        for i in range(1, 11):
            features = np.concatenate((features, features_3**i), axis=1)

        labels = np.random.randint(0, 2, size=100)
        train_features, test_features, train_labels, test_labels =\
            train_test_split(features, labels, random_state=0)

        optimal_dimensions = pca_evaluate.pca_viz_and_opt_dimensions(
            train_features,
            train_labels,
            test_features,
            test_labels,
            plot_figure=False)

        self.assertTrue(optimal_dimensions >= 1)

    def test_pca_plot_figure(self):
        """
        features_1, features_2 and features_3 are data generated from
        different normal distributions and they're all appended to for the
        feature-set called 'features'
        """
        features_1 = np.random.normal(size=100)
        features_1 = features_1.reshape(len(features_1), 1)
        features = features_1

        for i in range(2, 11):
            features = np.concatenate((features, features_1**i), axis=1)

        features_2 = np.random.normal(loc=1, scale=0.1, size=100)
        features_2 = features_2.reshape(len(features_2), 1)
        for i in range(1, 11):
            features = np.concatenate((features, features_2**i), axis=1)

        features_3 = np.random.normal(loc=2, scale=1.5, size=100)
        features_3 = features_3.reshape(len(features_3), 1)
        for i in range(1, 11):
            features = np.concatenate((features, features_3**i), axis=1)

        labels = np.random.randint(0, 2, size=100)

        train_features, test_features, train_labels, test_labels =\
            train_test_split(features, labels, random_state=0)

        fig, optimal_dimensions = pca_evaluate.pca_viz_and_opt_dimensions(
            train_features,
            train_labels,
            test_features,
            test_labels,
            plot_figure=True)

        self.assertTrue(optimal_dimensions >= 1)


if __name__ == "__main__":
    SUITE = unittest.TestLoader().loadTestsFromTestCase(TestPcaEval)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
