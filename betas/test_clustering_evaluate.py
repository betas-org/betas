"""
Unit tests for clustering.py
"""

import unittest
from sklearn.datasets.samples_generator import make_blobs
import clustering_evaluate

class TestKmeansEval(unittest.TestCase):
    """
    Testing the function to run k-means++ clustering algo
    """
    def test_kmeans(self):
        """
        Testing the function to run k-means++ clustering algo for
        various number of clusters on the given input_features
        """
        selected_n_clusters = 10
        x_train, _y_train = make_blobs(n_samples=10000,
                                       centers=selected_n_clusters,
                                       n_features=300,
                                       cluster_std=0.001,
                                       random_state=0)

        opt_clusters = \
            clustering_evaluate.kmeans_viz_and_opt_clusters(x_train,
                                                            plot_figure=False)
        self.assertTrue(selected_n_clusters == opt_clusters)
