import os
import unittest

from sklearn.datasets.samples_generator import make_blobs
import clustering_evaluate

class test_kmeans_eval(unittest.TestCase):
    def test_kmeans(self):
        """
        Testing the function to run k-means++ clustering algo for various number of
        clusters on the given input_features
        """
        selected_n_clusters = 10
        X, y = make_blobs(n_samples=10000, centers=selected_n_clusters, n_features=300,
            cluster_std=0.001, random_state=0)

        fig, opt_clusters = clustering_evaluate.kmeans_viz_and_opt_clusters(X)
        
        self.assertAlmostEqual(selected_n_clusters, opt_clusters)