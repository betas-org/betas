"""
A python module that helps visualize the results of running k-means++
on various number of clusters for a given dataset to assess the
optimal number of clusters
"""

import sklearn.cluster
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def get_cluster_vector(n_samples):
    """
    Function to create the vector of various cluster lenghts to evaluate
    Input:
        - n_samples: total number of samples/examples in the dataset
    Output:
        - n_clusters_vector: vector of various cluster lenghts to evaluate
    """
    upper_limit = n_samples / 100
    n_clusters_vector = []
    i = 2
    while i < min(10, upper_limit):
        n_clusters_vector.append(i)
        i += 2
    while i < min(100, upper_limit):
        n_clusters_vector.append(i)
        i += 10
    while i < upper_limit:
        n_clusters_vector.append(i)
        i += 25
    n_clusters_vector.append(int(upper_limit))
    return n_clusters_vector


def get_cost_from_kmeans(n_clusters_vector, data):
    """
    Function to derive the objective values for k-means++ for the
    various values of number of clusters that are input
    Input:
        - n_clusters_vector: vector of various cluster lenghts to evaluate
        - X: input dataset/features
    Output:
     - objVals: The list of final values of the inertia criterion (sum
      of squared distances to the closest centroid for all observations
      in the training set) for all values of number of clusters
    """
    obj_vals = []
    for n_clusters in n_clusters_vector:
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters,
                                        init='k-means++')
        kmeans.fit(data)
        obj_vals.append(kmeans.inertia_)
    return obj_vals


def visualize_kmeans(n_clusters_vector, obj_vals):
    """
    Function to visualize the results of running k-means++ on various
    number of clusters for a given dataset to assess the optimal number
    of clusters
    Input:
        - n_clusters_vector: vector of various cluster lenghts to evaluate
        - objVals: The list of final values of the inertia criterion (sum
          of squared distances to the closest centroid for all observations
          in the training set) for all values of number of clusters
    """
    fig = plt.figure(figsize=(10, 5))
    plt.plot(n_clusters_vector, obj_vals)
    plt.title('Objective value from K-means++ vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Objective value from K-means++')
    plt.show()
    return fig


def get_optimal_num_clusters(n_clusters_vector, obj_vals):
    """
    Function to find the optimal number of clusters for the given dataset
    Input:
        - n_clusters_vector: vector of various cluster lenghts to evaluate
        - objVals: The list of final values of the inertia criterion (sum
          of squared distances to the closest centroid for all observations
          in the training set) for all values of number of clusters
    Output:
        - optimal_num_clusters: optimal number of clusters for the given data
    """
    optimal_num_clusters = n_clusters_vector[0]
    epsilon = 0.05
    for i in range(1, len(n_clusters_vector)):
        if obj_vals[i - 1] - obj_vals[i] < epsilon:
            optimal_num_clusters = n_clusters_vector[i - 1]
            break
    return optimal_num_clusters


def kmeans_viz(input_features, plot_figure=True):
    """
    Function to run k-means++ clustering algo for various number of
    clusters on the given input_features, visualize it and then
    return the optimal number of clusters
    Input:
        - input_features: input dataset/features
    Output:
        - optimal_num_clusters: optimal number of clusters for the given data
    """
    n_samples = input_features.shape[0]
    n_clusters_vector = get_cluster_vector(n_samples)
    obj_vals = get_cost_from_kmeans(n_clusters_vector, input_features)

    optimal_num_clusters = get_optimal_num_clusters(n_clusters_vector,
                                                    obj_vals)

    if plot_figure:
        plt_clust = visualize_kmeans(n_clusters_vector, obj_vals)
        return plt_clust, optimal_num_clusters
    return optimal_num_clusters
