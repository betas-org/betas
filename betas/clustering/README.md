![logo](../../docs/logo_white.png)
# Clustering Evaluation

Visuailize the the performance/evaluation of Clustering for different number of clusters on the given input data using k-means++

## Methods
- `get_cluster_vector`: Function to create the vector of various cluster lenghts to evaluate
- `get_cost_from_kmeans`: Function to derive the objective values for k-means++ for the
    various values of number of clusters that are input
- `visualize_kmeans`: Function to visualize the results of running k-means++ on various
    number of clusters for a given dataset to assess the optimal number of clusters
- `get_optimal_num_clusters`: Function to find the optimal number of clusters for the given dataset
- `kmeans_viz_and_opt_clusters`: Function to run k-means++ clustering algo for various number of
    clusters on the given input_features, visualize it and then return the optimal number of clusters


## Methods Details

```python
get_cluster_vector(n_samples)
```
Function to create the vector of various cluster lenghts to evaluate

Parameter:
- n_samples: total number of samples/examples in the dataset

Returns:
- n_clusters_vector: vector of various cluster lenghts to evaluate


```python
get_cost_from_kmeans(n_clusters_vector, data)
```
Function to derive the objective values for k-means++ for the
    various values of number of clusters that are input

Parameter:
- n_clusters_vector: vector of various cluster lenghts to evaluate
- X: input dataset/features

Returns:
- objVals: The list of final values of the inertia criterion (sum
      of squared distances to the closest centroid for all observations
      in the training set) for all values of number of clusters

```python
visualize_kmeans(n_clusters_vector, obj_vals)
```
Function to visualize the results of running k-means++ on various
number of clusters for a given dataset to assess the optimal number
of clusters

Parameter:
- n_clusters_vector: vector of various cluster lenghts to evaluate
- objVals: The list of final values of the inertia criterion (sum
   of squared distances to the closest centroid for all observations
   in the training set) for all values of number of clusters

```python
get_optimal_num_clusters(n_clusters_vector, obj_vals)
```
Function to find the optimal number of clusters for the given dataset

Parameters:
- n_clusters_vector: vector of various cluster lenghts to evaluate
- objVals: The list of final values of the inertia criterion (sum
   of squared distances to the closest centroid for all observations
   in the training set) for all values of number of clusters

Returns:
- optimal_num_clusters: optimal number of clusters for the given dataset


```python
kmeans_viz_and_opt_clusters(input_features)
```
Function to run k-means++ clustering algo for various number of
clusters on the given input_features, visualize it and then
return the optimal number of clusters

Parameters:
- input_features: input dataset/features

Returns:
- optimal_num_clusters: optimal number of clusters for the given dataset
