![logo](../docs/logo_white.png)

# Documentation

- [Linear Regression](#linear-regression)
  - [Regression Diagnostics Dashboard](#regression-diagnostics-dashboard)
  - [analysis_plot](#regression_analysis_plot)
- [Binary Classification](#binary-classification)
  - [Binary Score Diagnostics Dashboard](#binary-score-diagnostics-dashboard)
  - [binary_score_plot](#binary_score_plot)
- [Principal Component Analysis](#principal-component-analysis)
  - [pca_evaluate](#pca_evaluate)
- [Clustering](#clustering)
  - [clustering_evaluate](#clustering_evaluate)

# Linear Regression

## Regression Diagnostics Dashboard
<p align="right">
  <a href="#documentation">[Back to top]</a>
</p>

There are four plots to check linear regression model assumptions
- Residuals VS fitted plot
- Normal qq plot
- Scale-location plot
- Residuals vs leverage plot

**Tutorial**

1. Run the following code to start:
```
python regression_diagnostics.py
```
2. Input your desinated CSV data file url or local path
3. Open <http://127.0.0.1:8050/> to use the model diagnostics tool
4. Select metrics that you are interested in and explore the data
5. Use `Ctrl C` to terminate

**Limitation**
This tool is designed for **CSV** data file only.

## regression_analysis_plot
<p align="right">
  <a href="#documentation">[Back to top]</a>
</p>

```python
class analysis_plot.AnalysisPlot(dataframe, predictors=None, response=None)
```
A class to create regression analysis plots based on a input dataframe, selected predictor variable(s) and a response variable.

Plot types basically includes:
- Data overview
- Linear regression model
- Model assumption diagnostics

### Parameters
- dataframe: A pandas dataframe with proper column names to be analyzed
- predictors: A list of predictor variable(s)
- response: A response variable

### Methods
- `get_dataframe`: Return the pandas dataframe
- `get_predictors`: Return the list of predictor variable(s)
- `get_response`: Return the response variable
- `get_model`: Return linear regression OLS model
- `set_predictors`: Set predictor variable(s)
- `set_response`: Set response variable
- `set_model`: Set linear regression OLS model
- `matrix_plot`: Create a matrix scatter plot
- `corr_heatmap`: Create a heat map for observing the correlations among all predictors
- `reg_plot`: Create a scatter plot with regression line
- `box_plot`: Create a box plot
- `dist_plot`: Create a distribution plot with probability density function (PDF) curves
- `reg`: Fit linear regress and print out regression model report
- `resid_plot`: Create a residuals VS fitted plot
- `qq_plot`: Creates a normal qq plot
- `scale_loc_plot`: Creates a scale-location plot
- `resid_lever_plot`: Creates a residuals vs leverage plot

### Methods Details

```python
__init__(self, dataframe, predictors=None, response=None)
```

```python
get_dataframe(self)
```
Return the pandas dataframe

```python
get_predictors(self)
```
Return the list of predictor variable(s)

```python
get_response(self)
```
Return the response variable

```python
get_model(self)
```
Return linear regression OLS model

```python
set_predictors(self)
```
Set predictor variable(s)

Parameters:
- predictors: *array of string*

  A list of predictor variable(s)

```python
set_response(self)
```
Set response variable

Parameters:
- response: *string*

  Response variable

```python
set_model(self)
```
Set linear regression OLS model

Parameters:
- model: *OLS model*

    A linear regression OLS model

```python
matrix_plot(self, label=None)
```
Create a matrix scatter plot

Parameters:
- label: *string*, *optional*

  A categorical label for plot legend, selected from dataframe column names

```python
corr_heatmap(self)
```
Create a heat map for observing the correlations among all predictors

```python
reg_plot(self, var_x, var_y)
```
Create a scatter plot with regression line

Parameters:
- var_x: *string*

  A variable on x-axis, selected from dataframe column names
- var_y: *string*

  A variable on y-axis, selected from dataframe column names

```python
box_plot(self, var_x, var_y):
```
Create a box plot

Parameters:
- var_x: *string*

  A variable on x-axis, selected from dataframe column names
- var_y: *string*

  A variable on y-axis, selected from dataframe column names

```python
dist_plot(self, var_x, var_y):
```
Create a distribution plot with probability density function (PDF) curves

Parameters:
- var_x: *string*

  A variable on x-axis, selected from dataframe column names
- var_y: *string*

  A variable on y-axis, selected from dataframe column names

```python
reg(self, var_x, var_y, report=False):
```
Fit linear regress and print out regression model report

Parameters:
- var_x: *string*

  A variable on x-axis, selected from dataframe column names
- var_y: *string*

  A variable on y-axis, selected from dataframe column names
- report: *boolean*, *optional*

  A boolean indicating if print model report

Returns:
- A fitted linear regresion model

```python
resid_plot(self, var_x=None, var_y=None):
```
Create a residuals VS fitted plot

Parameters:
- var_x: *array of string*, *optional*

  A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

  A response variable, selected from dataframe column names

```python
qq_plot(self, var_x=None, var_y=None):
```
Creates a normal qq plot

Parameters:
- var_x: *array of string*, *optional*

  A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

  A response variable, selected from dataframe column names

```python
scale_loc_plot(self, var_x=None, var_y=None):
```
Creates a scale-location plot.

Parameters:
- var_x: *array of string*, *optional*

  A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

  A response variable, selected from dataframe column names

```python
resid_lever_plot(self, var_x=None, var_y=None):
```
Creates a residuals vs leverage plot

Parameters:
- var_x: *array of string*, *optional*

  A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

  A response variable, selected from dataframe column names


# Binary Classification

## Binary Score Diagnostics Dashboard
<p align="right">
<a href="#documentation">[Back to top]</a>
</p>

This dashboard helps users better understand the distribution of modeled scores

**Tutorial**

1. Run the following code to start:
```
bokeh serve --show binary_score_diagnostics.py
```
2. Input your desinated data path
3. Adjust plots by threshold slider
4. Use `Ctrl C` to terminate

## binary_score_plot
<p align="right">
  <a href="#documentation">[Back to top]</a>
</p>

```python
class binary_score_plot.BinaryScorePlot(self, scores=np.array([]), labels=np.array([]))
```
A class to construct plots to analyze performance of binary classifiers. Mainly acts as a wrapper for exisiting metrics and plotting functions.

### Parameters
- scores: 1D numpy array of model scores to plot (defaults to empty array)
- labels: 1D numpy array of model labels to plot (defaults to empty array)

### Methods
- `get_scores`: Get model scores to plot
-  `get_labels`: Get model labels to plot
- `set_scores`: Set model scores to plot
- `set_labels`: Set model labels to plot
- `plot_hist`: Plot two histograms, one with actual binary labels and one with model scores
- `plot_pr_by_threshold`: Plot model precision and recall by threshold, allowing a user to visualize model performance at various thresholds
- `plot_roc`: Plot true positive rate vs false positive rate

### Methods Details

```python
__init__(self, scores=np.array([]), labels=np.array([]))
```

```python
get_scores(self)
```
Get model scores to plot

```python
get_labels(self)
```
Get model labels to plot

```python
set_scores(self, scores)
```
Set model scores to plot

Parameters:
- scores: *array*

  1D numpy array of model scores

```python
set_labels(self, labels)
```
Set model labels to plot

Parameters:
- scores: *array*

  1D numpy array of model scores

```python
plot_hist(self, bins=30)
```
Plot two histograms, one with actual binary labels and one with model scores

Parameters:
- bins: *integer*, *optional*

  Number of histogram bins to use

```python
plot_pr_by_threshold(self)
```
Plot model precision and recall by threshold, allowing a user to visualize model performance at various thresholds

```python
plot_roc(self)
```
Plot true positive rate vs false positive rate


# Principal Component Analysis

## pca_evaluate
<p align="right">
  <a href="#documentation">[Back to top]</a>
</p>

Visuailize the the performance/evaluation of PCA for different number of dimensions on the input data

### Methods
- `sigmoid`: Sigmoid/Logistic function
- `get_classification_error`: Compute the misclassification error from the predicted labels and the actual ones
- `scale_features`: Scale the training as well as test data
- `get_pca_cv_results`: Run the PCA on the training data and then use it to find optimal lambda from the logistic regression cross validation function
- `run_pca_across_dimensions`: Run the PCA algorithms for various number of dimensions and evaluate each one based on the computed missclassification error values
- `plot_pca_errors`: Plot the misclassification error values for the various PCA runs for different dimensions
- `visualize_pca`: Generate the analysis of all the error values from different PCA dimensions in order to decide the most suited number of dimensions for the given dataset


### Methods Details

```python
sigmoid(features)
```
Sigmoid/Logistic function

Parameter:
- features

Returns:
- result: *narray*

Computed sigmoid function result

```python
get_misclassification_error(features, labels, beta)
```
Compute the misclassification error from the predicted labels and the actual ones

Parameter:
- features
- labels
- beta

Returns:
- error: *float*

  Misclassification error

```python
scale_features(train_features, test_features)
```
Scale the training as well as test data

Parameter:
- train_features:

  Training features
- test_features:

  Test features

Returns:
- train_features_std: *narray*

Standardized training features
- test_features_std: *narray*

Standardized test features

```python
get_pca_and_cv_results(train_features_std, test_features_std, train_labels, dimension)
```
Run the PCA on the training data and then use it to find optimal lambda from the logistic regression cross validation function

Parameters:
- train_features_std: *narray*

  Standardized training features
- test_features_std: *narray*

  Standardized test features
- train_labels: *narray*

  Training labels
- dimensions: *integer*

  Number of components to run PCA on

Returns:
- train_features_pca_std: *narray*

  Standardized training features with PCA
- test_features_pca_std: *narray*

  Standardized test features with PCA
- lambda_star: *float*

  Optimal penalty parameter $\lambda$

```python
run_pca_across_dimensions(train_features_std, train_labels, test_features_std, test_labels)
```
Run the PCA algorithms for various number of dimensions and evaluate each one based on the computed missclassification error values

Parameters:
- train_features_std: *narray*

  Standardized training features
- train_labels: *narray*

  Training labels
- test_features_std: *narray*

  Standardized test features
- test_labels: *narray*

Test labels

Returns:
- missclassification_err_train: *float*

  Misclassification error on training set
- missclassification_err_test: *float*

  Misclassification error on test set
- dimensions: *array*

  A list of different dimensions

```python
plot_pca_errors(missclassification_err_train, missclassification_err_test, dimensions)
```
Plot the misclassification error values for the various PCA runs for different dimensions

Parameters:
- missclassification_err_train: *float*

  Misclassification error on training set
- missclassification_err_test: *float*

  Misclassification error on test set
- dimensions: *array*

  A list of different dimensions

Returns:
- fig: *figure*

Figure for the misclassification error plot for PCA with different dimensions

```python
visualize_pca(train_features, train_labels, test_features, test_labels)
```
Generate the analysis of all the error values from different PCA dimensions in order to decide the most suited number of dimensions for the given dataset

Parameters:
- train_features: *narray*

  Train features
- train_labels: *narray*

  Train labels
- test_features: *narray*

  Test features
- test_labels: *narray*

  Test labels

Returns:
- figure: *figure*

  Figure for the misclassification error plot for PCA with different dimensions


# Clustering

## clustering_evaluate
<p align="right">
  <a href="#documentation">[Back to top]</a>
</p>

Visuailize the the performance/evaluation of Clustering for different number of clusters on the given input data using k-means++

### Methods
- `get_cluster_vector`: Create the vector of various cluster lenghts to evaluate
- `get_cost_from_kmeans`: Derive the objective values for k-means++ for the various values of number of clusters that are input
- `visualize_kmeans`: Visualize the results of running k-means++ on various number of clusters for a given dataset to assess the optimal number of clusters
- `get_optimal_num_clusters`: Find the optimal number of clusters for the given dataset
- `kmeans_viz_and_opt_clusters`: Run k-means++ clustering algo for various number of clusters on the given input_features, visualize it and then return the optimal number of clusters


### Methods Details

```python
get_cluster_vector(n_samples)
```
Create the vector of various cluster lenghts to evaluate

Parameter:
- n_samples: *integer*

  Total number of samples/examples in the dataset

Returns:
- Vector of various cluster lenghts to evaluate

```python
get_cost_from_kmeans(n_clusters_vector, data)
```
Derive the objective values for k-means++ for the various values of number of clusters that are input

Parameter:
- n_clusters_vector:

  Vector of various cluster lenghts to evaluate
- X:

  Input dataset/features

Returns:
- The list of final values of the inertia criterion (sum of squared distances to the closest centroid for all observations in the training set) for all values of number of clusters

```python
visualize_kmeans(n_clusters_vector, obj_vals)
```
Visualize the results of running k-means++ on various number of clusters for a given dataset to assess the optimal number of clusters

Parameter:
- n_clusters_vector:

  Vector of various cluster lenghts to evaluate
- objVals:

  The list of final values of the inertia criterion (sum of squared distances to the closest centroid for all observations in the training set) for all values of number of clusters

```python
get_optimal_num_clusters(n_clusters_vector, obj_vals)
```
Find the optimal number of clusters for the given dataset

Parameters:
- n_clusters_vector:

  Vector of various cluster lenghts to evaluate
- objVals:

  The list of final values of the inertia criterion (sum of squared distances to the closest centroid for all observations in the training set) for all values of number of clusters

Returns:
- Optimal number of clusters for the given dataset


```python
kmeans_viz_and_opt_clusters(input_features)
```
Run k-means++ clustering algo for various number of clusters on the given input_features, visualize it and then return the optimal number of clusters

Parameters:
- input_features:

  Input dataset/features

Returns:
- Optimal number of clusters for the given dataset
