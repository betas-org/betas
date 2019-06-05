![logo](../../docs/logo_white.png)
# Principal Component Analysis (PCA) Evaluation

Visuailize the the performance/evaluation of PCA for different number of dimensions on the input data

## Methods
- `sigmoid`: Sigmoid/Logistic function
- `get_classification_error`: Compute the misclassification error from the predicted labels and the actual ones
- `scale_features`: Scale the training as well as test data
- `get_pca_cv_results`: Run the PCA on the training data and then use it to find optimal lambda from the logistic regression cross validation function
- `run_pca_across_dimensions`: Run the PCA algorithms for various number of dimensions and evaluate each one based on the computed missclassification error values
- `plot_pca_errors`: Plot the misclassification error values for the various PCA runs for different dimensions
- `visualize_pca`: Generate the analysis of all the error values from different PCA dimensions in order to decide the most suited number of dimensions for the given dataset


## Methods Details

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
