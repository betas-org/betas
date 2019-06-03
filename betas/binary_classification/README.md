![logo](../../docs/logo_white.png)
# Binary Classification

## binary_score_plot

```
class binary_score_plot(self, scores=np.array([]), labels=np.array([]))
```
A class to construct plots to analyze performance of binary classifiers. Mainly acts as a wrapper for exisiting metrics and plotting functions.

## Parameters
- scores: 1D numpy array of model scores to plot (defaults to empty array)
- labels: 1D numpy array of model labels to plot (defaults to empty array)

## Methods
- `get_scores`: Get model scores to plot
-  `get_labels`: Get model labels to plot
- `set_scores`: Set model scores to plot
- `set_labels`: Set model labels to plot
- `plot_hist`: Plot two histograms, one with actual binary labels and one with model scores
- `plot_pr_by_threshold`: Plot model precision and recall by threshold, allowing a user to visualize model performance at various thresholds
- `plot_roc`: Plot true positive rate vs false positive rate

## Methods Details

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


## probability_plot
