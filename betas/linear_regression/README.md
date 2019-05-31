![logo](../../docs/logo_white.png)
# Betas Documentation

## Linear Regression Model Diagnostics Tool

There are four plots to check linear regression model assumptions
- Residuals VS fitted plot
- Normal qq plot
- Scale-location plot
- Residuals vs leverage plot

1. Run the following code in Terminal:

```
python model_diagnostics.py
```

2. Open <http://127.0.0.1:8050/> to use the model diagnostics tool
3. Select metrics that you are interested in and explore the tool

## linear_regression.analysis_plot

```
class linear_regression.analysis_plot(dataframe, predictors=None, response=None)
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
Creates a scale-location plot
Goal: Check if the residuals suffer from non-constant variance, i.e., heteroscedasticity

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
