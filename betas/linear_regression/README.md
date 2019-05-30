![logo](../../docs/logo_white.png)
# Betas Documentation

## Linear Regression Model Diagnostics

1. Residuals VS fitted plot
2. Normal qq plot
3. Scale-location plot
4. Residuals vs leverage plot

```
python model_diagnostics.py
```
Open http://127.0.0.1:8050/ to use the model diagnostics plots

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
- `'dist_plot`: Create a distribution plot with probability density function (PDF) curves
- `reg`: Fit linear regress and print out regression model report
- `resid_plot`: Create a residuals VS fitted plot
- `qq_plot`: Creates a normal qq plot
- `scale_loc_plot`: Creates a scale-location plot
- `resid_lever_plot`: Creates a residuals vs leverage plot

### Methods Details

```
__init__(self, dataframe, predictors=None, response=None)
```

```
get_dataframe(self)
```
Return the pandas dataframe


```
get_predictors(self)
```
Return the list of predictor variable(s)

```
get_response(self)
```
Return the response variable

```
matrix_plot(self, label=None)
```
Create a matrix scatter plot

Parameters:
- label: *string*, *optional*

    A categorical label for plot legend, selected from dataframe column names

```
corr_heatmap(self)
```
Create a heat map for observing the correlations among all predictors

```
reg_plot(self, var_x, var_y)
```
Create a scatter plot with regression line

Parameters:
- var_x: *string*

    A variable on x-axis, selected from dataframe column names
- var_y: *string*

    A variable on y-axis, selected from dataframe column names

```
box_plot(self, var_x, var_y):
```
Create a box plot

Parameters:
- var_x: *string*

    A variable on x-axis, selected from dataframe column names
- var_y: *string*

    A variable on y-axis, selected from dataframe column names

```
dist_plot(self, var_x, var_y):
```
Create a distribution plot with probability density function (PDF) curves

Parameters:
- var_x: *string*

    A variable on x-axis, selected from dataframe column names
- var_y: *string*

    A variable on y-axis, selected from dataframe column names

```
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

```
resid_plot(self, var_x=None, var_y=None):
```
Create a residuals VS fitted plot

Parameters:
- var_x: *array of string*, *optional*

    A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

    A response variable, selected from dataframe column names

```
qq_plot(self, var_x=None, var_y=None):
```
Creates a normal qq plot

Parameters:
- var_x: *array of string*, *optional*

    A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

    A response variable, selected from dataframe column names
    
```
scale_loc_plot(self, var_x=None, var_y=None):
```
Creates a scale-location plot
Goal: Check if the residuals suffer from non-constant variance, i.e., heteroscedasticity

Parameters:
- var_x: *array of string*, *optional*

    A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

    A response variable, selected from dataframe column names

```
resid_lever_plot(self, var_x=None, var_y=None):
```
Creates a residuals vs leverage plot

Parameters:
- var_x: *array of string*, *optional*

    A list of predictor variable(s), selected from dataframe column names
- var_y: *array of string*, *optional*

    A response variable, selected from dataframe column names
