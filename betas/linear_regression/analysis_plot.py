'''
This module includes a class to create linear regression analysis plots
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot


class analysis_plot(object):
    '''
    A class to create regression analysis plots based on a input dataframe,
    selected predictor variable(s) and a response variable.
    Plot types basically includes:
        - Data overview
        - Linear regression model
        - Model assumption diagnostics
    '''

    def __init__(self, dataframe, predictors=None, response=None):
        '''
        A constructor to initialize the object
        Input:
            - dataframe: A pandas dataframe with proper column names
            - predictors: Predictor(s) (default=None)
            - response: Response (default=None)
        '''
        self.dataframe = dataframe
        self.predictors = predictors
        self.response = response

    def get_dataframe(self):
        '''
        Return the pandas dataframe
        Output:
            - A pandas dataframe
        '''
        return self.dataframe

    def get_predictors(self):
        '''
        Return the list of predictor variable(s)
        Output:
            - A list of string indicating the predictor variable(s)
        '''
        return self.predictors

    def get_response(self):
        '''
        Return the response variable
        Output:
            - A string indicating the response variable
        '''
        return self.response

    def matrix_plot(self, label=None):
        '''
        Create a matrix scatter plot
        Input:
            - label: A categorical label for plot legend (default=None)
        '''
        dataframe = self.get_dataframe()
        if label is not None: # priority: label argument
            huelabel = label
        else:
            #Y = self.get_response()
            huelabel = self.get_response()
        sns.pairplot(dataframe, hue=huelabel, palette='Set1')

    def corr_heatmap(self):
        '''
        Create a heat map for observing the correlations among all predictors
        '''
        dataframe = self.get_dataframe()
        sns.heatmap(dataframe.corr(), annot=True, cmap="YlGnBu", linewidths=.5)

    def reg_plot(self, var_x, var_y):
        '''
        Create a scatter plot with regression line
        Input:
            - var_x: A variable on x-axis
            - var_y: A variable on y-axis
        '''
        dataframe = self.get_dataframe()
        sns.regplot(x=var_x, y=var_y, data=dataframe)

    def box_plot(self, var_x, var_y):
        '''
        Create a box plot
        Input:
            - var_x: A variable on x-axis
            - var_y: A variable on y-axis
        '''
        dataframe = self.get_dataframe()
        sns.boxplot(x=var_x, y=var_y, data=dataframe)

    def dist_plot(self, var_x, var_y):
        '''
        Create a distribution plot with probability density function (PDF) curves
        Input:
            - var_x: A variable on x-axis
            - var_y: A categoricle variable shown in plot legend
        '''
        dataframe = self.get_dataframe()
        sns.FacetGrid(dataframe, hue=var_y, height=4).map(sns.distplot, var_x).add_legend()

    def reg(self, var_x, var_y, report=False):
        '''
        Fit linear regress and print out regression model report
        Input:
            - var_x: A variable on x-axis
            - var_y: A variable on y-axis
            - report: A boolean indicating if print model report (default=False)
        '''
        dataframe = self.get_dataframe()
        pred = dataframe[var_x]
        resp = dataframe[var_y]
        model = sm.OLS(resp, sm.add_constant(pred))
        model = model.fit()
        if report is True:
            print(model.summary())
        return model

    def resid_plot(self, var_x=None, var_y=None):
        '''
        Create a residuals VS fitted plot
        Input:
            - var_x: A list of predictor variable(s) (default=None)
            - var_y: A response variable (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if var_x is not None and var_y is not None: # priority: arguments var_x, var_y
            model = self.reg(var_x, var_y)
        else:
            var_x = self.get_predictors()
            var_y = self.get_response()
            if var_x is not None and var_y is not None:
                model = self.reg(var_x, var_y)
            else:
                raise ValueError('No predictors or response assigned')
        fitted = model.fittedvalues
        sns.residplot(fitted, var_y, data=dataframe, lowess=True,
                      scatter_kws={'alpha': 0.5},
                      line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
        plt.title('Residuals vs Fitted')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')

    def qq_plot(self, var_x=None, var_y=None):
        '''
        Creates a normal qq plot
        Input:
            - var_x: A list of predictor variable(s) (default=None)
            - var_y: A response variable (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        if var_x is not None and var_y is not None: # priority: arguments var_x, var_y
            model = self.reg(var_x, var_y)
        else:
            var_x = self.get_predictors()
            var_y = self.get_response()
            if var_x is not None and var_y is not None:
                model = self.reg(var_x, var_y)
            else:
                raise ValueError('No predictors or response assigned')
        resid_norm = model.get_influence().resid_studentized_internal
        qq_plt = ProbPlot(resid_norm)
        qq_plt.qqplot(color='#2077B4', alpha=0.5, line='45', lw=0.5)
        plt.title('Normal Q-Q')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Standardized Residuals')

    def scale_loc_plot(self, var_x=None, var_y=None):
        '''
        Creates a scale-location plot
        Goal: Check if the residuals suffer from non-constant variance, i.e.,
              heteroscedasticity
        Input:
            - var_x: A list of predictor variable(s) (default=None)
            - var_y: A response variable (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        if var_x is not None and var_y is not None: # priority: arguments var_x, var_y
            model = self.reg(var_x, var_y)
        else:
            var_x = self.get_predictors()
            var_y = self.get_response()
            if var_x is not None and var_y is not None:
                model = self.reg(var_x, var_y)
            else:
                raise ValueError('No predictors or response assigned')
        fitted = model.fittedvalues
        resid_norm = model.get_influence().resid_studentized_internal
        resid_norm_abs_sqrt = np.sqrt(np.abs(resid_norm))
        plt.scatter(fitted, resid_norm_abs_sqrt, alpha=0.5)
        sns.regplot(fitted, resid_norm_abs_sqrt, scatter=False, ci=False, lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
        plt.title('Scale-Location')
        plt.xlabel('Fitted values')
        plt.ylabel('Absolute squared normalized residuals')

    def resid_lever_plot(self, var_x=None, var_y=None):
        '''
        Creates a residuals vs leverage plot
        Input:
            - var_x: A list of predictor variable(s) (default=None)
            - var_y: A response variable (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        if var_x is not None and var_y is not None: # priority: arguments var_x, var_y
            model = self.reg(var_x, var_y)
        else:
            var_x = self.get_predictors()
            var_y = self.get_response()
            if var_x is not None and var_y is not None:
                model = self.reg(var_x, var_y)
            else:
                raise ValueError('No predictors or response assigned')
        model_leverage = model.get_influence().hat_matrix_diag
        resid_norm = model.get_influence().resid_studentized_internal
        plt.scatter(model_leverage, resid_norm, alpha=0.5)
        sns.regplot(model_leverage, resid_norm, scatter=False, ci=False, lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
        plt.title('Residuals vs Leverage')
        plt.xlabel('Leverage')
        plt.ylabel('Standardized Residuals')
