'''
This module includes a class to create linear regression analysis plots
'''
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import copy
plt.style.use('seaborn')

class regression_analysis_plot(object):
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
            - predictors: A list of predictor variable(s) (default=None)
            - response: A response variable (default=None)
        '''
        self.dataframe = dataframe
        if predictors is not None:
            for pred in predictors:
                if pred not in dataframe.columns:
                    raise ValueError('Input predictor variable(s) not existed in the given dataframe')
                    return
        if response is not None:
            if response not in dataframe.columns:
                raise ValueError('Input response variable not existed in the given dataframe')
                return
        self.predictors = predictors
        self.response = response

    def get_dataframe(self):
        '''
        Return the pandas dataframe
        Output:
            - A pandas dataframe
        '''
        return copy.deepcopy(self.dataframe)

    def get_predictors(self):
        '''
        Return the list of predictor variable(s)
        Output:
            - A list of string indicating the predictor variable(s)
        '''
        return copy.deepcopy(self.predictors)

    def get_response(self):
        '''
        Return the response variable
        Output:
            - A string indicating the response variable
        '''
        return copy.deepcopy(self.response)
    
    def set_predictors(self, predictors):
        '''
        Set predictor variable(s)
        Input:
            - predictors: A list of string indicating the predictor variable(s)
        '''
        dataframe = self.get_dataframe()
        for pred in predictors:
            if pred not in dataframe.columns:
                raise ValueError('Input predictor variable(s) not existed in the given dataframe')
                return
        self.predictors = predictors
    
    def set_response(self, response):
        '''
        Set response variable
        Input:
            - response: A string indicating the response variable
        '''
        dataframe = self.get_dataframe()
        if response not in dataframe.columns:
            raise ValueError('Input response variable not existed in the given dataframe')
            return
        self.response = response

    def matrix_plot(self, label=None):
        '''
        Create a matrix scatter plot
        Input:
            - label: A categorical label for plot legend (default=None)
        '''
        dataframe = self.get_dataframe()
        cols = self.get_predictors()
        huelabel = self.get_response()
        cols.append(huelabel)
        if label is not None: # priority: label argument
            huelabel = label
        if huelabel is not None:
            sns.pairplot(dataframe[cols], hue=huelabel, palette='Set1')
        else:
            sns.pairplot(dataframe[cols], palette='Set1')

    def corr_heatmap(self, **kwargs):
        '''
        Create a heat map for observing the correlations among all predictors
        '''
        dataframe = self.get_dataframe()
        if 'figsize' in kwargs:
            figsize = kwargs['figsize']
        else:
            figsize = (10,10)
        plt.figure(figsize=figsize)
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
            - var_y: A categorical variable shown in plot legend
        '''
        dataframe = self.get_dataframe()
        sns.FacetGrid(dataframe, hue=var_y, height=5).map(sns.distplot, var_x).add_legend()

    def reg(self, var_x=None, var_y=None, report=False):
        '''
        Fit linear regress and print out regression model report
        Input:
            - var_x: A list of predictor variable(s) (default=None)
            - var_y: A response variable (default=None)
            - report: A boolean indicating if print model report (default=False)
        '''
        dataframe = self.get_dataframe()
        if var_x is not None and var_y is not None: # priority: arguments
            pred = dataframe[var_x]
            resp = dataframe[var_y]
        else:
            pred = dataframe[self.get_predictors()]
            resp = dataframe[[self.get_response()]]
        try:
            model = sm.OLS(resp, sm.add_constant(pred))
            model = model.fit()
            if report is True:
                print(model.summary())
            return model
        except:
            print("TypeError: Predictor/Response data type cannot be casted. Please select again")


    def resid_plot(self, var_x=None, var_y=None):
        '''
        Create a residuals VS fitted plot
        Input:
            - var_x: A list of predictor variable(s) (default=None)
            - var_y: A response variable (default=None)
        '''
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
        theo = qq_plt.theoretical_quantiles
        sample = qq_plt.sample_quantiles
        plt.scatter(theo, sample, alpha=0.5)
        sns.regplot(theo, theo, scatter=False, ci=False, lowess=True,
                   line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
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
