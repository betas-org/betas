'''
This module includes a class to create regression analysis plots
'''

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot


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
            - predictors: Predictor(s) (default=None)
            - response: Response (default=None)
        '''
        self.dataframe = dataframe
        self.predictors = predictors
        self.response = response

    def get_dataframe(self):
        '''
        Get dataframe to plot
        Output:
            - A pandas dataframe
        '''
        return self.dataframe
    
    def get_predictors(self):
        '''
        Get predictor variable(s)
        Output:
            - A list of string indicating the predictor variable(s)
        '''
        return self.predictors
    
    def get_response(self):
        '''
        Get response variable
        Output:
            - A string indicating the response variable
        '''
        return self.response
    
    def matrix_plot(self, label=None):
        '''
        Matrix scatter plot
        Input:
            - label: A categorical label for plot legend (default=None)
        '''
        df = self.get_dataframe()
        if label != None: # priority: label argument
            huelabel = label
        else:
            Y = self.get_response()
            huelabel = self.get_response()
        sns.pairplot(df, hue=huelabel, palette='Set1')

    def corr_heatmap(self):
        '''
        A heat map for observing the correlations among all predictors
        '''
        df = self.get_dataframe()
        sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", linewidths=.5)
    
    def reg_plot(self, X, Y):
        '''
        Scatter plot with regression line
        Input:
            - X: A variable on x-axis
            - Y: A variable on y-axis
        '''
        df = self.get_dataframe()
        sns.regplot(x=X, y=Y, data=df)
    
    def box_plot(self, X, Y):
        '''
        Box plot
        Input:
            - X: A variable on x-axis
            - Y: A variable on y-axis
        '''
        df = self.get_dataframe()
        sns.boxplot(x=X, y=Y, data=df)
    
    def dist_plot(self, X, Y):
        '''
        Distribution plot with probability density function (PDF) curves
        Input:
            - X: A variable on x-axis
            - Y: A categoricle variable shown in plot legend
        '''
        df = self.get_dataframe()
        sns.FacetGrid(df, hue=Y, height=4).map(sns.distplot, X).add_legend()
    
    def reg(self, X, Y, report=False):
        '''
        Regression model report
        Input:
            - X: A variable on x-axis
            - Y: A variable on y-axis
            - report: A boolean indicating if print model report (default=False)
        '''
        df = self.get_dataframe()
        pred = df[X]
        resp = df[Y]
        model = sm.OLS(resp, sm.add_constant(pred))
        model = model.fit()
        if report == True:
            print(model.summary())
        return model

    def resid_plot(self, X=None, Y=None):
        '''
        Residuals VS fitted plot
        Input:
            - X: A variable on x-axis (default=None)
            - Y: A variable on y-axis (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None: # priority: arguments X, Y
            model = self.reg(X, Y)
        else:
            X = self.get_predictors()
            Y = self.get_response()
            if X != None and Y != None:
                model = self.reg(X, Y)
            else:
                raise ValueError('No predictors or response assigned')
        fitted = model.fittedvalues
        sns.residplot(fitted, Y, data=dataframe, lowess=True,
                      scatter_kws={'alpha': 0.5},
                      line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
        plt.title('Residuals vs Fitted')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')

    def qq_plot(self, X=None, Y=None):
        '''
        Normal qq plot
            - X: A variable on x-axis (default=None)
            - Y: A variable on y-axis (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None: # priority: arguments X, Y
            model = self.reg(X, Y)
        else:
            X = self.get_predictors()
            Y = self.get_response()
            if X != None and Y != None:
                model = self.reg(X, Y)
            else:
                raise ValueError('No predictors or response assigned')
        resid_norm = model.get_influence().resid_studentized_internal
        qq = ProbPlot(resid_norm)
        qq.qqplot(color='#2077B4', alpha=0.5, line='45', lw=0.5)
        plt.title('Normal Q-Q')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Standardized Residuals')

    def scale_location_plot(self, X=None, Y=None):
        '''
        Scale-location plot
        Goal: Check if the residuals suffer from non-constant variance, i.e.,
              heteroscedasticity
        Input:
            - X: A variable on x-axis (default=None)
            - Y: A variable on y-axis (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None: # priority: arguments X, Y
            model = self.reg(X, Y)
        else:
            X = self.get_predictors()
            Y = self.get_response()
            if X != None and Y != None:
                model = self.reg(X, Y)
            else:
                raise ValueError('No predictors or response assigned')
        fitted = model.fittedvalues  
        resid_norm = model.get_influence().resid_studentized_internal
        resid_norm_abs_sqrt = np.sqrt(np.abs(resid_norm))
        plt.scatter(fitted, resid_norm_abs_sqrt, alpha=0.5)
        sns.regplot(fitted, resid_norm_abs_sqrt, scatter=False, ci=False, lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 1});
        plt.title('Scale-Location')
        plt.xlabel('Fitted values')
        plt.ylabel('Absolute squared normalized residuals')

    def resid_lever_plot(self, X=None, Y=None):
        '''
        Residuals vs leverage plot
        Input:
            - X: A variable on x-axis (default=None)
            - Y: A variable on y-axis (default=None)
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None: # priority: arguments X, Y
            model = self.reg(X, Y)
        else:
            X = self.get_predictors()
            Y = self.get_response()
            if X != None and Y != None:
                model = self.reg(X, Y)
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
