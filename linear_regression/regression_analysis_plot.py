# Linear Regression Analysis Plot

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot


# [Improvement: define a linear regression class]
class regression_analysis_plot(object):
    '''
    Dataframe, selected predictors and response
    Fitted into alinear regression model
    Create linear regression plot and model assumption diagnostic plots
    
    Input:
        - A dataframe
        - choose predictor(s)
        - choose response
    '''

    
    def __init__(self, dataframe, predictors=None, response=None):
        self.dataframe = dataframe
        self.predictors = predictors
        self.response = response


    def get_dataframe(self):
        '''
        Get dataframe to plot
        Output:
            - dataframe
        '''
        return self.dataframe

    
    def get_predictors(self):
        '''
        Get predictor variable(s)
        Output:
            - A list of string indicating predictor variable(s)
        '''
        return self.predictors
    
    
    def get_response(self):
        '''
        Get response variable
        Output:
            - A string indicating response variable
        '''
        return self.response
    
    
    def matrix_plot(self):
        '''
        matrix plot
        '''
        df = self.get_dataframe()
        huelabel = self.get_response()
        sns.pairplot(df, hue=huelabel, palette='Set1')

        
    def corr_heatmap(self):
        '''
        Correlation Heat Map
        '''
        df = self.get_dataframe()
        sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", linewidths=.5)

        
    def reg_plot(self, X, Y):
        '''
        Regression Plot
        Select 2 metrics to plot
        '''
        df = self.get_dataframe()
        sns.regplot(x=X, y=Y, data=df)

        
    def reg(self, X, Y, report=False):
        '''
        Regression model report
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
        Residuals VS Fitted Plot
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None:
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
        Normal QQ Plot
        '''
        # fix color
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None:
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
        qq.qqplot(alpha=0.5, line='45', lw=1)
        plt.title('Normal Q-Q')
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Standardized Residuals')


    def scale_location_plot(self, X=None, Y=None):
        '''
        Scale-Location Plot
        Check if the residuals suffer from non-constant variance, aka heteroscedasticity.
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None:
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
        sns.regplot(fitted, resid_norm_abs_sqrt,scatter=False, ci=False, lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
        plt.title('Scale-Location')
        plt.xlabel('Fitted values')
        plt.ylabel('Absolute squared normalized residuals')

        
    def resid_lever_plot(self, X=None, Y=None):
        '''
        Residuals vs Leverage Plot
        '''
        # [Improvement: Tell how to observe this plot]
        dataframe = self.get_dataframe()
        if X != None and Y != None:
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

# [Improvement: Set same colors and text size for diagnostic plots]

