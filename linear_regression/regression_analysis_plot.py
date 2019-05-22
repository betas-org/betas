# Linear Regression Analysis Plot

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot

# [Improvement: define a linear regression class]
# - Dataframe, selected predictors and response
# - Fitted into alinear regression model
# - Create linear regression plot and model assumption diagnostic plots

def matrix_plot(dataframe, huelabel):
    '''
    matrix plot
    '''
    sns.pairplot(dataframe, hue=huelabel, palette='Set1')
    return

def corr_heatmap(dataframe):
    '''
    Correlation Heat Map
    '''
    sns.heatmap(dataframe.corr(), annot=True, cmap="YlGnBu", linewidths=.5)
    return

def reg_plot(dataframe, X, Y):
    '''
    Regression Plot
    select 2 metrics to plot
    '''
    sns.regplot(x=X, y=Y, data=dataframe)
    return

def reg(X, Y):
    '''
    Regression model
    '''
    # [Improvement: user chosen predictors and response]
    model = sm.OLS(Y, sm.add_constant(X))
    model = model.fit()
    print(model.summary())
    return model

def resid_plot(dataframe, model, response_name):
    '''
    Residuals VS Fitted Plot
    '''
    # [Improvement: Tell how to observe this plot]
    fitted = model.fittedvalues
    sns.residplot(fitted, response_name, data=dataframe, lowess=True,
                         scatter_kws={'alpha': 0.5},
                         line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    return

def normal_qq_plot(model):
    '''
    Normal QQ Plot
    '''
    resid_norm = model.get_influence().resid_studentized_internal
    qq = ProbPlot(resid_norm)
    qq.qqplot(alpha=0.5, line='45', lw=1)
    plt.title('Normal Q-Q')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Standardized Residuals');
    return


def scale_location_plot(model):
    '''
    Scale-Location Plot
    Check if the residuals suffer from non-constant variance, aka heteroscedasticity.
    '''
    fitted = model.fittedvalues  
    resid_norm = model.get_influence().resid_studentized_internal
    resid_norm_abs_sqrt = np.sqrt(np.abs(resid_norm))
    plt.scatter(fitted, resid_norm_abs_sqrt, alpha=0.5);
    sns.regplot(fitted, resid_norm_abs_sqrt,
                       scatter=False, ci=False, lowess=True,
                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
    plt.title('Scale-Location')
    plt.xlabel('Fitted values')
    plt.ylabel('Absolute squared normalized residuals');
    return


def resid_lever_plot(model):
    '''
    Residuals vs Leverage Plot
    '''
    model_leverage = model.get_influence().hat_matrix_diag
    resid_norm = model.get_influence().resid_studentized_internal
    plt.scatter(model_leverage, resid_norm, alpha=0.5);
    sns.regplot(model_leverage, resid_norm, scatter=False, ci=False, lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    plt.title('Residuals vs Leverage')
    plt.xlabel('Leverage')
    plt.ylabel('Standardized Residuals');
    return

# [Improvement: Set same colors and text size for diagnostic plots]

