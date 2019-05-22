# Linear Regression Analysis Plot

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
    plot = plt.figure()
    plot = sns.residplot(fitted, response_name, data=dataframe, lowess=True,
                         scatter_kws={'alpha': 0.5},
                         line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    plot.set_title('Residuals vs Fitted')
    plot.set_xlabel('Fitted values')
    plot.set_ylabel('Residuals')
    return

def normal_qq_plot(model):
    '''
    Normal QQ Plot
    '''
    resid_norm = model.get_influence().resid_studentized_internal
    qq = ProbPlot(resid_norm)
    plot = qq.qqplot(alpha=0.5, line='45', lw=1)
    plot.axes[0].set_title('Normal Q-Q')
    plot.axes[0].set_xlabel('Theoretical Quantiles')
    plot.axes[0].set_ylabel('Standardized Residuals');
    return

# Scale-Location Plot
# Residuals vs Leverage Plot

# [Improvement: Set same colors and text size for diagnostic plots]