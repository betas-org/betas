"""
Basic Dash to present linear regression model assumptions diagnostics
To run: python model_diagnostics.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objs as go
from statsmodels.graphics.gofplots import ProbPlot

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from regression_analysis_plot import regression_analysis_plot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
lowess = sm.nonparametric.lowess

# Read csv dataset
address = input('Please enter CSV data file url or path:\nUrl example: www.someplace.com/mydata.csv\nPath example: ./mydata.csv\n')
DF = pd.read_csv(address, sep=',', index_col=0)
DF = DF.dropna() # remove missing data
DF = DF.select_dtypes(exclude=['object']) # keep only numeric columns
COLS = DF.columns
MYCLASS = plt.analysis_plot(DF)

# Layout and Plots
app.layout = html.Div([
    html.H1('Linear Regression Model Diagnostics', style={'textAlign': 'center'}),
    html.Div('Residuals versus Predicted Plot:', style={'font-weight':'bold'}),
    html.Div('Checking the assumption of linearity and homoscedasticity. If the residuals appear to be very large (big positive value or big negative value), the model does not meet the linear model assumption. To assess the assumption of linearity we want to ensure that the residuals are not too far away from 0.'),
    html.Div('Normal Q-Q Plot:', style={'font-weight':'bold'}),
    html.Div('Checking the normality assumption by comparing the residuals to "ideal" normal observations. If observations lie well along the 45-degree line in the QQ-plot, we may assume that normality holds here.'),
    html.Div('Scale-Location Plot:', style={'font-weight':'bold'}),
    html.Div('Checking the assumption of homoscedasticity(equal variance). If residuals are spread equally along the ranges of predictors, the assumption of equal variance (homoscedasticity) is satisfied. It’s good if you see a horizontal line with equally (randomly) spread points.'),
    html.Div('Residuals vs Leverage Plot:', style={'font-weight':'bold'}),
    html.Div('Checking if there are influential cases. Not all outliers are influential in linear regression analysis, we watch out for outlying values at the upper right corner or at the lower right corner, since the regression results will be altered if we exclude those cases.'),
    
    html.Div('Data Source: %s\n' % address, style={'font-weight':'bold'}),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='predictors',
                options=[{'label': i, 'value': i} for i in COLS],
                placeholder='Select predictor variable(s)',
                multi=True
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='response',
                options=[{'label': i, 'value': i} for i in COLS],
                placeholder='Select a response variable'
            )
        ], style={'width': '49%', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'}),
    html.Div([
        dcc.Graph(id='residual-plot')
    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='qq-plot')
    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='scale-location-plot')
    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(id='residual-leverage-plot')
    ], style={'width': '49%', 'display': 'inline-block'})
])

@app.callback(
    Output('residual-plot', 'figure'),
    [Input('predictors', 'value'),
     Input('response', 'value')])
def update_resid_plot(predictors, response):
    """
    Residual Plot
    """
    model = MYCLASS.reg(predictors, response)
    fitted = model.fittedvalues
    resid = model.resid
    smooth = lowess(resid, fitted)
    marker_size = 8
    opacity = 0.5
    traces = [
        go.Scatter(
            x=fitted,
            y=resid,
            mode='markers',
            name='data',
            marker=dict(
                size=marker_size,
                opacity=opacity
            )
        ),
        go.Scatter(
            x=smooth[:, 0],
            y=smooth[:, 1],
            mode='lines',
            name='smoother',
            line=dict(
                width=1,
                color='red'
            )
        )
    ]

    return {
        'data': traces,
        'layout': dict(
            title='Residuals vs Fitted',
            xaxis={'title': 'Fitted Values'},
            yaxis={'title': 'Residuals'},
            plot_bgcolor='#e6e6e6',
            showlegend=False,
            hovermode='closest'
        )
    }

@app.callback(
    Output('qq-plot', 'figure'),
    [Input('predictors', 'value'),
     Input('response', 'value')])
def update_qq_plot(predictors, response):
    """
    Normal QQ Plot
    """
    model = MYCLASS.reg(predictors, response)
    resid_norm = model.get_influence().resid_studentized_internal
    qq = ProbPlot(resid_norm)
    theo = qq.theoretical_quantiles
    sample = qq.sample_quantiles
    marker_size = 8
    opacity = 0.5
    traces = [
        go.Scatter(
            x=theo,
            y=sample,
            mode='markers',
            name='data',
            marker=dict(
                size=marker_size,
                opacity=opacity
            )
        ),
        go.Scatter(
            x=theo,
            y=theo,
            type='scatter',
            mode='lines',
            name='line',
            line=dict(
                width=1,
                color='red'
            )
        )
    ]
        
    return {
        'data': traces,
        'layout': dict(
            title='Normal Q-Q',
            xaxis={'title': 'Theoretical Quantiles'},
            yaxis={'title': 'Standardized Residuals'},
            plot_bgcolor='#e6e6e6',
            showlegend=False,
            hovermode='closest'
        )
}

@app.callback(
    Output('scale-location-plot', 'figure'),
    [Input('predictors', 'value'),
    Input('response', 'value')])
def update_scale_loc_plot(predictors, response):
    """
    Scale-Location Plot
    """
    model = MYCLASS.reg(predictors, response)
    fitted = model.fittedvalues
    resid_norm = model.get_influence().resid_studentized_internal
    resid_norm_abs_sqrt = np.sqrt(np.abs(resid_norm))
    smooth = lowess(resid_norm_abs_sqrt, fitted)
    marker_size = 8
    opacity = 0.5
    traces = [
        go.Scatter(
            x=fitted,
            y=resid_norm_abs_sqrt,
            mode='markers',
            name='data',
            marker=dict(
                size=marker_size,
                opacity=opacity
            )
        ),
        go.Scatter(
            x=smooth[:, 0],
            y=smooth[:, 1],
            mode='lines',
            name='smoother',
            line=dict(
                width=1,
                color='red'
            )
        )
    ]
        
    return {
        'data': traces,
        'layout': dict(
            title='Scale Location',
            xaxis={'title': 'Fitted values'},
            yaxis={'title': 'Absolute Squared Normalized Residuals'},
            plot_bgcolor='#e6e6e6',
            showlegend=False,
            hovermode='closest'
        )
    }

@app.callback(
    Output('residual-leverage-plot', 'figure'),
    [Input('predictors', 'value'),
    Input('response', 'value')])
def update_resid_lever_plot(predictors, response):
    """
    Residual vs Leverage Plot
    """
    model = MYCLASS.reg(predictors, response)
    model_leverage = model.get_influence().hat_matrix_diag
    resid_norm = model.get_influence().resid_studentized_internal
    smooth = lowess(resid_norm, model_leverage)
    marker_size = 8
    opacity = 0.5
    traces = [
        go.Scatter(
            x=model_leverage,
            y=resid_norm,
            mode='markers',
            name='data',
            marker=dict(
                size=marker_size,
                opacity=opacity
            )
        ),
        go.Scatter(
            x=smooth[:, 0],
            y=smooth[:, 1],
            mode='lines',
            name='smoother',
            line=dict(
                width=1,
                color='red'
            )
        )
    ]

    return {
        'data': traces,
        'layout': dict(
            title='Residuals vs Leverage',
            xaxis={'title': 'Leverage'},
            yaxis={'title': 'Standardized Residuals'},
            plot_bgcolor='#e6e6e6',
            showlegend=False,
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=False)
