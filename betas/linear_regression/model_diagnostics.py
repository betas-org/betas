"""
Basic Dash to present linear regression model assumptions diagnostics
To run: python model_diagnostics.py
"""

#import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import plotly.graph_objs as go
from statsmodels.graphics.gofplots import ProbPlot

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from analysis_plot import analysis_plot


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
lowess = sm.nonparametric.lowess

# temp dataset: iris
DF = sns.load_dataset('iris')
COLS = DF.columns
# filter out non-numerical variables
MYCLASS = analysis_plot(DF)

app.layout = html.Div([
    html.H1('Linear Regression Model Diagnostics',
            style={'textAlign': 'center'}),
    html.Div("Hi, there should be some instruction here. \
              Next step, how to understand the plots."),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='predictors',
                options=[{'label': i, 'value': i} for i in COLS],
                value=[COLS[0]],
                multi=True
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='response',
                options=[{'label': i, 'value': i} for i in COLS],
                value=[COLS[1]]
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
