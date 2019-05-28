"""
Basic Dash.
To run: python app_basic.py
"""
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from tool import load_data, classify
from sklearn.metrics import precision_recall_curve, roc_curve, auc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

scores, actual_label = load_data()

app.layout = html.Div([
    html.Div([
        dcc.Slider(id='slider', min=0, max=1, step=0.01, value=0.5),
        dcc.Input(id='input', value=30, type='number')
    ],
    style = {
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(id='scores-by-group')
    ],  style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='roc')
    ],  style={'width': '49%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='score-hist')
    ],  style={'width': '49%', 'display': 'inline-block'})
])

@app.callback(
    Output('scores-by-group', 'figure'),
    [Input('slider', 'value')])
def update_scatter(threshold):
    df = classify(scores=scores, actual_label=actual_label, threshold=threshold)
    marker_size = 4
    opacity = 0.5
    traces = [
        go.Scatter(
            x = df['position'][df.group == 'TP'],
            y = df['scores'][df.group == 'TP'],
            mode = 'markers',
            name = 'TP',
            marker = dict(
                size = marker_size,
                opacity = opacity
            )
        ),
        go.Scatter(
            x = df['position'][df.group == 'TN'],
            y = df['scores'][df.group == 'TN'],
            mode = 'markers',
            name = 'TN',
            marker = dict(
                size = marker_size,
                opacity = opacity
            )
        ),
        go.Scatter(
            x = df['position'][df.group == 'FP'],
            y = df['scores'][df.group == 'FP'],
            mode = 'markers',
            name = 'FP',
            marker = dict(
                size = marker_size,
                opacity = opacity
            )
        ),
        go.Scatter(
            x = df['position'][df.group == 'FN'],
            y = df['scores'][df.group == 'FN'],
            mode = 'markers',
            name = 'FN',
            marker = dict(
                size = marker_size,
                opacity = opacity
            )
        )]
    return {
        'data': traces,
        'layout': dict(
            xaxis = {'title': 'Actual Label', 'zeroline': False},
            yaxis = {'title': 'Scores', 'zeroline': False},
            shapes = [{'type': 'line', 'x0': -0.3, 'y0': threshold, 'x1': 1.3, 'y1': threshold,
                     'line': {'color': 'red', 'width': 4}
                    }],
            hovermode = 'closest'
        )
    }

@app.callback(
    Output('roc', 'figure'),
    [Input('slider', 'value')])
def update_roc(threshold):
    df = classify(scores=scores, actual_label=actual_label, threshold=threshold)
    fpr, tpr, thresholds = roc_curve(df.actual_label, df.scores)
    tnr = 1 - fpr
    tf = tpr - tnr
    optimal_cutoff = thresholds[abs(tf).argsort()[0]]
    roc_auc = auc(fpr, tpr)
    diff = thresholds - threshold
    x_coord = fpr[abs(diff).argsort()[0]]
    y_coord = tpr[abs(diff).argsort()[0]]

    traces = [
        go.Scatter(
            x = fpr,
            y = tpr,
            mode = 'lines',
            name = 'AUC: ' + str(round(roc_auc, 3)),
            showlegend = True,
            marker = dict(
                size = 5,
                opacity = 0.8
            )
        )]
    return {
        'data': traces,
        'layout': dict(
            xaxis = {'title': 'False Positive Rate'},
            yaxis = {'title': 'True Positive Rate'},
            shapes = [{'type': 'line', 'x0': x_coord, 'y0': 0, 'x1': x_coord, 'y1': y_coord,
                     'line': {'color': 'red', 'width': 2}
                    },
                      {'type': 'line', 'x0': x_coord, 'y0': y_coord, 'x1': 1, 'y1': y_coord,
                     'line': {'color': 'red', 'width': 2}
                    }],
            hovermode = 'closest',
            legend = dict(x=0.4, y=0.4, size=0.5)
        )
    }

@app.callback(
    Output('score-hist', 'figure'),
    [Input('slider', 'value'),
     Input('input', 'value')])
def update_hist(threshold, nbins):
    df = classify(scores=scores, actual_label=actual_label, threshold=threshold)
    split = pd.cut(df['scores'], nbins).to_frame()
    vline_height = np.array(split.iloc[:,0].value_counts().sort_values(ascending=False))[0]
    traces = [
        go.Histogram(
            x = df['scores'],
            xbins = dict(size = 1/(nbins-1)),
            autobinx = False,
            marker = dict(color = 'steelblue'),
            opacity = 0.7
        )
    ]
    return {
        'data': traces,
        'layout': dict(
            xaxis = {'title': 'Scores'},
            yaxis = {'title': 'Frequency'},
            bargap = 0.1,
            shapes = [{'type': 'line', 'x0': threshold, 'y0': 0, 'x1': threshold, 'y1': vline_height,
                     'line': {'color': 'red', 'width': 4}
                    }],
            hovermode = 'closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
