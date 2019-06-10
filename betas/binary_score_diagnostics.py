'''
This module contains implementations of a bokeh dashboard of model scores and
actual label. To use the dashboard, use the command
bokeh serve --show binary_score_diagnostics.py
'''

import numpy as np
import pandas as pd
from os.path import dirname, join
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from bokeh.models import ColumnDataSource, Legend, Slider, Label, CustomJS
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import Button, PreText, TextInput
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, column


# Scatterplot
scores = np.concatenate((np.random.uniform(low=0.9, high=1, size=100),
                         np.random.uniform(low=0.8, high=0.9, size=50),
                         np.random.uniform(low=0.7, high=0.8, size=25),
                         np.random.uniform(low=0.5, high=0.7, size=5),
                         np.random.uniform(low=0, high=0.1, size=100),
                         np.random.uniform(low=0.1, high=0.2, size=50),
                         np.random.uniform(low=0.2, high=0.3, size=25),
                         np.random.uniform(low=0.3, high=0.5, size=5),
                         np.random.uniform(low=0.5, high=1, size=20),
                         np.random.uniform(low=0, high=0.5, size=20)))
labels = np.concatenate((np.ones(180), np.zeros(180),
                         np.zeros(20), np.ones(20)))


def classify(scores, labels, threshold=0.5):
    pred_label = (scores > threshold) + 0
    cal_1 = pred_label + labels
    cal_2 = pred_label - labels
    result = pd.DataFrame({'scores': scores, 'actual_label': labels,
                           'pred_label': pred_label, 'group': ''})
    result.at[cal_1 == 2, 'group'] = 'TP'
    result.at[cal_1 == 0, 'group'] = 'TN'
    result.at[cal_2 == 1, 'group'] = 'FP'
    result.at[cal_2 == -1, 'group'] = 'FN'
    np.random.seed(0)
    noise = np.random.uniform(low=-0.3, high=0.3, size=len(result))
    result['position'] = result['actual_label'] + noise
    return result


target = classify(scores, labels, 0.65)

# Scatterplot
hline = ColumnDataSource(data=dict(x=[-0.3, 1.3],
                         y=[0.7, 0.7]))
TP = ColumnDataSource(data=dict(x=target['position'][target.group == 'TP'],
                                y=target['scores'][target.group == 'TP']))
TN = ColumnDataSource(data=dict(x=target['position'][target.group == 'TN'],
                                y=target['scores'][target.group == 'TN']))
FP = ColumnDataSource(data=dict(x=target['position'][target.group == 'FP'],
                                y=target['scores'][target.group == 'FP']))
FN = ColumnDataSource(data=dict(x=target['position'][target.group == 'FN'],
                                y=target['scores'][target.group == 'FN']))

p_scatter = figure(plot_width=380, plot_height=280,
                   title='Scatterplot of Model Scores')
r0 = p_scatter.circle('x', 'y', source=TP, size=3, color='navy')
r1 = p_scatter.circle('x', 'y', source=TN, size=3, color='green')
r2 = p_scatter.circle('x', 'y', source=FP, size=3, color='orange')
r3 = p_scatter.circle('x', 'y', source=FN, size=3, color='tomato')
p_scatter.line('x', 'y', source=hline, line_width=1.5, color='red',
               line_dash='dotdash')
p_scatter.xaxis.axis_label = 'Actual Label'
p_scatter.yaxis.axis_label = 'Scores'
p_scatter.xaxis.ticker = FixedTicker(ticks=[0, 1])
p_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_scatter.title.text_font_size = "20px"
legend = Legend(items=[('TP', [r0]), ('TN', [r1]), ('FP', [r2]), ('FN', [r3])],
                location="center")
p_scatter.add_layout(legend, 'right')


# ROC curve
fpr, tpr, thresholds = roc_curve(target.actual_label, target.scores)
roc_auc = auc(fpr, tpr)
threshold = 0.7
diff = thresholds - threshold
x_coord = fpr[abs(diff).argsort()[0]]
y_coord = tpr[abs(diff).argsort()[0]]
roc_vline = ColumnDataSource(data=dict(x=[x_coord, x_coord], y=[0, y_coord]))
roc_hline = ColumnDataSource(data=dict(x=[x_coord, 1], y=[y_coord, y_coord]))
roc = ColumnDataSource(data=dict(x=fpr, y=tpr))

p_roc = figure(plot_width=380, plot_height=280,
               title='ROC Curve of Model Scores')
p_roc.line('x', 'y', source=roc, line_width=2)
p_roc.line('x', 'y', source=roc_vline, line_width=1.5, color='red',
           line_dash='dotdash')
p_roc.line('x', 'y', source=roc_hline, line_width=1.5, color='red',
           line_dash='dotdash')
label_tpr = Label(x=0.4, y=0.3, text='TPR: ' + str(
                  round(y_coord, 3)), text_font_size='10pt')
label_tnr = Label(x=0.4, y=0.2, text='TNR: ' + str(
                  round(1 - x_coord, 3)), text_font_size='10pt')
p_roc.xaxis.axis_label = 'False Positive Rate'
p_roc.yaxis.axis_label = 'True Positive Rate'
p_roc.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_roc.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_roc.title.text_font_size = "20px"
p_roc.add_layout(label_tpr)
p_roc.add_layout(label_tnr)


# Precision and recall curve
precision, recall, thresholds_pr = precision_recall_curve(labels, scores)
thresholds_pr = np.append(thresholds_pr, 1)
diff_pr = thresholds_pr - 0.7
x_coord_pr = recall[abs(diff_pr).argsort()[0]]
y_coord_pr = precision[abs(diff_pr).argsort()[0]]
pr_vline = ColumnDataSource(data=dict(x=[x_coord_pr, x_coord_pr],
                                      y=[0, y_coord_pr]))
pr_hline = ColumnDataSource(data=dict(x=[0, x_coord_pr],
                                      y=[y_coord_pr, y_coord_pr]))
pr = ColumnDataSource(data=dict(x=recall, y=precision))

p_pr = figure(plot_width=380, plot_height=280,
              title='Precision and Recall Curve')
p_pr.line('x', 'y', source=pr, line_width=2)
p_pr.line('x', 'y', source=pr_vline, line_width=1.5, color='red',
          line_dash='dotdash')
p_pr.line('x', 'y', source=pr_hline, line_width=1.5, color='red',
          line_dash='dotdash')
label_pre = Label(x=0.4, y=0.3, text='Precision: ' + str(
                  round(y_coord_pr, 3)), text_font_size='10pt')
label_rec = Label(x=0.4, y=0.2, text='Recall: ' + str(
                  round(x_coord_pr, 3)), text_font_size='10pt')
p_pr.xaxis.axis_label = 'Recall'
p_pr.yaxis.axis_label = 'Precision'
p_pr.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_pr.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_pr.title.text_font_size = "20px"
p_pr.add_layout(label_pre)
p_pr.add_layout(label_rec)


# Bar plot
bar_data = ColumnDataSource(data=dict(
    act=['0', '1'],
    hit=[target.group.value_counts()['TN'],
         target.group.value_counts()['TP']],
    miss=[target.group.value_counts()['FP'],
          target.group.value_counts()['FN']]))
p_bar = figure(x_range=['0', '1'], plot_width=380, plot_height=100,
               title='')
p_bar.vbar_stack(['hit', 'miss'], x='act', width=0.3,
                 color=['steelblue', 'red'], source=bar_data, alpha=0.7,
                 legend=['Hit', 'Miss'])
p_bar.y_range.start = 0
p_bar.x_range.range_padding = 0.1
p_bar.xgrid.grid_line_color = None
p_bar.axis.minor_tick_line_color = None
p_bar.outline_line_color = None
p_bar.legend.location = "center"
p_bar.xaxis.axis_label = 'Actual Label'
p_bar.yaxis.axis_label = 'Frequency'


# Histogram
hist, edges = np.histogram(target.scores, bins=30)
hist_data = ColumnDataSource(data=dict(scores=hist, left=edges[:-1],
                                       right=edges[1:]))
p_hist = figure(plot_width=380, plot_height=180)
p_hist.quad(bottom=0, top='scores', left='left', right='right',
            source=hist_data, fill_color='steelblue', line_color='white')
p_hist.title.text_font_size = "20px"
p_hist.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_hist.xaxis.axis_label = 'Scores'
p_hist.yaxis.axis_label = 'Frequency'


# Data input
text_input = TextInput(value='', title='Data path: ')
slider = Slider(title='Threshold', start=0, end=1, value=0.7, step=0.01)
error_msg = PreText(text='')
roc_msg = PreText(text='Optimal Threshold by ROC: ', width=100, height=20)
pr_msg = PreText(text='Optimal Threshold by PR: ', width=100, height=20)
download = ColumnDataSource(data=dict(
    scores=target.scores,
    group=target.group))


def update_data(attrname, old, new):
    try:
        df = pd.read_csv(text_input.value)
        scores = np.array(df.scores)
        labels = np.array(df.labels)
        val = slider.value
        hline.data = dict(x=[-0.3, 1.3], y=[val, val])
        target = classify(scores, labels, val)

        # Scatterplot
        scatter = target.sample(n=500, random_state=0)
        TP.data = dict(x=scatter['position'][scatter.group == 'TP'],
                       y=scatter['scores'][scatter.group == 'TP'])
        TN.data = dict(x=scatter['position'][scatter.group == 'TN'],
                       y=scatter['scores'][scatter.group == 'TN'])
        FP.data = dict(x=scatter['position'][scatter.group == 'FP'],
                       y=scatter['scores'][scatter.group == 'FP'])
        FN.data = dict(x=scatter['position'][scatter.group == 'FN'],
                       y=scatter['scores'][scatter.group == 'FN'])

        # ROC
        fpr, tpr, thresholds = roc_curve(target.actual_label, target.scores)
        tnr = 1 - fpr
        tf = tpr - tnr
        optimal_threshold = thresholds[abs(tf).argsort()[0]]
        roc_msg.text = 'Optimal Threshold by ROC: ' + str(round(
                       optimal_threshold, 2))
        diff = thresholds - val
        x_coord = fpr[abs(diff).argsort()[0]]
        y_coord = tpr[abs(diff).argsort()[0]]
        roc_vline.data = dict(x=[x_coord, x_coord], y=[0, y_coord])
        roc_hline.data = dict(x=[x_coord, 1], y=[y_coord, y_coord])
        roc.data = dict(x=fpr, y=tpr)
        label_tpr.text = 'TPR: ' + str(round(y_coord, 3))
        label_tnr.text = 'TNR: ' + str(round(1 - x_coord, 3))

        # PR
        precision, recall, thresholds_pr = precision_recall_curve(labels,
                                                                  scores)
        thresholds_pr = np.append(thresholds_pr, 1)
        pr.data = dict(x=recall, y=precision)
        diff_pr = thresholds_pr - val
        tf_pr = precision - recall
        optimal_threshold_pr = thresholds_pr[abs(tf_pr).argsort()[0]]
        pr_msg.text = 'Optimal Threshold by PR: ' + str(round(
                      optimal_threshold_pr, 2))
        x_coord_pr = recall[abs(diff_pr).argsort()[0]]
        y_coord_pr = precision[abs(diff_pr).argsort()[0]]
        pr_vline.data = dict(x=[x_coord_pr, x_coord_pr], y=[0, y_coord_pr])
        pr_hline.data = dict(x=[0, x_coord_pr], y=[y_coord_pr, y_coord_pr])
        label_pre.text = 'Precision: ' + str(round(y_coord_pr, 3))
        label_rec.text = 'Recall: ' + str(round(x_coord_pr, 3))

        # Bar plot
        bar_data.data = dict(act=['0', '1'],
                             hit=[target.group.value_counts()['TN'],
                                  target.group.value_counts()['TP']],
                             miss=[target.group.value_counts()['FP'],
                                   target.group.value_counts()['FN']])

        # Histogram
        hist, edges = np.histogram(target.scores, bins=30)
        hist_data.data = dict(scores=hist, left=edges[:-1], right=edges[1:])

        # Download
        download.data = dict(scores=target.scores, group=target.group)

        # Error message
        error_msg.text = ''

    except FileNotFoundError:
        error_msg.text = 'File does not exist.'


text_input.on_change('value', update_data)
slider.on_change('value', update_data)
button = Button(label="Download", button_type="success")
button.callback = CustomJS(args=dict(source=download), code=open(
                           join(dirname(__file__), "download.js")).read())


grid = gridplot([[text_input, slider],
                 [column([error_msg, roc_msg, pr_msg, button]),
                 p_scatter, p_roc],
                 [None, column(p_bar, p_hist), p_pr]])

curdoc().add_root(grid)
