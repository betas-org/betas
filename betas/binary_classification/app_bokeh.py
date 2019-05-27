import numpy as np
import pandas as pd
from data import load_data, classify
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from bokeh.models import CategoricalColorMapper, ColumnDataSource, FactorRange
from bokeh.models import Legend, CustomJS, ColumnDataSource, Slider
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import column
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot

# Scatterplot
scores, actual_label = load_data()
df = classify(scores, actual_label, 0.5)
hline = ColumnDataSource(data=dict(x=[-0.3, 1.3], y=[0.5, 0.5]))
TP = ColumnDataSource(data=dict(x=df['position'][df.group == 'TP'],
                                y=df['scores'][df.group == 'TP']))
TN = ColumnDataSource(data=dict(x=df['position'][df.group == 'TN'],
                                y=df['scores'][df.group == 'TN']))
FP = ColumnDataSource(data=dict(x=df['position'][df.group == 'FP'],
                                y=df['scores'][df.group == 'FP']))
FN = ColumnDataSource(data=dict(x=df['position'][df.group == 'FN'],
                                y=df['scores'][df.group == 'FN']))

p_scatter = figure(plot_width=500, plot_height=400, title='Scatterplot of Model Scores')
r0 = p_scatter.circle('x', 'y', source=TP, size=3, color='navy')
r1 = p_scatter.circle('x', 'y', source=TN, size=3, color='green')
r2 = p_scatter.circle('x', 'y', source=FP, size=3, color='orange')
r3 = p_scatter.circle('x', 'y', source=FN, size=3, color='tomato')
p_scatter.line('x', 'y', source=hline, line_width=1.5, color='red', line_dash='dotdash')
p_scatter.xaxis.axis_label = 'Actual Label'
p_scatter.yaxis.axis_label = 'Scores'
p_scatter.xaxis.ticker = FixedTicker(ticks=[0, 1])
p_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_scatter.title.text_font_size = "20px"
legend = Legend(items=[('TP', [r0]), ('TN', [r1]), ('FP', [r2]), ('FN', [r3])],
                location="center")
p_scatter.add_layout(legend, 'right')


# ROC curve
scores, actual_label = load_data()
df = classify(scores, actual_label, 0.5)
fpr, tpr, thresholds = roc_curve(df.actual_label, df.scores)
tnr = 1 - fpr
tf = tpr - tnr
optimal_cutoff = thresholds[abs(tf).argsort()[0]]
roc_auc = auc(fpr, tpr)
threshold = 0.5
diff = thresholds - threshold
x_coord = fpr[abs(diff).argsort()[0]]
y_coord = tpr[abs(diff).argsort()[0]]


roc_vline = ColumnDataSource(data=dict(x=[x_coord, x_coord], y=[0, y_coord]))
roc_hline = ColumnDataSource(data=dict(x=[x_coord, 1], y=[y_coord, y_coord]))
roc = ColumnDataSource(data=dict(x=fpr, y=tpr))

p_roc = figure(plot_width=500, plot_height=400, title='ROC Curve of Model Scores')
p_roc.line('x', 'y', source=roc, line_width=2)
p_roc.line('x', 'y', source=roc_vline, line_width=1.5, color='red',
            line_dash='dotdash')
p_roc.line('x', 'y', source=roc_hline, line_width=1.5, color='red',
            line_dash='dotdash')
p_roc.xaxis.axis_label = 'False Positive Rate'
p_roc.yaxis.axis_label = 'True Positive Rate'
p_roc.title.text_font_size = "20px"

# Histogram
hist, edges = np.histogram(df.scores, bins=30)
hist_data = pd.DataFrame({'scores': hist,
                          'left': edges[:-1],
                          'right': edges[1:]})

p_hist = figure(plot_width=500, plot_height=400, title='Histogram of Model Scores')
p_hist.quad(bottom=0, top=hist_data['scores'],
       left=hist_data['left'], right=hist_data['right'],
       fill_color='steelblue', line_color='white')
p_hist.title.text_font_size = "20px"
p_hist.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_hist.xaxis.axis_label = 'Scores'
p_hist.yaxis.axis_label = 'Frequency'


def update_data(attrname, old, new):
    val = slider.value
    hline.data = dict(x=[-0.3, 1.3], y=[val, val])
    df = classify(scores, actual_label, val)
    TP.data = dict(x=df['position'][df.group == 'TP'],
                   y=df['scores'][df.group == 'TP'])
    TN.data = dict(x=df['position'][df.group == 'TN'],
                   y=df['scores'][df.group == 'TN'])
    FP.data = dict(x=df['position'][df.group == 'FP'],
                   y=df['scores'][df.group == 'FP'])
    FN.data = dict(x=df['position'][df.group == 'FN'],
                   y=df['scores'][df.group == 'FN'])

    fpr, tpr, thresholds = roc_curve(df.actual_label, df.scores)
    tnr = 1 - fpr
    tf = tpr - tnr
    optimal_cutoff = thresholds[abs(tf).argsort()[0]]
    roc_auc = auc(fpr, tpr)
    diff = thresholds - val
    x_coord = fpr[abs(diff).argsort()[0]]
    y_coord = tpr[abs(diff).argsort()[0]]
    roc_vline.data = dict(x=[x_coord, x_coord], y=[0, y_coord])
    roc_hline.data = dict(x=[x_coord, 1], y=[y_coord, y_coord])
    roc.data = dict(x=fpr, y=tpr)


slider = Slider(title='Cutoff', start=0, end=1, value=0.5, step=0.01)
slider.on_change('value', update_data)

# plots = row(p_scatter, p_roc)
# layout = column(slider, plots)
# layout = row(slider, p_scatter, p_roc)
grid = gridplot([[slider, None], [p_scatter, p_roc], [p_hist, None]])

curdoc().add_root(grid)
