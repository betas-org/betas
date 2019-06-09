'''
This module contains implementations of a bokeh dashboard of model scores and
actual label. To use the dashboard, use the command
bokeh serve --show app_bokeh.py
'''

import numpy as np
import pandas as pd
from os.path import dirname, join
from sklearn.metrics import precision_recall_curve, roc_curve
from bokeh.models import ColumnDataSource, Legend, Slider
from bokeh.models import Label, CustomJS, Plot
from bokeh.models.tickers import FixedTicker
from bokeh.models.widgets import Button, PreText, TextInput
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot
from bokeh.models.glyphs import Quad


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


target = pd.DataFrame(dict(scores=[0.5, 0.5, 0.5, 0.5],
                           position=[1, 0, 0, 1],
                           actual_label=[0, 0, 0, 0],
                           group=['TP', 'TN', 'FP', 'FN']))

# Scatterplot
hline = ColumnDataSource(data=dict(x=[],
                                   y=[]))
TP = ColumnDataSource(data=dict(x=[], y=[]))
TN = ColumnDataSource(data=dict(x=[], y=[]))
FP = ColumnDataSource(data=dict(x=[], y=[]))
FN = ColumnDataSource(data=dict(x=[], y=[]))

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
# p_scatter.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_scatter.title.text_font_size = "20px"
legend = Legend(items=[('TP', [r0]), ('TN', [r1]), ('FP', [r2]), ('FN', [r3])],
                location="center")
p_scatter.add_layout(legend, 'right')


# ROC curve
pre_roc = PreText(text='', width=100, height=20)

roc_vline = ColumnDataSource(data=dict(x=[], y=[]))
roc_hline = ColumnDataSource(data=dict(x=[], y=[]))
roc = ColumnDataSource(data=dict(x=[], y=[]))

p_roc = figure(plot_width=380, plot_height=280,
               title='ROC Curve of Model Scores')
p_roc.line('x', 'y', source=roc, line_width=2)
p_roc.line('x', 'y', source=roc_vline, line_width=1.5, color='red',
           line_dash='dotdash')
p_roc.line('x', 'y', source=roc_hline, line_width=1.5, color='red',
           line_dash='dotdash')
label_tpr = Label(x=0.4, y=0.3, text='', text_font_size='10pt')
label_tnr = Label(x=0.4, y=0.2, text='', text_font_size='10pt')
p_roc.xaxis.axis_label = 'False Positive Rate'
p_roc.yaxis.axis_label = 'True Positive Rate'
# p_roc.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
# p_roc.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_roc.title.text_font_size = "20px"
p_roc.add_layout(label_tpr)
p_roc.add_layout(label_tnr)

# Precision-Recall curve
pre_pr = PreText(text='', width=100, height=20)
pr_vline = ColumnDataSource(data=dict(x=[], y=[]))
pr_hline = ColumnDataSource(data=dict(x=[], y=[]))
pr = ColumnDataSource(data=dict(x=[], y=[]))

p_pr = figure(plot_width=380, plot_height=280,
              title='Precision and Recall Curve')
p_pr.line('x', 'y', source=pr, line_width=2)
p_pr.line('x', 'y', source=pr_vline, line_width=1.5, color='red',
          line_dash='dotdash')
p_pr.line('x', 'y', source=pr_hline, line_width=1.5, color='red',
          line_dash='dotdash')
label_pre = Label(x=0.4, y=0.3, text='', text_font_size='10pt')
label_rec = Label(x=0.4, y=0.2, text='', text_font_size='10pt')
p_pr.xaxis.axis_label = 'Recall'
p_pr.yaxis.axis_label = 'Precision'
# p_pr.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
# p_pr.yaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
p_pr.title.text_font_size = "20px"
p_pr.add_layout(label_pre)
p_pr.add_layout(label_rec)


# Bar plot
bar_data = ColumnDataSource(data=dict(act=['0', '1'], hit=[], miss=[]))
p_bar = figure(x_range=['0', '1'], plot_width=380, plot_height=280,
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
# hist, edges = np.histogram(target.scores, bins=30)
# hist_data = pd.DataFrame({'scores': hist,
#                           'left': edges[:-1],
#                           'right': edges[1:]})
#
# p_hist.quad(bottom=0, top=hist_data['scores'],
#             left=hist_data['left'], right=hist_data['right'],
#             fill_color='steelblue', line_color='white')
# p_hist.title.text_font_size = "20px"
# p_hist.xaxis.ticker = FixedTicker(ticks=np.arange(0, 1.1, 0.1))
# p_hist.xaxis.axis_label = 'Scores'
# p_hist.yaxis.axis_label = 'Frequency'
hist_source = ColumnDataSource(data=dict(left=[], top=[], right=[], bottom=[]))

plot = Plot(title=None, plot_width=380, plot_height=180,
            min_border=0, toolbar_location=None)
glyph = Quad(left="left", right="right", top="top", bottom="bottom",
             fill_color="#b3de69")
plot.add_glyph(hist_source, glyph)


download = ColumnDataSource(data=dict(
    scores=target.scores,
    group=target.group))

text_input = TextInput(value='', title='Data path: ', width=300, height=1)
read_error_msg = PreText(text='', width=50, height=20)
slider = Slider(title='Threshold', start=0, end=1, value=0.5, step=0.01)
button = Button(label="Download", button_type="success")
button.callback = CustomJS(args=dict(source=download),
                           code=open(join(dirname(__file__),
                                          "download.js")).read())


def update_data(attrname, old, new):
    data_path = text_input.value
    try:
        df = pd.read_csv(data_path)
#        is_file_loaded = True
        scores = np.array(df.scores)
        actual_label = np.array(df.actual_label)
        val = slider.value
        # Scatterplot
        hline.data = dict(x=[-0.3, 1.3], y=[val, val])
        target = classify(scores, actual_label, val)
        TP.data = dict(x=target['position'][target.group == 'TP'],
                       y=target['scores'][target.group == 'TP'])
        TN.data = dict(x=target['position'][target.group == 'TN'],
                       y=target['scores'][target.group == 'TN'])
        FP.data = dict(x=target['position'][target.group == 'FP'],
                       y=target['scores'][target.group == 'FP'])
        FN.data = dict(x=target['position'][target.group == 'FN'],
                       y=target['scores'][target.group == 'FN'])
        # ROC curve
        fpr, tpr, thresholds = roc_curve(target.actual_label, target.scores)
        diff = thresholds - val
        x_coord = fpr[abs(diff).argsort()[0]]
        y_coord = tpr[abs(diff).argsort()[0]]
        roc_vline.data = dict(x=[x_coord, x_coord], y=[0, y_coord])
        roc_hline.data = dict(x=[x_coord, 1], y=[y_coord, y_coord])
        roc.data = dict(x=fpr, y=tpr)
        label_tpr.text = 'TPR: ' + str(round(y_coord, 3))
        label_tnr.text = 'TNR: ' + str(round(1-x_coord, 3))

        # PR curve
        precision, recall, thresholds_pr = precision_recall_curve(actual_label,
                                                                  scores)
        pr.data = dict(x=recall, y=precision)
        thresholds_pr = np.append(thresholds_pr, 1)
        diff_pr = thresholds_pr - val
        x_coord_pr = recall[abs(diff_pr).argsort()[0]]
        y_coord_pr = precision[abs(diff_pr).argsort()[0]]
        pr_vline.data = dict(x=[x_coord_pr, x_coord_pr], y=[0, y_coord_pr])
        pr_hline.data = dict(x=[0, x_coord_pr], y=[y_coord_pr, y_coord_pr])
        label_pre.text = 'Precision: ' + str(round(y_coord_pr, 3))
        label_rec.text = 'Recall: ' + str(round(x_coord_pr, 3))

        # hist, edges = np.histogram(scores, bins=30)
        # hist_data = pd.DataFrame({'scores': hist, 'left': edges[:-1],
        #                           'right': edges[1:]})
        # p_hist.quad(bottom=0, top=hist_data['scores'],
        #             left=hist_data['left'], right=hist_data['right'],
        #             fill_color='steelblue', line_color='white')
        # hist_source.data=dict(left=edges[:-1], top=hist, right=edges[1:],
        #                       bottom=0)

        # Bar plot
        bar_data.data = dict(act=['0', '1'],
                             hit=[target.group.value_counts()['TN'],
                                  target.group.value_counts()['TP']],
                             miss=[target.group.value_counts()['FP'],
                                   target.group.value_counts()['FN']])

        download.data = dict(scores=target.scores, group=target.group)
    except FileNotFoundError:
        read_error_msg.text = 'File does not exist.'
    except Exception:
        read_error_msg.text = "Unexpected error"


# Update
text_input.on_change('value', update_data)
slider.on_change('value', update_data)

grid = gridplot([[text_input, slider],
                 [p_scatter, p_roc],
                 [p_bar, p_pr]])

curdoc().add_root(grid)
