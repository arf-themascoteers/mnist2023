from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np


def plot(loss_list, acc_list):
    p = figure(
        y_axis_label="Loss",
        width=850,
        y_range=(0, 1),
        title="PyTorch ConvNet results"
    )
    p.extra_y_ranges = {
        "Accuracy": Range1d(start=0, end=100)
    }
    p.add_layout(
        LinearAxis(y_range_name="Accuracy", axis_label="Accuracy (%)"),
        "right"
    )
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(
        np.arange(len(loss_list)),
        np.array(acc_list) * 100,
        y_range_name="Accuracy",
        color="red"
    )
    show(p)