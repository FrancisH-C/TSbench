import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from typing import Callable, Optional, Type
from TSbench.TSdata.data import AnyData, Data


def visualize_loader(loader):
    df = None

    def interactive_select(
        start: Optional[int | str] = None,
        end: Optional[int | str] = None,
    ):
        nonlocal df
        df = loader.get_timeseries(
            start=start,
            end=end,
        )

    w = widgets.interact(
        interactive_select,
        start=1,
        end=1,
    )

    return w


def plot_df(df):
    # Define the data
    x = df.index.get_level_values("timestamp")
    y = df
    print(x)
    print(y)

    # Create the plot
    plt.plot(x, y)

    # # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Simple Line Plot")
    plt.show()
