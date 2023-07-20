import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def lineplot(data: pd.Series) -> plt.Figure:
    """Plot a lineplot of the data."""
    fig, ax = plt.subplots()
    data.plot(ax=ax, alpha=0.5, marker="o", markersize=2, linestyle="-")
    return fig


def errLineplot(data: pd.DataFrame, y: str, err: str) -> plt.Figure:
    """Plot a lineplot of the data with error bars."""
    fig = lineplot(data=data[y])
    plt.fill_between(
        data.index.to_list(), data[y] - data[err], data[y] + data[err], alpha=0.5
    )
    return fig
