import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def lineplot(data: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots()
    data.plot(ax=ax)
    return fig
