import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def price_lineplot(data: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots()
    ax = data.plot()
    return fig
