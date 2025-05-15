import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

def histogram_and_box_plot(column_name: str, df: pd.DataFrame) -> None:
    # histogram
    fig1 = px.histogram(df, x=column_name, nbins = 20, title = column_name)
    # boxplot
    fig2= px.box(df, y=column_name, title="Box Plot for Earned Premium 2020")
     
    fig = make_subplots(rows=1, cols = 2, subplot_titles=(f'Histogram of {column_name}', f'Boxplot of {column_name}'))

    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)
    
    fig.update_xaxes(title_text= column_name, row=1, col=1)
    fig.update_yaxes(title_text="Frequency",row=1, col=1)
    fig.update_xaxes(title_text=column_name, row =1, col=2)
    fig.show()


def scatter_plot(df = pd.DataFrame, x_label = str, y_label = str)-> None :
    plt.figure(figsize=(4,3))
    sns.scatterplot(x=df[x_label], y=df[y_label], alpha = 0.5)
    plt.title(f"{y_label} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def correlation_matrix(df : pd.DataFrame) -> None :
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()

