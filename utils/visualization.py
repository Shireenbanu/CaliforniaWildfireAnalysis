import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots


def histogram_and_box_plot(column_name: str, df: pd.DataFrame):
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