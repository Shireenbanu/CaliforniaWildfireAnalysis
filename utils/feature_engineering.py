import ppscore as pps
import plotly.express as px
import pandas as pd
import numpy as np

def feature_stats_histogram( df: pd.DataFrame, column: str, target:str)-> dict:
    fig = px.histogram(df, x= column, nbins=20, title=column)
    fig.show()
    score_column = np.float32(pps.score(df, column, target)["ppscore"]).round(4)
    print(f"column : {column}")
    print(f"Predictive Power Score: {score_column:.4f}")
    print(f"Correlation with Target: {df[column].corr(df[target])}")
    print(f"Skewness of the feature: {df[column].skew()}")
    return {"PredictivePower": score_column.round(4), "Correlation":df[column].corr(df[target])}