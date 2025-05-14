import pandas as pd
import numpy as np


def calculate_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    return lower_bound, upper_bound

def highly_correlated_features(variance_filterd_df: pd.DataFrame, target:str) -> list:
    corr_matrix = variance_filterd_df.corr().abs()  # Absolute correlation values
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    features_to_drop = []
    for column in upper.columns:
        for row in upper.index:
            if(not(np.isnan(upper.loc[row,column])) and (upper.loc[row, column]>0.9 and (not(row==column)))):
                # print(column, row,df[row].corr(df["Earned Premium 2021"]),df[column].corr(df["Earned Premium 2021"]) )
                feature1_corr =variance_filterd_df[row] 
                feature2_corr = variance_filterd_df[column]
                feature_to_drop = row if feature1_corr.corr(variance_filterd_df[target]) >feature2_corr.corr(variance_filterd_df[target]) else column
                features_to_drop.append(feature_to_drop)
    return features_to_drop