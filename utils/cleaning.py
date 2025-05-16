import pandas as pd
import numpy as np
import ppscore as pps

def calculate_outliers(df, column_name):
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR
    return lower_bound, upper_bound

def highly_correlated_features(df: pd.DataFrame, target:str) -> list:
    corr_matrix = df.corr().abs()  # Absolute correlation values
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # print(upper)
    features_to_drop = []
    for column in upper.columns:
        for row in upper.index:
            if(not(np.isnan(upper.loc[row,column])) and (upper.loc[row, column]>0.9) and (target not in [row, column])):
                print(column, row,df[column].corr(df[target]),df[row].corr(df[target]) )
                feature1_corr =df[row] 
                feature2_corr = df[column]
                score_row = np.float32(pps.score(df, column, target)["ppscore"]).round(4)
                score_column = np.float32(pps.score(df, row, target)["ppscore"]).round(4)
                if(score_column == 0 and score_row == 0):
                    feature_to_drop = row if feature1_corr.corr(df[target]) < feature2_corr.corr(df[target]) else column
                else:
                    feature_to_drop = row if score_row<score_column else column
                print(f"feature_to_drop: {feature_to_drop}, predictive score {row}: {score_row}, predictive score {column}: {score_column}")
                features_to_drop.append(feature_to_drop)
    return features_to_drop