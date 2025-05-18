
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from pandas.core.indexes.base import Index
import pandas as pd
import matplotlib.pyplot as plt

def lasso_regression(X_train, X_test, y_train, y_test):   
    lasso_model = LassoCV(cv=5)
    lasso_model.fit(X_train,y_train)
    y_pred = lasso_model.predict(X_test)

    # Step 7: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Step 8: Print metrics
    print("📊 Lasso Regression Evaluation:")
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))
    print("R² Score:", round(r2, 4))


def linear_regression(X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train,y_train)
    y_pred = lr_model.predict(X_test)

    # Step 7: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Step 8: Print metrics
    print("📊 Linear Regression Evaluation:")
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))
    print("R² Score:", round(r2, 4))


def ridge_regression(X_train, X_test, y_train, y_test):
    ridge_model = RidgeCV(cv=5)
    ridge_model.fit(X_train,y_train)
    y_pred = ridge_model.predict(X_test)

    # Step 7: Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Step 8: Print metrics
    print("📊 Ridge Regression Evaluation:")
    print("RMSE:", round(rmse, 2))
    print("MAE:", round(mae, 2))
    print("R² Score:", round(r2, 4))
    return ridge_model, y_pred



def coefficients_plot(model : RidgeCV , features: Index ,title : str) -> None:   
     # Get coefficients
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    print(coef_df)
    # Plot
    plt.figure(figsize=(10,6))
    plt.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel("Coefficient Value")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.show()
