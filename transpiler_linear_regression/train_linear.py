import os
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import pandas as pd

def main():
    data_path = os.path.join('..', 'data', 'houses.csv')
    X = None
    y = None
    cols = None

    df = pd.read_csv(data_path)
    if all(c in df.columns for c in ['size', 'nb_rooms', 'price']):
        X = df[['size', 'nb_rooms']].values
        y = df['price'].values
        cols = ['size', 'nb_rooms']
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 3:
            X = df[numeric[:2]].values
            y = df[numeric[-1]].values
            cols = numeric[:2]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, 'model.joblib')
    print('Saved model to model.joblib')
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)

if __name__ == '__main__':
    main()
