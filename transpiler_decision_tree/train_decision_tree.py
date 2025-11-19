import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def main():
    import os
    data_path = os.path.join('..', 'data', 'houses.csv')
    X = None
    y = None
    df = pd.read_csv(data_path)
    cols = ['size', 'nb_rooms']
    if all(c in df.columns for c in cols) and 'price' in df.columns:
        X = df[cols].values
        y = df['price'].values
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 3:
            X = df[numeric[:2]].values
            y = df[numeric[-1]].values

    model = DecisionTreeRegressor(max_depth=6, random_state=1)
    model.fit(X, y)

    joblib.dump(model, 'model_dt.joblib')
    print('Saved DecisionTreeRegressor to model_dt.joblib')
    print('Tree depth:', model.get_depth())


if __name__ == '__main__':
    main()
