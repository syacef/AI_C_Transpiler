import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    data_path = os.path.join('..', 'data', 'diabetes.csv')
    X = None
    y = None
    cols = None
    if os.path.exists(data_path):
        try:
            import pandas as pd
            df = pd.read_csv(data_path)
            if 'Outcome' in df.columns:
                numeric = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'Outcome' in numeric:
                    numeric.remove('Outcome')
                if len(numeric) >= 1:
                    X = df[numeric].values
                    y = df['Outcome'].values
                    cols = numeric
        except Exception:
            X = None

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, 'model.joblib')
    print('Saved logistic model to model.joblib')
    print('Classes:', model.classes_)
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)


if __name__ == '__main__':
    main()
