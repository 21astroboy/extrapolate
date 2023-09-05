import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd


def linear_extrapolation(x):
    X = np.array(x.dropna().index.astype(int)).reshape(-1, 1)
    Y = np.array(x.dropna().values).reshape(-1, 1)
    if X.shape[0] > 0:
        f = LinearRegression().fit(X, Y)
        for i in x.index:
            v = x.loc[i]
            if v != v:
                v = f.predict([[int(i)]])[0][0]
                if v < 0:
                    v = 0
                x.loc[i] = v
    return x
