import pandas as pd
from sklearn.base import BaseEstimator


def predict(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X), index=X.index)
