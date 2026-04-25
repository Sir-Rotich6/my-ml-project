import mlflow
import pandas as pd
from sklearn.base import BaseEstimator


def train(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    with mlflow.start_run():
        model.fit(X, y)
        mlflow.log_params(model.get_params())
    return model
