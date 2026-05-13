import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import pickle
import sys


def train(input_path, model_path):
    df = pd.read_csv(input_path)
    X = df.drop(columns=['ScoreDelta'])
    y = df['ScoreDelta']

    mlflow.set_experiment("UDisc_Generalized_Prediction")
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        mlflow.log_metric("r2_score", model.score(X, y))
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
