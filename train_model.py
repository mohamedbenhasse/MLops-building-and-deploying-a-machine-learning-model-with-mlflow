import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

if __name__ == "__main__":
    processed_train_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_train.csv"
    processed_test_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_test.csv"

    train_data = pd.read_csv(processed_train_data_path)
    test_data = pd.read_csv(processed_test_data_path)

    X_train = train_data.drop("Survived", axis=1)
    y_train = train_data["Survived"]

    with mlflow.start_run(run_name="Random Forest Training") as run:
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)

        mlflow.log_param("data_path", processed_train_data_path)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(clf, "random_forest_model")
