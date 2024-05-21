import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

if __name__ == "__main__":
    processed_train_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_train.csv"
    processed_test_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_test.csv"

    train_data = pd.read_csv(processed_train_data_path)
    test_data = pd.read_csv(processed_test_data_path)

    X_test = test_data.drop("Survived", axis=1)
    y_test = test_data["Survived"]

    model_name = "random_forest_model"
    model_version = 1

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run(run_name="Evaluate Model"):
        mlflow.log_metric("accuracy", accuracy)
