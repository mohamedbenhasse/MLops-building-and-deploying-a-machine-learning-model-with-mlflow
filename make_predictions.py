import os
import requests
import pandas as pd
import json
import mlflow
from mlflow.tracking import MlflowClient

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

client = MlflowClient()

experiment_name = "Titanic Survival Prediction"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(f"Experiment {experiment_name} does not exist")

experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

def make_predictions(model_uri, input_data_path, output_path):
    input_data = pd.read_csv(input_data_path)
    data = input_data.to_json(orient='split')
    data_json = json.dumps(data)
    
    url = f"{os.environ['MLFLOW_TRACKING_URI']}/invocations"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=data_json)

    if response.status_code != 200:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
    
    predictions = pd.read_json(response.json(), orient='records')
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    model_uri = "models:/random_forest_model/1"
    input_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_test.csv"
    output_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\submission.csv"

    make_predictions(model_uri, input_data_path, output_path)
