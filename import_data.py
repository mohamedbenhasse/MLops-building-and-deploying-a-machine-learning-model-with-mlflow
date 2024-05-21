import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    train_data["Age"] = train_data["Age"].fillna(-0.5)
    test_data["Age"] = test_data["Age"].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    train_data['AgeGroup'] = pd.cut(train_data["Age"], bins, labels=labels)
    test_data['AgeGroup'] = pd.cut(test_data["Age"], bins, labels=labels)
    train_data = train_data.fillna({"Embarked": "S"})
    combine = [train_data, test_data]
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
    age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}
    for dataset in combine:
        dataset['AgeGroup'] = dataset.apply(
            lambda row: age_title_mapping[row['Title']] if row['AgeGroup'] == 'Unknown' else row['AgeGroup'],
            axis=1
        )
    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
    for dataset in combine:
        dataset['AgeGroup'] = dataset['AgeGroup'].map(age_mapping)
    train_data = train_data.drop(['Ticket', 'Cabin', 'Name', 'Age'], axis=1)
    test_data = test_data.drop(['Ticket', 'Cabin', 'Name', 'Age'], axis=1)
    sex_mapping = {"male": 0, "female": 1}
    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map(sex_mapping).astype(int)
        dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)
    for dataset in [test_data]:
        dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
    train_data['FareBand'] = pd.qcut(train_data['Fare'], 4, labels=[1, 2, 3, 4])
    test_data['FareBand'] = pd.qcut(test_data['Fare'], 4, labels=[1, 2, 3, 4])
    train_data = train_data.drop(['Fare'], axis=1)
    test_data = test_data.drop(['Fare'], axis=1)

    return train_data, test_data

if __name__ == "__main__":
    train_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\train.csv"
    test_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\test.csv"
    processed_train_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_train.csv"
    processed_test_data_path = r"C:\Users\mohdm\Downloads\MLflow_Project\data\processed_test.csv"

    train_data, test_data = load_data(train_data_path, test_data_path)
    train_data, test_data = preprocess_data(train_data, test_data)

    with mlflow.start_run(run_name="Train Data Import"):
        mlflow.log_param("data_path", train_data_path)
        train_data.to_csv(processed_train_data_path, index=False)
        mlflow.log_artifact(processed_train_data_path)

    with mlflow.start_run(run_name="Test Data Import"):
        mlflow.log_param("data_path", test_data_path)
        test_data.to_csv(processed_test_data_path, index=False)
        mlflow.log_artifact(processed_test_data_path)
