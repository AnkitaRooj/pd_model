import numpy as np # type: ignore 
import pandas as pd  #type: ignore
from imblearn.over_sampling import SMOTE  #type: ignore
import matplotlib.pyplot as plt  #type: ignore
import seaborn as sns #type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold #type: ignore
import xgboost as xgb  #type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix #type: ignore
import json
import yaml #type: ignore
import mlflow #type: ignore
import mlflow.xgboost  #type: ignore
from mlflow.models.signature import infer_signature  #type: ignore
from data_preprocess import data_clean
from data_clean import load_data
import os
import joblib  #type: ignore

def class_imbalance(df:pd.DataFrame)->pd.DataFrame:
    smote = SMOTE()
    df = df.copy()

    if 'loan_status' in df.columns:
        X = df.drop('loan_status',axis=1)
        y = df['loan_status']

    X_smote,y_smote=smote.fit_resample(X,y)
    return X_smote,y_smote  # type: ignore

def data_split(df:pd.DataFrame):
    df = df.copy()
    if 'loan_status' in df.columns:
        X = df.drop('loan_status',axis=1)
        y = df['loan_status']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def model_training(X_train, X_test, y_train, y_test):

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("XGBoost_credit_risk_experiment")

    assets_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
    os.makedirs(assets_dir_path, exist_ok=True)

    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.9], # Subsample ratio of the training instance
    'colsample_bytree': [0.7, 1.0]
    }
    n_splits = 5
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss')
    with mlflow.start_run(run_name="XGBoost_Hyperparameter_Tuning") as run:
        run_id = run.info.run_id
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='accuracy', # Choose your main evaluation metric
            cv=skfold,          # Use the defined K-Fold object
            verbose=1,         # Controls the verbosity: 1 is standard
            n_jobs=-1          # Use all available cores for parallel processing
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        mlflow.log_dict(param_grid, "search_space/full_param_grid.json")
        mlflow.log_params(best_params)

        # Save to YAML file
        yaml_file_path = os.path.join(assets_dir_path, "best_params.yaml")
        with open(yaml_file_path, "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)
        mlflow.log_artifact(yaml_file_path)


        signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.xgboost.log_model( # type: ignore
            xgb_model=best_model,
            artifact_path="xgboost_model",
            signature=signature,
            # Register the model in the MLflow Model Registry
            registered_model_name="BestXGBoostClassifier" 
        )
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../models'), exist_ok=True)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../models/xgboost_model.pkl')
        joblib.dump(best_model,model_path)
        print(f"Trained xgboost model and logged with MLflow")

    print(f"\nMLflow Run ID: {run_id}")
    print(f"Best Parameters: {best_params}")

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/features/credit_risk_preprocessed.csv')
    df = load_data(file_path)
    print(df.shape)

    df_clean = data_clean(df)
    print(df_clean.shape) 

    X_smote,y_smote = class_imbalance(df_clean)
    print(X_smote.shape, y_smote.shape) # type: ignore
    print(y_smote.value_counts()) # type: ignore

    df = pd.concat([X_smote, y_smote], axis=1)  # type: ignore
    X_train, X_test, y_train, y_test = data_split(df)  # type: ignore

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    processed_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/splitted_data')
    os.makedirs(processed_dir_path, exist_ok=True)
    X_train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/splitted_data/X_train.csv')
    X_test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/splitted_data/X_test.csv')
    y_train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/splitted_data/y_train.csv')
    y_test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/splitted_data/y_test.csv')
    X_train.to_csv(X_train_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_train.to_csv(y_train_file, index=False)
    y_test.to_csv(y_test_file, index=False)

    model_training(X_train, X_test, y_train, y_test)