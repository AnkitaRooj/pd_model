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

def model_evaluate(X_test,y_test):
    model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/xgboost_model.pkl'))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cls_rpt = classification_report(y_test, y_pred,output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../metrics'), exist_ok=True)
    metrics_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../metrics')

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("XGBoost_credit_risk_experiment")
    with mlflow.start_run(run_name="XGBoost_Model_Evaluation") as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_weighted", cls_rpt['weighted avg']['f1-score']) # type: ignore
        mlflow.log_metric("precision_weighted", cls_rpt['weighted avg']['precision']) # type: ignore

        print(f"Accuracy: {accuracy}")
        print("Classification Report:")
        print(cls_rpt)


        plt.figure(figsize=(8,6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')   
        confusion_mat_file_path = os.path.join(metrics_dir_path, "confusion_matrix.png")
        plt.savefig(confusion_mat_file_path,dpi=100)
        mlflow.log_artifact(confusion_mat_file_path)
        plt.close()

        cls_rpt_filePath = os.path.join(metrics_dir_path, "classification_report.json")
        with open(cls_rpt_filePath, "w") as f:
            json.dump(cls_rpt, f, indent=4)
        mlflow.log_artifact(cls_rpt_filePath)
    


if __name__ == "__main__":
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/splitted_data')
    X_test = load_data(os.path.join(dir_path, 'X_test.csv'))
    print(X_test.shape)

    y_test = load_data(os.path.join(dir_path, 'y_test.csv'))
    print(y_test.shape)

    X_test_clean = data_clean(X_test)
    print(X_test.shape)

    y_test_clean = data_clean(y_test)
    print(y_test.shape)

    model_evaluate(X_test_clean,y_test_clean)

    