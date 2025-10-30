import numpy as np  # type: ignore
import pandas as pd
from imblearn.over_sampling import SMOTE # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import os

from data_clean import load_data

def data_clean (df:pd.DataFrame)->pd.DataFrame:

    if df.isna().sum().any():
        df = df.dropna()

    return df

def preprocess(df:pd.DataFrame)->pd.DataFrame:
    data = df.copy()

    del_col = ['loan_status','loan_intent','cb_person_default_on_file','person_home_ownership','loan_grade']
    if all( col in df.columns for col in del_col):
        loan_status = data['loan_status']
        data = data.drop(columns=['loan_status','loan_intent','cb_person_default_on_file','person_home_ownership','loan_grade'])
    print(data.head())

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(data_scaled, columns=data.columns)
    print(scaled_df.head())

    person_home = pd.get_dummies(df['person_home_ownership'],drop_first=True).astype(np.int8)
    loans_intent = pd.get_dummies(df['loan_intent'], drop_first=True).astype(np.int8)
    cb_person_default = df['cb_person_default_on_file'].map({'N':0, 'Y':1}).astype(np.int8)
    df_oth = pd.concat([scaled_df, person_home, loans_intent, cb_person_default,loan_status ], axis=1)
    print(df_oth.head())

    return df_oth



if __name__ == "__main__":

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/processed/credit_risk_cleaned.csv')
    df = load_data(file_path)
    print(df.shape)

    df_clean = data_clean(df)
    print(df_clean.shape)

    df_preprocessed = preprocess(df_clean)
    print(df_preprocessed.shape)

    processed_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/features')
    os.makedirs(processed_dir_path, exist_ok=True)

    preprocessed_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/features/credit_risk_preprocessed.csv')
    df_preprocessed.to_csv(preprocessed_file, index=False)

    print("Preprocessing completed and file saved.")


