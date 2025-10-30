import numpy as np  # type: ignore
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

import os

def load_data(file_path:str)->pd.DataFrame:

    with open(file_path, 'r') as file:
        df = pd.read_csv(file)

    return df

def data_cleaning(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()

    # since 65-18 years is the comming employment range
    if 'person_emp_length' in df.columns:
        df = df[df['person_emp_length'] < 47] # type: ignore

    # its seen that age > 70 their loan has status 0
    if 'person_age' in df.columns:
        df = df[df['person_age']<=70] # type: ignore

    if 'loan_int_rate' in df.columns:
        loan_rate_median = df['loan_int_rate'].median()
        df.loc[:,'loan_int_rate'] = df['loan_int_rate'].fillna(loan_rate_median)


    return df



if __name__ == "__main__":

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/credit_risk_dataset.csv')
    df = load_data(file_path)
    print(df.shape)

    df_cleaned = data_cleaning(df)
    print(df_cleaned.shape)

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/processed'), exist_ok=True)
    clean_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/processed/credit_risk_cleaned.csv')
    df_cleaned.to_csv(clean_file, index=False)



