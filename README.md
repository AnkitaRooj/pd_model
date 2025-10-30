# Credit Risk Analytics Model

A machine learning project for predicting credit risk using XGBoost, with MLflow tracking and DVC for experiment management.

<img width="1281" height="736" alt="Image" src="https://github.com/user-attachments/assets/1932dab0-9145-4d4b-b601-9b8a63c40c38" />

## Project Overview

This project implements a credit risk prediction model using XGBoost classifier. It includes:
- Data cleaning and preprocessing pipeline
- Feature engineering with SMOTE for handling class imbalance
- Model training with hyperparameter tuning using GridSearchCV
- Model evaluation and metrics tracking using MLflow
- Version control for data and models using DVC

## Project Structure

```
pd_model/
├── assets/                    # Model artifacts and parameters
├── datasets/                  # Data directory
│   ├── features/             # Preprocessed features
│   ├── processed/            # Cleaned datasets
│   └── splitted_data/        # Train/test splits
├── metrics/                  # Model evaluation metrics
├── models/                   # Saved model files
├── notebook/                 # Jupyter notebooks
│   ├── data_preprocessing.ipynb
│   └── eda.ipynb
├── src/                     # Source code
│   ├── data_clean.py       # Data cleaning functions
│   ├── data_preprocess.py  # Feature preprocessing
│   ├── evaluate.py         # Model evaluation
│   └── model_build.py      # Model training
|
|
├── requirements.txt         # Project dependencies
└── dvc.yaml                # DVC pipeline configuration
```

## Setup and Installation

1. Create a Python virtual environment:
```bash
python -m venv vpd
```

2. Activate the virtual environment:
```powershell
# On Windows
.\vpd\Scripts\Activate.ps1

# On Unix/MacOS
source vpd/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Pipeline

The data processing pipeline consists of three main stages:

1. **Data Cleaning** (`data_clean.py`):
   - Handles missing values
   - Filters outliers in employment length and age
   - Processes loan interest rates

2. **Preprocessing** (`data_preprocess.py`):
   - Feature standardization
   - One-hot encoding for categorical variables
   - SMOTE for class imbalance

3. **Model Training** (`model_build.py`):
   - XGBoost classifier
   - Hyperparameter tuning via GridSearchCV
   - Cross-validation with StratifiedKFold

## MLflow Tracking

The project uses MLflow to track:
- Model parameters
- Training metrics
- Model artifacts
- Evaluation results

To start the MLflow UI:
```bash
mlflow ui
```
Then visit `http://127.0.0.1:5000` in your browser.


## Model Evaluation

Model performance metrics are stored in `metrics/`:
- Classification report (JSON format)
- Confusion matrix visualization
- Accuracy, precision, and F1 scores

## DVC Data Version Control

The project uses DVC for data and model versioning. Key commands:

```bash
# Pull latest data
dvc pull

# Track changes in data
dvc add datasets/credit_risk_dataset.csv

# Update pipeline
dvc repro
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.