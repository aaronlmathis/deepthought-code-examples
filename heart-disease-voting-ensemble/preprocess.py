import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np
import os


def load_and_preprocess_data(data_dir: str = 'data/raw'):
    """
    Loads and preprocesses heart disease datasets from multiple sources.

    This function is responsible for:
    - Loading several heart disease datasets from disk, handling different file formats.
    - Concatenating them into a single DataFrame for unified processing.
    - Converting all columns to numeric types and binarizing the target variable.
    - Engineering additional polynomial features to capture non-linear relationships.
    - Constructing a preprocessing pipeline for both numeric and categorical features,
      including imputation, scaling, and one-hot encoding.

    Args:
        data_dir (str): Directory containing the raw data files.

    Returns:
        X (pd.DataFrame): Feature matrix after initial engineering (before transformation).
        y (pd.Series): Binary target vector.
        preprocessor (ColumnTransformer): Scikit-learn transformer for full preprocessing.
    """

    # Define the expected column names for all datasets, ensuring consistency.
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    # List of all data files to be loaded and combined.
    data_files = [
        'processed.cleveland.data',
        'processed.va.data',
        'processed.switzerland.data',
        'reprocessed.hungarian.data'
    ]

    df_list = []
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        
        # Set up arguments for pandas.read_csv, handling missing values and file format differences.
        read_csv_kwargs = {
            'header': None,
            'names': column_names,
            'na_values': '?'
        }
        
        # The Hungarian dataset uses whitespace as a delimiter, unlike the others.
        if file_name == 'reprocessed.hungarian.data':
            read_csv_kwargs['delim_whitespace'] = True
        else:
            read_csv_kwargs['sep'] = ','

        # Only attempt to load files that exist, logging a warning if missing.
        if os.path.exists(file_path):
            df_single = pd.read_csv(file_path, **read_csv_kwargs)
            df_list.append(df_single)
        else:
            print(f"Data file not found and will be skipped: {file_path}")

    # If no files were loaded, raise an error and log it.
    if not df_list:
        print(f"No data files found in the specified directory: {data_dir}")
        raise FileNotFoundError(f"No data files found in {data_dir}")

    # Concatenate all loaded DataFrames into one, resetting the index.
    print(f"Loading and concatenating {len(df_list)} data files...")
    df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset has {len(df)} rows.")

    # Convert all columns to numeric types, coercing errors to NaN.
    df = df.apply(pd.to_numeric)

    # Binarize the target: 0 = no disease, 1 = presence of disease (any value > 0).
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Separate features and target variable.
    X = df.drop('target', axis=1)
    y = df['target']

    # Feature engineering: add squared terms for selected numeric features to capture non-linear effects.
    for col in ['age', 'trestbps', 'chol', 'thalach']:
        X[f'{col}_sq'] = X[col]**2

    # Identify categorical and numeric columns for preprocessing.
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    # Define a pipeline for numeric features: impute missing values with mean, then standardize.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define a pipeline for categorical features: impute with most frequent value, then one-hot encode.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    # Combine both pipelines into a ColumnTransformer for full preprocessing.
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='passthrough')  # Any columns not specified are passed through unchanged.

    return X, y, preprocessor
