'''This script is the main script for the app intended to work on the local
machine.
Following is the flow:
    - feed data as a CSV file containing IMU acceleration data
    - preprocessed the data
    - load model
    - predict'''

import pandas as pd
import pickle
import numpy as np
import xgboost as xgb


def preprocess_acceleration_data(csv_file_path):
    """
    Reads a CSV file, renames columns, and returns a DataFrame with acceleration
    columns.

    Args:
        csv_file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns 'Time [s]',
                                             'X_acceleration',
                                             'Y_acceleration',
                                             'Z_acceleration'.
    """
    try:
        # Read the CSV file and skip the first row
        df = pd.read_csv(csv_file_path, skiprows=[1], header=None)

        # Rename columns (adjust column names as needed)
        column_mapping = {
            0: 'Time [s]',
            1: 'X_acceleration',
            2: 'Y_acceleration',
            3: 'Z_acceleration'
        }
        df.rename(columns=column_mapping, inplace=True)

        return df
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None



def load_model():
    """
    Loads a pre-trained boosting model from a saved file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded boosting model.
    """

    # Load the model assuming it is located in the same folder as the "main"
    # file and it is named "model.pkl"
    model_file_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    try:
        with open(model_file_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        return loaded_model
    except FileNotFoundError:
        print(f"Error: Model file '{model_file_path}' not found.")
        return None
