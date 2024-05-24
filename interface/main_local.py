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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import pandas as pd

import pandas as pd

def load_data(csv_file_path):
    """
    Reads a CSV file, checks if the time vector is equally spaced,
    deduces the sampling rate, and resamples the dataset at 32 Hz.
    Returns a DataFrame with columns 'Time [s]', 'X_acceleration',
    'Y_acceleration', and 'Z_acceleration'.

    Args:
        csv_file_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Resampled DataFrame with columns 'X_acceleration',
                      'Y_acceleration', 'Z_acceleration', and 'New_Time'.
    """
    # original data sampling rate

    orig_sampling_rate = 32 # [Hz]

    try:
        # Read the CSV file and skip the first row
        df = pd.read_csv(csv_file_path, header=None)

        # Remove the first row (index 0)
        df = df.iloc[1:]

        # Rename columns (adjust column names as needed)
        column_mapping = {
            0: 'Time [s]',
            1: 'X_acceleration',
            2: 'Y_acceleration',
            3: 'Z_acceleration',
            4: 'Absolute_acceleration'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Compute differences between consecutive time values
        time_diff = df['Time [s]'].diff()

        # Calculate the mean of time differences
        mean_time_diff = round(time_diff.mean(), 3)

        # Calculate the sampling rate

        sampling_rate = 1 / mean_time_diff

        # Create a new column 'New_Time' starting from 0 and incrementing by mean_time_diff
        df['New_Time'] = pd.Series(range(len(df))) * mean_time_diff

        # Set the index to the 'New_Time' column
        df.set_index('New_Time', inplace=True)

        # Convert the index to a datetime-like index
        df.index = pd.to_timedelta(df.index, unit='s')

        # Resample the dataframe to 32 Hz

        df = df.resample('31.25ms').mean()

        return df[['X_acceleration', 'Y_acceleration', 'Z_acceleration']]

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None



def normalize_data(data_subject):
    '''Return a dataframe with normalized datas for a certain subject '''
    num_transformer=make_pipeline(StandardScaler())
    num_transformer.fit(data_subject)
    return pd.DataFrame(num_transformer.transform(data_subject), columns=data_subject.columns)

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

def process_dataframe(df):
    """
    Processes a pandas DataFrame by removing the first and last 15% of rows,
    computing the square root of the sum of squares for each row,
    and returning the resulting values as a list.

    Args:
        df (pd.DataFrame): Input DataFrame with 3 columns.

    Returns:
        list: Square root of the sum of squares for each row.
    """
    # Calculate the number of rows to remove
    num_rows_to_remove = int(0.15 * len(df))

    # Remove first and last rows
    df = df.iloc[num_rows_to_remove:-num_rows_to_remove]

    # Compute the square root of the sum of squares for each row
    result = np.linalg.norm(df, axis=1)

    # Convert the result to a list
    result_list = result.tolist()

    return result_list

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame (replace with your actual data)
    data = {
        "Column1": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        "Column2": [5, 15, 25, 35, 45, 55, 65, 75, 85],
        "Column3": [2, 4, 6, 8, 10, 12, 14, 16, 18],
    }
    df = pd.DataFrame(data)

def pred(CSV_file_path: str) -> int:
    # print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    X_pred = load_data(CSV_file_path)

    if X_pred is None:
        y_pred = 0

    model = load_model()

    X_processed = process_dataframe(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")

    return y_pred
