import streamlit as st
import pandas as pd

import datetime
import requests

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



from PIL import Image

logo = Image.open('logo_v01.jpg')

st.image(logo, caption="he sad")

st.title("Some text, and a logo?")

st.write("First time? Upload a .csv file")

uploaded_file = st.file_uploader("Upload your .csv here", type=["csv", "zip"])

if uploaded_file is not None:
    
    dataframe = load_data(uploaded_file)

    ##add pre-processing here

    ##add getting a prediction here

    st.button('Does this button do anything?')

    st.write(f'Your file is {len(dataframe)} and that is a function run on a dataframe!')
