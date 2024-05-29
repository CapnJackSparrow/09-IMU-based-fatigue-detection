import streamlit as st
import pandas as pd
import numpy as np 

from PIL import Image
import seaborn as sns

import datetime
import requests
import pickle
import time

from sklearn.preprocessing import RobustScaler



def load_short_data(csv_file_path):
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

        # Remove the first 2 rows
        df = df.iloc[2:]

        # Rename columns (adjust column names as needed)
        column_mapping = {
            0: 'time',
            1: 'seconds_elapsed',
            2: 'z',
            3: 'y',
            4: 'x'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Declare a robust scaler

        scaler = RobustScaler()

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Calculate the total length of the DataFrame
        total_duration = len(df)
        
        seconds = round(total_duration/32)

        # Calculate the start time for the 90 seconds window (center of data)
        window_start = (total_duration - 2880) // 2
        # Calculate the end time for the 90 seconds window (center of data)
        window_end = window_start + 2880
        # Extract data within a 90 seconds window located at the center of
        # original DataFrame
        df = df.iloc[window_start:window_end]

        return df[['x', 'y', 'z']], seconds

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None


def load_long_data(csv_file_path):
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

        # Remove the first 2 rows
        df = df.iloc[2:]

        # Rename columns (adjust column names as needed)
        column_mapping = {
            0: 'time',
            1: 'seconds_elapsed',
            2: 'z',
            3: 'y',
            4: 'x'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Declare a robust scaler

        scaler = RobustScaler()

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])

        # Calculate the total length of the DataFrame
        total_duration = len(df)
        
        seconds = round(total_duration/32)

        # Calculate the start time for the 30 minutes window (center of data)
        window_start = (total_duration - 57600) // 2
        # Calculate the end time for the 30 minutes window (center of data)
        window_end = window_start + 57600
        # Extract data within a 30 minutes window located at the center of
        # original DataFrame
        df = df.iloc[window_start:window_end]

        return df[['x', 'y', 'z']], seconds

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None



# load the model from disk

short_model_filename = "voting_clf_model.pkl"
long_model_filename = "long_model.pkl"

short_model = pickle.load(open(short_model_filename, 'rb'))
long_model = pickle.load(open(long_model_filename, 'rb'))

logo = Image.open('IMU_ilogo-transparent.png')

#define streamlit columns
col1, col2 = st.columns([3, 5])

col1.image(logo, width=240)

col2.title("Fatigue Detection Tool")

#st.write("First time? Upload a .csv file.")
st.write("Let's see how tired you are.")

model_type = st.radio(
    "Are you checking a short walk (at least 90 seconds), or long walk (at least 30 minutes)?",
    ["Short", "Long"],
    index=None,
)

st.write("You selected:", model_type)

uploaded_file = st.file_uploader("Upload your .csv with accelerometer data here", type=["csv", "zip"])

pred_class = None

if uploaded_file is not None:
    if model_type == "Short":
        dataframe, seconds = load_short_data(uploaded_file)
        df_copy = dataframe
        if seconds < 90:
            st.write("This sample isn't long enough. Please provide at least 90 seconds.")
        else:
            flat_chunk_of_90 = dataframe.to_numpy().flatten()
            dataframe = flat_chunk_of_90
            dataframe =  dataframe.reshape(1, -1) #the model asked for this reshaping, for one sample

            new_pred = short_model.predict(dataframe)
            new_pred_proba = short_model.predict_proba(dataframe)

            class_names = ['not tired', 'tired']

            pred_class = class_names[int(new_pred)]

            not_tired_conf_pc = "{:.0%}".format(new_pred_proba[0][0])


    if model_type == "Long":
        dataframe, seconds = load_long_data(uploaded_file) 
        df_copy = dataframe
        if seconds < 1800:
            st.write("This sample isn't long enough. Please provide at least 30 minutes.")
        else:
            flat_chunk_of_1800 = dataframe.to_numpy().flatten()
            dataframe = flat_chunk_of_1800
            dataframe =  dataframe.reshape(1, -1) #the model asked for this reshaping, for one sample

            new_pred = long_model.predict(dataframe)
            new_pred_proba = long_model.predict_proba(dataframe)

            class_names = ['not tired', 'tired']

            pred_class = class_names[int(new_pred)]

            not_tired_conf_pc = "{:.0%}".format(new_pred_proba[0][0])

    if pred_class is not None:
        progress_text = "Analyzing motion data... please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        st.write(f'You uploaded {seconds} seconds of data.')

        if pred_class == 'not tired':
            st.subheader(f'You are **:green[{pred_class}!]**')
            st.write(f'We are {not_tired_conf_pc} percent sure.')

        if pred_class == 'tired':
            st.subheader(f'You are **:red[{pred_class}!]**')
            st.write(f'We are {not_tired_conf_pc} percent sure')
            st.write('You should take a break.')


        #graph = sns.lineplot(df_copy["z"])
        st.line_chart(df_copy)