import streamlit as st
import pandas as pd
import numpy as np

import datetime
import requests
import pickle

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
            0: 'time',
            1: 'seconds_elapsed',
            2: 'z',
            3: 'y',
            4: 'x'
        }
        df.rename(columns=column_mapping, inplace=True)

        ###truncate to ten sf
        df['time'] = df['time'].astype(str).str[:10] + '.' + df['time'].astype(str).str[10:]

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')



        # Compute differences between consecutive time values
        time_diff = df['time'].diff()

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

        return df[['x', 'y', 'z']]

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return None



from PIL import Image

# load the model from disk

filename = "voting_clf_model.pkl"
loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)

logo = Image.open('logo_v01.jpg')

st.image(logo)

st.title("Fatigue Detection")

st.write("Upload a .csv file")

# user_s_rate = st.radio(
#     "Your data's sampling rate?",
#     ["32Hz", "96Hz", "100Hz", "200Hz", "500Hz", "Attempt auto-detection"],
#     index=None,
# )

#st.write("You selected:", user_s_rate)

uploaded_file = st.file_uploader("Upload your .csv here", type=["csv", "zip"])

if uploaded_file is not None:

    dataframe = load_data(uploaded_file)

    ### should be a function, cheap and nasty version for monday morning
    chunk_of_90 = dataframe[:2880]
    flat_chunk_of_90 = chunk_of_90.to_numpy().flatten()
    dataframe = flat_chunk_of_90
    dataframe =  dataframe.reshape(1, -1) #the model asked for this reshaping, for one sample

    ##add pre-processing here

    ##add getting a prediction here
    new_pred = loaded_model.predict(dataframe)
    new_pred_proba = loaded_model.predict_proba(dataframe)

    class_names = ['not tired', 'tired']

    pred_class = class_names[np.argmax(new_pred)]

    not_tired_conf_pc = "{:.0%}".format(new_pred_proba[0][0])

    st.write(f'You are {pred_class}! We are {not_tired_conf_pc} percent sure.')
