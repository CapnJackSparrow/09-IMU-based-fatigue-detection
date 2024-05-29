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



# def load_data(csv_file_path):
#     """
#     Reads a CSV file, checks if the time vector is equally spaced,
#     deduces the sampling rate, and resamples the dataset at 32 Hz.
#     Returns a DataFrame with columns 'Time [s]', 'X_acceleration',
#     'Y_acceleration', and 'Z_acceleration'.

#     Args:
#         csv_file_path (str): Path to the input CSV file.

#     Returns:
#         pd.DataFrame: Resampled DataFrame with columns 'X_acceleration',
#                       'Y_acceleration', 'Z_acceleration', and 'New_Time'.
#     """
#     # original data sampling rate

#     orig_sampling_rate = 32 # [Hz]

#     try:
#         # Read the CSV file and skip the first row
#         df = pd.read_csv(csv_file_path, header=None)

#         # Remove the first row (index 0)
#         df = df.iloc[1:]

#         # Rename columns (adjust column names as needed)
#         column_mapping = {
#             0: 'time',
#             1: 'seconds_elapsed',
#             2: 'z',
#             3: 'y',
#             4: 'x'
#         }
#         df.rename(columns=column_mapping, inplace=True)

#         ###truncate to ten sf
#         df['time'] = df['time'].astype(str).str[:10] + '.' + df['time'].astype(str).str[10:]

#         # Convert all columns to numeric
#         for col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')



#         # Compute differences between consecutive time values
#         time_diff = df['time'].diff()

#         # Calculate the mean of time differences
#         mean_time_diff = round(time_diff.mean(), 3)

#         # Calculate the sampling rate

#         sampling_rate = 1 / mean_time_diff

#         # Create a new column 'New_Time' starting from 0 and incrementing by mean_time_diff
#         df['New_Time'] = pd.Series(range(len(df))) * mean_time_diff

#         # Set the index to the 'New_Time' column
#         df.set_index('New_Time', inplace=True)

#         # Convert the index to a datetime-like index
#         df.index = pd.to_timedelta(df.index, unit='s')

#         # Resample the dataframe to 32 Hz

#         df = df.resample('31.25ms').mean()

#         return df[['x', 'y', 'z']]

#     except FileNotFoundError:
#         print(f"Error: CSV file '{csv_file_path}' not found.")
#         return None



# from PIL import Image

# # load the model from disk

# filename = "voting_clf_model.pkl"
# loaded_model = pickle.load(open(filename, 'rb'))
# #result = loaded_model.score(X_test, Y_test)
# #print(result)

# logo = Image.open('logo_v01.jpg')

# st.image(logo, caption="he sad")

# st.title("Some text, and a logo?")

# st.write("First time? Upload a .csv file")

# # user_s_rate = st.radio(
# #     "Your data's sampling rate?",
# #     ["32Hz", "96Hz", "100Hz", "200Hz", "500Hz", "Attempt auto-detection"],
# #     index=None,
# # )

# #st.write("You selected:", user_s_rate)

# uploaded_file = st.file_uploader("Upload your .csv here", type=["csv", "zip"])

# if uploaded_file is not None:

#     dataframe = load_data(uploaded_file)

#     ### should be a function, cheap and nasty version for monday morning
#     chunk_of_90 = dataframe[:2880]
#     flat_chunk_of_90 = chunk_of_90.to_numpy().flatten()
#     dataframe = flat_chunk_of_90
#     dataframe =  dataframe.reshape(1, -1) #the model asked for this reshaping, for one sample

#     ##add pre-processing here

#     ##add getting a prediction here
#     new_pred = loaded_model.predict(dataframe)
#     new_pred_proba = loaded_model.predict_proba(dataframe)

#     class_names = ['not tired', 'tired']

#     pred_class = class_names[np.argmax(new_pred)]

#     not_tired_conf_pc = "{:.0%}".format(new_pred_proba[0][0])

#     st.write(f'You are {pred_class}! We are {not_tired_conf_pc} percent sure.')

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



# load the model from disk

filename = "voting_clf_model.pkl"
loaded_model = pickle.load(open(filename, 'rb'))

logo = Image.open('IMU_ilogo-transparent.png')

#define streamlit columns
col1, col2 = st.columns([3, 5])

col1.image(logo, width=240)

col2.title("Fatigue Detection Tool")

#st.write("First time? Upload a .csv file.")
st.write("Let's see how tired you are.")

# user_s_rate = st.radio(
#     "Your data's sampling rate?",
#     ["32Hz", "96Hz", "100Hz", "200Hz", "500Hz", "Attempt auto-detection"],
#     index=None,
# )

#st.write("You selected:", user_s_rate)

uploaded_file = st.file_uploader("Upload your .csv with accelerometer data here", type=["csv", "zip"])

if uploaded_file is not None:

    dataframe, seconds = load_data(uploaded_file)
    df_copy = dataframe
    if seconds < 90:
        st.write("This sample isn't long enough. Please provide at least 90 seconds.")

    flat_chunk_of_90 = dataframe.to_numpy().flatten()
    dataframe = flat_chunk_of_90
    dataframe =  dataframe.reshape(1, -1) #the model asked for this reshaping, for one sample

    new_pred = loaded_model.predict(dataframe)
    new_pred_proba = loaded_model.predict_proba(dataframe)

    class_names = ['not tired', 'tired']

    pred_class = class_names[np.argmax(new_pred)]

    not_tired_conf_pc = "{:.0%}".format(new_pred_proba[0][0])


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
