'''This class is used to import the data from the original raw dataset.'''

import numpy as np
import os
import pandas as pd

class DataLoader:
    def __init__(self):
        # Initialize the list of subjects
        self.subjects = [f"S{i}" for i in range (1,24)]

        # Initialize the mapping between the type of events recorded in the dataset
        # and the middle path for the respective data

        self.events = {"morning": "1.morning", "evening": "2.evening"}

        #Initialize the list of data types in the dataset

        self.types_of_data = ["ACC", "BVP", "EDA", "EEG", "HR", "IBI", "TEMP", ]

    def data_summary(self):
        '''Return general information about the dataset as a Pandas DataFrame.'''
        try:
            # Try loading the <<Subject list>> sheet of the <<general_info.csv>>
            # file from the <<data>> folder into a Pandas dataframe
            df = pd.read_csv("data/general_info.csv", sheet_name="Subject list")
            #Return the dataframe containing general information about the dataset
            return df

        except FileNotFoundError:
            #If the <<general_info.csv>> cannot be found (e.g. wrong path),
            # return a warning message
            print("Error: 'general_info.csv' not found. Make sure the file exists or that the path to the dataset is correctly set.")
            return None

    def load_ACC_data(self, subject = "S1", event = "morning"):
        '''Load the ACC - acceleration data for the specified subject and event
        into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/ACC.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, names = ["X_acceleration",
                    "Y_acceleration", "Z_acceleration"], dtype=float)
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
                starting_time = df.iloc[0, 0]
                # Load the data sampling rate from the first value in the second
                # row of the csv file
                sampling_rate = df.iloc[1, 0]
                # Removes the first two rows of the DataFrame, as they contain
                # info about the measurment starting time and sampling rate
                df = df.tail(-2).reset_index(drop=True)
                # Determine the number of samples in the data
                number_of_samples = len(df)
                # Generate the time vector based on the measurement starting time
                # and sampling rate
                time_vector = np.arange(starting_time, starting_time +
                            number_of_samples / sampling_rate, 1 / sampling_rate)
                #Add the UNIX time vector to the Pandas DataFrame
                df["UNIX_time"] = time_vector
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['UNIX_time'].max() - df['UNIX_time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['UNIX_time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['UNIX_time'] >= window_start) & (df['UNIX_time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None

    def load_BVP_data(self, subject = "S1", event = "morning"):
        '''Load the BVP - blood volume pulse data for the specified subject and
        event into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/BVP.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, names = ["Blood_volume_pulse"],
                                dtype=float)
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
                starting_time = df.iloc[0, 0]
                # Load the data sampling rate from the first value in the second
                # row of the csv file
                sampling_rate = df.iloc[1, 0]
                # Removes the first two rows of the DataFrame, as they contain
                # info about the measurment starting time and sampling rate
                df = df.tail(-2).reset_index(drop=True)
                # Determine the number of samples in the data
                number_of_samples = len(df)
                # Generate the time vector based on the measurement starting time
                # and sampling rate
                time_vector = np.arange(starting_time, starting_time +
                            number_of_samples / sampling_rate, 1 / sampling_rate)
                #Add the UNIX time vector to the Pandas DataFrame
                df["UNIX_time"] = time_vector
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['UNIX_time'].max() - df['UNIX_time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['UNIX_time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['UNIX_time'] >= window_start) & (df['UNIX_time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None

    def load_EDA_data(self, subject = "S1", event = "morning"):
        '''Load the EDA - electrodermal activity data for the specified subject and
        event into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/EDA.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, names = ["Electrodermal_activity"],
                                dtype=float)
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
                starting_time = df.iloc[0, 0]
                # Load the data sampling rate from the first value in the second
                # row of the csv file
                sampling_rate = df.iloc[1, 0]
                # Removes the first two rows of the DataFrame, as they contain
                # info about the measurment starting time and sampling rate
                df = df.tail(-2).reset_index(drop=True)
                # Determine the number of samples in the data
                number_of_samples = len(df)
                # Generate the time vector based on the measurement starting time
                # and sampling rate
                time_vector = np.arange(starting_time, starting_time +
                            number_of_samples / sampling_rate, 1 / sampling_rate)
                #Add the UNIX time vector to the Pandas DataFrame
                df["UNIX_time"] = time_vector
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['UNIX_time'].max() - df['UNIX_time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['UNIX_time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['UNIX_time'] >= window_start) & (df['UNIX_time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None

    def load_EEG_data(self, subject = "S1", event = "morning"):
        '''Load the EEG - electroencephalogram data for the specified subject and
        event into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/EEG.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path,
                                 skiprows = 1,
                                 names = ["Observation", "Time", "Delta",
                                          "Theta", "Alpha1", "Alpha2", "Beta1",
                                          "Beta2", "Gamma1", "Gamma2",
                                          "Attention", "Meditation", "Derived",
                                          "Total_power", "Class"])
                # Convert the data to numeric values where possible
                df = df.apply(pd.to_numeric, errors="coerce").fillna(df)
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['Time'].max() - df['Time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['Time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['Time'] >= window_start) & (df['Time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None

    def load_IBI_data(self, subject = "S1", event = "morning"):
        '''Load the IBI - interbeat interval activity data for the specified subject and
        event into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/IBI.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, names = ["UNIX_time",
                                                     "Interbeat_interval"])
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
                starting_time = df.iloc[0, 0]
                # Removes the first row of the DataFrame, as it contains
                # info about the measurment starting time
                df = df.tail(-1).reset_index(drop=True)
                # Convert the data to numeric values
                df = df.apply(pd.to_numeric, errors='coerce')
                # Determine the number of samples in the data
                number_of_samples = len(df)
                #Add the measurement starting time to the UNIX time
                df["UNIX_time"] = df["UNIX_time"] + starting_time
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['UNIX_time'].max() - df['UNIX_time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['UNIX_time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['UNIX_time'] >= window_start) & (df['UNIX_time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None

    def load_HR_data(self, subject = "S1", event = "morning"):
        '''Load the HR - heart rate activity data for the specified subject and
        event into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/HR.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, names = ["Heart_rate"],
                                dtype=float)
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
                starting_time = df.iloc[0, 0]
                # Load the data sampling rate from the first value in the second
                # row of the csv file
                sampling_rate = df.iloc[1, 0]
                # Removes the first two rows of the DataFrame, as they contain
                # info about the measurment starting time and sampling rate
                df = df.tail(-2).reset_index(drop=True)
                # Determine the number of samples in the data
                number_of_samples = len(df)
                # Generate the time vector based on the measurement starting time
                # and sampling rate
                time_vector = np.arange(starting_time, starting_time +
                            number_of_samples / sampling_rate, 1 / sampling_rate)
                #Add the UNIX time vector to the Pandas DataFrame
                df["UNIX_time"] = time_vector
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['UNIX_time'].max() - df['UNIX_time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['UNIX_time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['UNIX_time'] >= window_start) & (df['UNIX_time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None

    def load_temperature_data(self, subject = "S1", event = "morning"):
        '''Load the TEMP - temperature data for the specified subject and
        event into a Pandas DataFrame.
        Args:
            subject (str): Subject identifier (e.g. S1, S2, ...).
            event (str): Event ("morning" or "evening").
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.'''

            # Initialize the mapping between the type of events recorded in the
            # dataset and the middle path for the respective data
        mapping = {"morning": "1.morning", "evening": "2.evening"}

        try:
            # Try to create the path to the requested data
            data_path = f"data/subject_{subject[1:]}/{mapping[event]}/TEMP.csv"
            # Load the requested data to a Pandas DataFrame if the path exists
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, names = ["Temperature"],
                                dtype=float)
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
                starting_time = df.iloc[0, 0]
                # Load the data sampling rate from the first value in the second
                # row of the csv file
                sampling_rate = df.iloc[1, 0]
                # Removes the first two rows of the DataFrame, as they contain
                # info about the measurment starting time and sampling rate
                df = df.tail(-2).reset_index(drop=True)
                # Determine the number of samples in the data
                number_of_samples = len(df)
                # Generate the time vector based on the measurement starting time
                # and sampling rate
                time_vector = np.arange(starting_time, starting_time +
                            number_of_samples / sampling_rate, 1 / sampling_rate)
                #Add the UNIX time vector to the Pandas DataFrame
                df["UNIX_time"] = time_vector
                # Calculate the total duration of the DataFrame (in seconds)
                total_duration = df['UNIX_time'].max() - df['UNIX_time'].min()
                # Calculate the start time for the 30-minute window (center of data)
                half_duration = total_duration / 2
                center_time = df['UNIX_time'].min() + half_duration
                # Define the 30-minute window
                window_start = center_time - 900  # 900 seconds = 30 minutes
                window_end = center_time + 900
                # Extract data within a 30-minute window located at the center of
                # original DataFrame
                df = df[(df['UNIX_time'] >= window_start) & (df['UNIX_time'] <= window_end)]
                # Return the desired data as a Pandas DataFrame
                return df.reset_index(drop=True)
            else:
                #If the path to the requested data is not valid, return a warning
                print(f"Error: File '{data_path}' not found.")
                return None

        except Exception as e:
            # If the data cannot be accesse (e.g. data file name or path wrong),
            # return a warning
            print(f"Error loading data: {e}")
            return None
