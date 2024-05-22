from src.DataLoader import DataLoader
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
### 4.1.5 - Models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline


def Tensor_data():
    '''Return a list (size = 46) of dataframes (shape = (57600, 3)) where :
    each list represents a subject and each dataframe has the columns X_acceleration  Y_acceleration and Z_acceleration '''
    dfs = []
    # Subject_list = [f'S{i}' for i in range(1,24)]
    for subject in DataLoader().subjects:
        for event in DataLoader().events:
            df = DataLoader().load_ACC_data(subject,event)

        # df = DataLoader().load_ACC_data(subject,"morning")
                # Load the time when the data measurement started from the first
                # value in the first row of the csv file
            starting_time = 1675144125
                    # Load the data sampling rate from the first value in the second
                    # row of the csv file
            sampling_rate = 32
                    # Removes the first two rows of the DataFrame, as they contain
                    # info about the measurment starting time and sampling rate
            # df = df.tail(-2).reset_index(drop=True)
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
            df = df.drop(columns='UNIX_time')
            df = df.reset_index(drop=True)
            dfs.append(df)
    return dfs



def normalize_data(data_subject):
    '''Return a dataframe with normalized datas for a certain subject '''
    num_transformer=make_pipeline(StandardScaler())
    num_transformer.fit(data_subject)
    return pd.DataFrame(num_transformer.transform(data_subject), columns=data_subject.columns)

def normalize_list_data(data_list):
    '''Return a list of all dataframes with normalized datas for each subject '''
    df_copy = data_list.copy()
    for i in range(0,len(df_copy)):
        df_copy[i] = normalize_data(df_copy[i])
    return df_copy


def label_creator():
    '''Return Y_label where the first column is a number from the survey which explains how much tired is the subject
     and the second column is a number between 0 and 1 (0:Not tired, 1:tired) with a threshold = 11'''
    sleepy_list = [11, 18, 0, 12, 10, 11, 5, 11, 2, 5, 9, 9, 17, 4, 2, 12, 12, 19, 13, 7, 11, 3, 5, 14, 14, 14, 12, 13, 11, 1, 11, 13, 11, 8, 19, 21, 22, 16, 15, 12, 26, 4, 19, 21, 18, 20]
    sleepy_df = pd.DataFrame(sleepy_list)
    sleepy_df["fatigue"] = sleepy_df[0]
    sleepy_df = sleepy_df.drop(columns=0)
    sleepy_df["is_tired"] = 0
    sleepy_df['is_tired'] = (sleepy_df['fatigue'] > 11).astype(int)
    return sleepy_df



def chunk_datas(data_list_normalized, chunk_size):
    # chunk_size = 9600 which represents 5mins
    chunk_list = []
    df_copy = data_list_normalized.copy()
    for i in range(0,len(df_copy)):
        #divides each df into 6ths
        for chunk in range(6):
                #adds first 6th to the output list
                df_c = df_copy[i][:chunk_size]
                chunk_list.append(df_c)
                #discards first 6th
                df_copy[i] = df_copy[i].iloc[chunk_size:]
    return chunk_list



def chunk_label(Y, split_size):
    # split_size = 6
    Y_split = pd.DataFrame(np.repeat(Y.values, split_size, axis=0))
    Y_split.columns = Y.columns
    return Y_split
