import pandas as pd
import numpy as np

def Load_datas_pred(csv_file_path):
    """ Read the CSV file and skip the first row"""
    df = pd.read_csv(f"{csv_file_path}/Accelerometer.csv", header=None)
    # Remove the first row (index 0)
    df = df.iloc[1:]
    # Rename columns (adjust column names as needed)
    column_mapping = {
        0: 'UNIX_time',
        1: 'Seconds_elapsed',
        2: 'Z_acceleration',
        3: 'Y_acceleration',
        4: 'X_acceleration'
    }
    df.rename(columns=column_mapping, inplace=True)
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    # Convert the UNIX time to a float with separator after the 10th digit
    df['UNIX_time'] = df['UNIX_time'].astype(str).str[:10] + '.' \
                            + df['UNIX_time'].astype(str).str[10:]
    # Transform the unix time into date and time
    df['UNIX_time'] = pd.to_datetime(df['UNIX_time'], unit='s')
    # Set the unix time as dataframe index
    df.set_index('UNIX_time', inplace=True)
    # Resample the dataframe to 32 Hz
    df = df.resample('31.25ms').mean()
    return df



def Datas_truncature(data_pred,time_laps=90):
    """ this function takes the middle datas during a time_laps predetermined
    i.e => For 90 sec, it will find the middle of the datasets and takes only 45 sec of datas before and after this"""
    # Changing the order of columns => X, Y,Z => same names as the one from Data Loader
    datas_pred_ordered = data_pred[["X_acceleration","Y_acceleration","Z_acceleration"]]
    # Find the middle and the number of lines in the datasets
    len_datas = datas_pred_ordered.shape[0]
    # time_laps = 90  : for test
    nb_event_time_laps = 32*time_laps
    middle_target_time = np.round(len_datas/2)
    begin_index_truncated = int(middle_target_time - (nb_event_time_laps/2))
    end_index_truncated = int(middle_target_time + (nb_event_time_laps/2))
    # verification
    end_index_truncated - begin_index_truncated == nb_event_time_laps
    # truncate the datas in order to have 90 sec
    df_pred_truncated = datas_pred_ordered.iloc[begin_index_truncated:end_index_truncated,:]
    return df_pred_truncated


## call the function :
# csv_path = "../../../marieyalap/Project/Traian_Project/Datas/Datas_for_pred/test/test/Traian_walk_fresh"
# df_pred_call = Datas_truncature(Load_datas_pred(csv_path),90)
