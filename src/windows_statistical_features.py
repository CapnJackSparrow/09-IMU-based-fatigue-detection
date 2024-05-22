'''This script defines a function that computes essential statistical features
(e.g. mean, standard deviation, etc.) from accelerometer data captured by an
Inertial Measurement Unit (IMU). The input is a Pandas DataFrame whose first
three columns represent X, Y, and Z axis accelerations. The function splits the
data into windows and calculates statistical parameters for each one,
consolidating them into a cohesive DataFrame at the end.'''

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew

def windows_statistical_features(data, window_size = 5):
    '''Compute the essential statistical features of accelerometer data over
    windows of time
        Args:
            data (pd.DataFrame): Accelerometer data captured by an
                Inertial Measurement Unit (IMU). The first columns of the
                DataFrame should be labelled "X_acceleration", "Y_acceleration",
                respectively "Z_acceleration"
            windows_size (int): The size of the windows into which the input
                DataFrame will be split; measured in seconds
        Returns:
            pd.DataFrame: DataFrame containing the statistical features.'''

    # Declare the input data sampling rate
    sampling_rate = 32 # [Hz]
    # Transform the window size from seconds into number of data entries
    window_size *= sampling_rate
    # Compute the number of windows into which the input data is devided
    num_windows = len(data) // window_size

    # Initialize an empty DataFrame to store computed features
    columns = [
                # X-axis acceleration mean over the window interval
               'mean_x',
               # Y-axis acceleration mean over the window interval
               'mean_y',
               # Z-axis acceleration mean over the window interval
               'mean_z',
               # X-axis acceleration standard deviation over the window interval
               'std_x',
               # Y-axis acceleration standard deviation over the window interval
               'std_y',
               # Z-axis acceleration standard deviation over the window interval
               'std_z',
               # X-axis acceleration variation over the window interval
                'var_x',
               # Y-axis acceleration variation over the window interval
                'var_y',
               # Z-axis acceleration variation over the window interval
                'var_z',
               # Energy (Euclidean norm) over the window interval
                'energy',
               # Entropy over the window interval
            #    'entropy',
               # X-axis acceleration skew over the window interval
                'skew_x',
               # Y-axis acceleration skew over the window interval
                'skew_y',
               # Z-axis acceleration skew over the window interval
                'skew_z',
               # X-axis acceleration kurtosis over the window interval
                'kurt_x',
               # Y-axis acceleration kurtosis over the window interval
                'kurt_y',
               # Z-axis acceleration kurtosis over the window interval
                'kurt_z',
               # X-axis acceleration range over the window interval
                'range_x',
               # Y-axis acceleration range over the window interval
                'range_y',
               # Z-axis acceleration range over the window interval
                'range_z',
               # X-axis acceleration bottom quarter percentile over the window
               # interval
                'p25_x',
               # X-axis acceleration half percentile over the window interval
                'p50_x',
               # X-axis acceleration top quarter percentile over the window
               # interval
                'p75_x',
               # Y-axis acceleration bottom quarter percentile over the window
               # interval
                'p25_y',
               # Y-axis acceleration half percentile over the window interval
                'p50_y',
               # Y-axis acceleration top quarter percentile over the window
               # interval
                'p75_y',
               # Z-axis acceleration bottom quarter percentile over the window
               # interval
                'p25_z',
               # Z-axis acceleration half percentile over the window interval
                'p50_z',
               # Z-axis acceleration top quarter percentile over the window
               # interval
                'p75_z',
            #    'approx_entropy',
            #    'sample_entropy',
            #    'perm_entropy'
               ]

    # Aceleration axes labels
    axis_labels = ["X", "Y", "Z"]

    # Add cross-correlation columns
    for i in range(len(axis_labels)):
        for j in range(i + 1, len(axis_labels)):
            columns.append(f"cross_corr_{axis_labels[i]}_{axis_labels[j]}")

    # Create an empty Pandas DataFrame to store the features into
    features_df = pd.DataFrame(columns=columns)

    # Loop over the windows
    for i in range(num_windows):
        # Slice the DataFrame according to the window size
        window_data = data.iloc[i * window_size : (i + 1) * window_size]
        # Transform the DataFrame slice into a Numpy array
        if isinstance(window_data, pd.DataFrame):
            window_data = window_data.to_numpy()

        # Compute features within the window
        features = {
            # Compute the X-axis acceleration mean over the window interval
            'mean_x': np.mean(window_data[:, 0]),
            # Compute the Y-axis acceleration mean over the window interval
            'mean_y': np.mean(window_data[:, 1]),
            # Compute the Z-axis acceleration mean over the window interval
            'mean_z': np.mean(window_data[:, 2]),
            # Compute the X-axis acceleration standard deviation over the window
            # interval
            'std_x': np.std(window_data[:, 0]),
            # Compute the Y-axis acceleration standard deviation over the window
            # interval
            'std_y': np.std(window_data[:, 1]),
            # Compute the Z-axis acceleration standard deviation over the window
            # interval
            'std_z': np.std(window_data[:, 2]),
            # Compute the X-axis acceleration variance over the window interval
             'var_x': np.var(window_data[:, 0]),
            # Compute the Y-axis acceleration variance over the window interval
             'var_y': np.var(window_data[:, 1]),
            # Compute the Z-axis acceleration variance over the window interval
             'var_z': np.var(window_data[:, 2]),
            # Compute the energy (euclidean norm) along over the window interval
             'energy': np.linalg.norm(window_data[:, :3], axis = None),
            # Compute entropy over the window interval
            # 'entropy': -np.sum(np.abs(window_data) *
            #                    np.log2(np.abs(window_data))),
            # Compute the X-axis acceleration skew over the window interval
             'skew_x': skew(window_data[:, 0]),
            # Compute the Y-axis acceleration skew over the window interval
             'skew_y': skew(window_data[:, 1]),
            # Compute the Z-axis acceleration skew over the window interval
             'skew_z': skew(window_data[:, 2]),
            # Compute the X-axis acceleration kurtosis over the window interval
             'kurt_x': kurtosis(window_data[:, 0]),
            # Compute the Y-axis acceleration kurtosis over the window interval
             'kurt_y': kurtosis(window_data[:, 1]),
            # Compute the Z-axis acceleration kurtosis over the window interval
             'kurt_z': kurtosis(window_data[:, 2]),
            # Compute the X-axis acceleration range over the window interval
             'range_x': np.max(window_data[:, 0]) - np.min(window_data[:, 0]),
            # Compute the Y-axis acceleration range over the window interval
             'range_y': np.max(window_data[:, 1]) - np.min(window_data[:, 1]),
            # Compute the Z-axis acceleration range over the window interval
             'range_z': np.max(window_data[:, 2]) - np.min(window_data[:, 2]),
            # Compute the X-axis acceleration bottom quarter percentile over the
            # window interval
             'p25_x': np.percentile(window_data[:, 0], 25),
            # Compute the X-axis acceleration half percentile over the window
            # interval
             'p50_x': np.percentile(window_data[:, 0], 50),
            # Compute the  X-axis acceleration top quarter percentile over the
            # window interval
             'p75_x': np.percentile(window_data[:, 0], 75),
            # Compute the Y-axis acceleration bottom quarter percentile over the
            # window interval
             'p25_y': np.percentile(window_data[:, 1], 25),
            # Compute the Y-axis acceleration half percentile over the window
            # interval
             'p50_y': np.percentile(window_data[:, 1], 50),
            # Compute the  Y-axis acceleration top quarter percentile over the
            # window interval
             'p75_y': np.percentile(window_data[:, 1], 75),
            # Compute the Z-axis acceleration bottom quarter percentile over the
            # window interval
             'p25_z': np.percentile(window_data[:, 2], 25),
            # Compute the Z-axis acceleration half percentile over the window
            # interval
             'p50_z': np.percentile(window_data[:, 2], 50),
            # Compute the  Z-axis acceleration top quarter percentile over the
            # window interval
             'p75_z': np.percentile(window_data[:, 2], 75),
            # 'approx_entropy': app_entropy(window_data.flatten()),
            # 'sample_entropy': sample_entropy(window_data.flatten()),
            # 'perm_entropy': perm_entropy(window_data.flatten())
        }

        # Compute cross-correlation
        for i in range(3):
            for j in range(i + 1, 3):
                features[f"cross_corr_{axis_labels[i]}_{axis_labels[j]}"] = np.corrcoef(
                    window_data[:, i],
                    window_data[:, j]
                    )[0, 1]

        # Make a Pandas DataFrame from the features computed for each window
        features = pd.DataFrame(features, index = [0])

        # Append features to the DataFrame
        features_df = pd.concat([features_df, features], ignore_index=True)

    return features_df
