'''This script defines a function that computes FFT features
(e.g. mean, standard deviation, etc.) from accelerometer data captured by an
Inertial Measurement Unit (IMU). The input is a Pandas DataFrame whose first
three columns represent X, Y, and Z axis accelerations. The function splits the
data into windows and calculates statistical parameters for each one,
consolidating them into a cohesive DataFrame at the end.'''

import numpy as np
import pandas as pd
from scipy.stats import entropy

def windows_FFT_features(data, window_size = 5):
    '''Compute the FFT features of accelerometer data over windows of time
        Args:
            data (pd.DataFrame): Accelerometer data captured by an
                Inertial Measurement Unit (IMU). The first columns of the
                DataFrame should be labelled "X_acceleration", "Y_acceleration",
                respectively "Z_acceleration"
            windows_size (int): The size of the windows into which the input
                DataFrame will be split; measured in seconds
        Returns: pd.DataFrame, pd.DataFrame: Two DataFrames containing the
        computed features:
            - First DataFrame: Contains FFT numerical features (e.g., dominant
            frequencies, entropy, etc.); one value in each cell.
            - Second DataFrame: Contains FFT magnitude spectrograms; a list of
            spectrum magnitudes in each cell.'''

    # Declare the input data sampling rate
    sampling_rate = 32 # [Hz]
    # Transform the window size from seconds into number of data entries
    window_size *= sampling_rate
    # Compute the number of windows into which the input data is devided
    num_windows = len(data) // window_size

    # Initialize an empty DataFrame to store computed features
    columns = [
               # X-axis acceleration dominant frequency
               'dominant_freq_x',
               # Y-axis acceleration dominant frequency
               'dominant_freq_y',
               # Z-axis acceleration dominant frequency
               'dominant_freq_z',
               # X-axis acceleration signed entropy
               'entropy_x',
               # Y-axis acceleration signed entropy
               'entropy_y',
               # Z-axis acceleration signed entropy
               'entropy_z',
               # X-axis acceleration low frequency band energy
               'low_freq_band_energy_x',
               # Y-axis acceleration low frequency band energy
               'low_freq_band_energy_y',
               # Z-axis acceleration low frequency band energy
               'low_freq_band_energy_z'
    ]

    # Create an empty Pandas DataFrame to store the features into
    features_df = pd.DataFrame(columns=columns)

    # Create an empty Pandas DataFrame to store the FFT magnitudes into
    magnitudes_df = pd.DataFrame(columns=[
        "FFT_mag_X",
        "FFT_mag_Y",
        "FFT_mag_Z"
        ])

    # Loop over the windows
    for i in range(num_windows):
        # Slice the DataFrame according to the window size
        window_data = data.iloc[i * window_size : (i + 1) * window_size]
        # Transform the DataFrame slice into a Numpy array
        if isinstance(window_data, pd.DataFrame):
            window_data = window_data.to_numpy()

        # Compute the X-axis acceleration FFT over the window interval
        fft_x = np.fft.fft(window_data[:, 0])
        # Compute the Y-axis acceleration FFT over the window interval
        fft_y = np.fft.fft(window_data[:, 1])
        # Compute the Z-axis acceleration FFT over the window interval
        fft_z = np.fft.fft(window_data[:, 2])

        # Compute the FFT frequencies axis
        freq_axis = np.fft.fftfreq(len(window_data), d=1.0 / sampling_rate)

        # Retrieve the positive frequencies of the frequencies axis and exclude
        # the first two bins (the DC component and the first positive
        # frequency), as they do not have much significance (given the way FFT
        # is computed, the maximum amplitude is always achieved for the lowest
        # frequency)

        positive_freq_indices = np.where(freq_axis >= 0)[0][2:]

        # Retrieve the positive frequencies axis

        pos_freq_axis = freq_axis[positive_freq_indices]

        # Compute the X-axis acceleration magnitude spectrum over the window
        # interval
        magnitude_spectrum_x = np.abs(fft_x)
        # Compute the Y-axis acceleration magnitude spectrum over the window
        # interval
        magnitude_spectrum_y = np.abs(fft_y)
        # Compute the Z-axis acceleration magnitude spectrum over the window
        # interval
        magnitude_spectrum_z = np.abs(fft_z)

        # Retrieve magnitudes for positive frequencies within the window
        # interval; the first 2 magnitudes in the spectrum are dropped, as they
        # do not have much significance (given the way FFT is computed, the
        # maximum amplitude is always achieved for the lowest frequency)
        magnitudes = {
            "FFT_mag_X": [magnitude_spectrum_x[positive_freq_indices].tolist()],
            "FFT_mag_Y": [magnitude_spectrum_y[positive_freq_indices].tolist()],
            "FFT_mag_Z": [magnitude_spectrum_z[positive_freq_indices].tolist()]
        }

        # Retrieve the acceleration dominant frequencies for each axis over the
        # window interval from the frequencies axis using the index of the
        # maximum values in the magnitude spectrum after droping the first 2
        # values, as they do not have much significance (given the way FFT is
        # computed, the maximum amplitude is always achieved for the lowest
        # frequency)

        # Retrieve the the X-axis acceleration dominant frequency over the
        # window interval
        dominant_freq_x = pos_freq_axis[np.argmax(magnitude_spectrum_x[
                                        positive_freq_indices].tolist())]
        # Retrieve the the Z-axis acceleration dominant frequency over the
        # window interval
        dominant_freq_y = pos_freq_axis[np.argmax(magnitude_spectrum_y[
                                        positive_freq_indices].tolist())]
        # Retrieve the the Z-axis acceleration dominant frequency over the
        # window interval
        dominant_freq_z = pos_freq_axis[np.argmax(magnitude_spectrum_z[
                                        positive_freq_indices].tolist())]

        # Declare a small constant used to transform null acceleration values
        # into non-null ones so that the logarithm can be computed
        eps = 1e-10

        # Compute the X-axis acceleration signed entropy over the window
        # interval
        entropy_x = np.sum(np.abs(window_data[:, 0]) * np.log(
                                            np.abs(window_data[:, 0] + eps)))
        # Compute the Y-axis acceleration signed entropy over the window
        # interval
        entropy_y = np.sum(np.abs(window_data[:, 1]) * np.log(
                                            np.abs(window_data[:, 1] + eps)))
        # Compute the Z-axis acceleration signed entropy over the window
        # interval
        entropy_z = np.sum(np.abs(window_data[:, 2]) * np.log(
                                            np.abs(window_data[:, 2] + eps)))

        # Compute the X-axis acceleration low-frequency band energy over the
        # window interval
        low_freq_band_energy_x = np.sum(magnitude_spectrum_x[1:10])
        # Compute the Y-axis acceleration low-frequency band energy over the
        # window interval
        low_freq_band_energy_y= np.sum(magnitude_spectrum_y[1:10])
        # Compute the z-axis acceleration low-frequency band energy over the
        # window interval
        low_freq_band_energy_z = np.sum(magnitude_spectrum_z[1:10])


        # Retrieve features within the window interval
        features = {
            # Retrieve the X-axis acceleration dominant frequnecy over the
            # window interval
            'dominant_freq_x': dominant_freq_x,
            # Retrieve the Y-axis acceleration dominant frequnecy over the
            # window interval
            'dominant_freq_y': dominant_freq_y,
            # Retrieve the Z-axis acceleration dominant frequnecy over the
            # window interval
            'dominant_freq_z': dominant_freq_z,
            # Retrieve X-axis acceleration entropy over the window interval
            'entropy_x': entropy_x,
            # Retrieve Y-axis acceleration entropy over the window interval
            'entropy_y': entropy_y,
            # Retrieve Z-axis acceleration entropy over the window interval
            'entropy_z': entropy_z,
            # Retrieve X-axis acceleration low frequency band energy over the
            # window interval
            'low_freq_band_energy_x': low_freq_band_energy_x,
            # Retrieve Y-axis acceleration low frequency band energy over the
            # window interval
            'low_freq_band_energy_y': low_freq_band_energy_y,
            # Retrieve Z-axis acceleration low frequency band energy over the
            # window interval
            'low_freq_band_energy_z': low_freq_band_energy_z
        }

        # Make a Pandas DataFrame from the features computed for each window
        features = pd.DataFrame(features, index = [0])

        # Make a Pandas DataFrame from the magnitudes computed for each window
        magnitudes = pd.DataFrame(magnitudes)

        # Append features to the DataFrame
        features_df = pd.concat([features_df, features], ignore_index=True)

        # Append magnitudes to the DataFrame
        magnitudes_df = pd.concat([magnitudes_df, magnitudes], ignore_index=True)

    return features_df, magnitudes_df
