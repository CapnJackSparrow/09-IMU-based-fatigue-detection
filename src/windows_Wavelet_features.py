'''This script defines a function that computes Wavelet transform features
(e.g. Wavelet coefficients, etc.) from accelerometer data captured by an
Inertial Measurement Unit (IMU). The input is a Pandas DataFrame whose first
three columns represent X, Y, and Z axis accelerations. The function splits the
data into windows and calculates Wavelet transform characteristics for each one,
consolidating them into two cohesive DataFrame at the end.'''

import numpy as np
import pandas as pd
import pywt

def windows_Wavelet_features(data, window_size = 5):
    '''Compute the Wavelet transform features of accelerometer data over windows
    of time
        Args:
            data (pd.DataFrame): Accelerometer data captured by an
                Inertial Measurement Unit (IMU). The first columns of the
                DataFrame should be labelled "X_acceleration", "Y_acceleration",
                respectively "Z_acceleration"
            windows_size (int): The size of the windows into which the input
                DataFrame will be split; measured in seconds
        Returns: pd.DataFrame: DataFrame containing the computed Wavelet features.'''

    # Declare the input data sampling rate
    sampling_rate = 32 # [Hz]
    # Transform the window size from seconds into number of data entries
    window_size *= sampling_rate
    # Compute the number of windows into which the input data is devided
    num_windows = len(data) // window_size
    # Define the frequency band (0.5 Hz to 3 Hz) for which the energy will be
    # computed over each window interval
    low_freq_band = (0.5, 3)  # Frequency range in Hz
    # Compute the frequency resolution
    freq_resolution = sampling_rate/window_size
    # Retrieve the low frequency band indexes
    low_freq_indices = np.arange(int(low_freq_band[0] / freq_resolution),
                                 int(low_freq_band[1] / freq_resolution) + 1)

    # Initialize an empty DataFrame to store computed features
    columns = [
               # X-axis acceleration low frequency band energy over the window
               # interval
               'Energy_LowFreq_x',
               # Y-axis acceleration low frequency band energy over the window
               # interval
               'Energy_LowFreq_y',
               # Z-axis acceleration low frequency band energy over the window
               # interval
               'Energy_LowFreq_z',
            #    # X-axis acceleration dominant frequency over the window interval
            #    'DominantFreq_x',
            #    # Y-axis acceleration dominant frequency over the window interval
            #    'DominantFreq_y',
            #    # Z-axis acceleration dominant frequency over the window interval
            #    'DominantFreq_z',
               # X-axis acceleration energy over the window interval
               'Entropy_x',
               # Y-axis acceleration energy over the window interval
               'Entropy_y',
               # Z-axis acceleration energy over the window interval
               'Entropy_z',
    ]

    # Create an empty Pandas DataFrame to store the features into
    features_df = pd.DataFrame(columns=columns)

    # Create an empty list to store the X-axis acceleration Wavelet transform
    # coefficients for each window interval
    coefficients_x = []
    # Create an empty list to store the Y-axis acceleration Wavelet transform
    # coefficients for each window interval
    coefficients_y = []
    # Create an empty list to store the Z-axis acceleration Wavelet transform
    # coefficients for each window interval
    coefficients_z = []

    # Loop over the windows
    for i in range(num_windows):
        # Slice the DataFrame according to the window size
        window_data = data.iloc[i * window_size : (i + 1) * window_size]
        # Transform the DataFrame slice into a Numpy array
        if isinstance(window_data, pd.DataFrame):
            window_data = window_data.to_numpy()

        # Compute the X-axis acceleration Wavelet transform (using Daubechies
        # 4 wavelet) over the window interval
        coeffs_x, _ = pywt.dwt(window_data[:, 0], 'db4')
        # Append the computed X-axis acceleration Wavelet transform coefficients
        # for the current window interval to the list of coefficients for the
        # whole time domain
        coefficients_x.append(coeffs_x)
        # Compute the Y-axis acceleration Wavelet transform (using Daubechies
        # 4 wavelet) over the window interval
        coeffs_y, _ = pywt.dwt(window_data[:, 1], 'db4')
        # Append the computed Y-axis acceleration Wavelet transform coefficients
        # for the current window interval to the list of coefficients for the
        # whole time domain
        coefficients_y.append(coeffs_y)
        # Compute the Z-axis acceleration Wavelet transform (using Daubechies
        # 4 wavelet) over the window interval
        coeffs_z, _ = pywt.dwt(window_data[:, 2], 'db4')
        # Append the computed Z-axis acceleration Wavelet transform coefficients
        # for the current window interval to the list of coefficients for the
        # whole time domain
        coefficients_z.append(coeffs_z)

        # Compute the X-axis acceleration energy in the 0.5 - 3Hz frequency
        # band over the window interval
        energy_low_freq_x = np.sum(coeffs_x[low_freq_indices] ** 2)
        # Compute the Y-axis acceleration energy in the 0.5 - 3Hz frequency
        # band over the window interval
        energy_low_freq_y = np.sum(coeffs_y[low_freq_indices] ** 2)
        # Compute the Z-axis acceleration energy in the 0.5 - 3Hz frequency
        # band over the window interval
        energy_low_freq_z = np.sum(coeffs_z[low_freq_indices] ** 2)

            # # Retrieve X-axis acceleration dominant frequencies over the window
            # # interval
            # dominant_freq_x = sampling_rate / (2 ** np.argmax(np.abs(coeffs_x)))
            # # Retrieve Y-axis acceleration dominant frequencies over the window
            # # interval
            # dominant_freq_y = sampling_rate / (2 ** np.argmax(np.abs(coeffs_y)))
            # # Retrieve Z-axis acceleration dominant frequencies over the window
            # # interval
            # dominant_freq_z = sampling_rate / (2 ** np.argmax(np.abs(coeffs_z)))

        # Declare a small constant used to transform null acceleration values
        # into non-null ones so that the logarithm can be computed
        eps = 1e-10

        # Compute the entropy of the X-axis acceleration Wavelet coefficients
        # over the window interval
        entropy_x = -np.sum(coeffs_x ** 2 * np.log(coeffs_x ** 2 + eps))
        # Compute the entropy of the Y-axis acceleration Wavelet coefficients
        # over the window interval
        entropy_y = -np.sum(coeffs_y ** 2 * np.log(coeffs_y ** 2 + eps))
        # Compute the entropy of the X-axis acceleration Wavelet coefficients
        # over the window interval
        entropy_z = -np.sum(coeffs_z ** 2 * np.log(coeffs_z ** 2 + eps))

        # Retrieve features within the window interval
        features = {
            # Retrieve the X-axis acceleration low frequency band energy over
            # the window interval
            'Energy_LowFreq_x': energy_low_freq_x,
            # Retrieve the Y-axis acceleration low frequency band energy over
            # the window interval
            'Energy_LowFreq_y': energy_low_freq_y,
            # Retrieve the Z-axis acceleration low frequency band energy over
            # the window interval
            'Energy_LowFreq_z': energy_low_freq_z,
            # # Retrieve the X-axis acceleration dominant frequency over the
            # # window interval
            # 'DominantFreq_x': dominant_freq_x,
            # # Retrieve the Y-axis acceleration dominant frequency over the
            # # window interval
            # 'DominantFreq_y': dominant_freq_y,
            # # Retrieve the Z-axis acceleration dominant frequency over the
            # # window interval
            # 'DominantFreq_z': dominant_freq_z,
            # Retrieve the X-axis acceleration energy over the window interval
            'Entropy_x': entropy_x,
            # Retrieve the X-axis acceleration energy over the window interval
            'Entropy_y': entropy_y,
            # Retrieve the X-axis acceleration energy over the window interval
            'Entropy_z': entropy_z
        }

        # Make a Pandas DataFrame from the features computed for each window
        features = pd.DataFrame(features, index = [0])

        # Append features to the DataFrame
        features_df = pd.concat([features_df, features], ignore_index=True)

    # Create a DataFrame to store the Wavelet transform coefficients for each
    # window interval
    coefficients_df = pd.DataFrame({
        'X_coeffs': coefficients_x,
        'Y_coeffs': coefficients_y,
        'Z_coeffs': coefficients_z
    })

    return features_df, coefficients_df
