import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

### expected flow is -
###
### generate_df_list()      ##(this assumes the csvs are available all together in data2/ )
### df_list_to_spectrograms_lists(df_list) 
### save_spectrographs(spectrograms_list, t_list, f_list)    ##will make a new spectrograms/ folder)



def df_to_spectrogram(df, fs=32):
  # Extract the acceleration data from each axis.
  x = df['x'].values
  y = df['y'].values
  z = df['z'].values

  # Calculate the spectrogram for each axis.
  f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=160)
  f, t, Syy = signal.spectrogram(y, fs=fs, nperseg=160)
  f, t, Szz = signal.spectrogram(z, fs=fs, nperseg=160)

  # Combine the spectrograms into a single array.
  spectrogram = np.dstack((Sxx, Syy, Szz))

  return spectrogram, t , f
  
def df_list_to_spectrograms_lists(df_list):
    t_list = []
    f_list = []
    spectrograms_list = []

    for df in df_list:
        spectrogram, t ,f = df_to_spectrogram(df, fs=32)
        spectrograms_list.append(spectrogram)
        t_list.append(t)
        f_list.append(f)  

    return spectrograms_list, t_list, f_list


def draw_spectrographs(spectrograms_list, t_list, f_list):
    # Loop through each spectrogram and plot each axis separately.
    for i, spectrogram in enumerate(spectrograms[0:24]):
    
        # Separate the spectrogram for each axis.
        Sxx = spectrogram[:, :, 0]
        Syy = spectrogram[:, :, 1]
        Szz = spectrogram[:, :, 2]

        # Plot the spectrogram for each axis.
        plt.pcolormesh(t_list[i], f_list[i], Sxx, shading='auto', norm='log', cmap="hsv")
        #cmap = plt.colormaps['PiYG']
        plt.ylabel('Frequency [Hz]')
        plt.ylim(0.1, 8)  # Set the limits to avoid zero frequency and noisy upper freqs
        plt.yscale('asinh')
        plt.xlabel('Time [s]')
        plt.title('Spectrogram of Accelerometer Data (X-axis)')
        plt.show()

        plt.pcolormesh(t_list[i], f_list[i], Syy, shading='auto', norm='log', cmap="hsv")
        plt.ylabel('Frequency [Hz]')
        plt.yscale('asinh')
        plt.ylim(0.1, 8)  # Set the limits to avoid zero frequency and noisy upper freqs
        plt.xlabel('Time [s]')
        plt.title('Spectrogram of Accelerometer Data (Y-axis)')
        plt.show()

        plt.pcolormesh(t_list[i], f_list[i], Szz, shading='auto', norm='log', cmap="hsv")
        plt.ylabel('Frequency [Hz]')
        plt.ylim(0.1, 8)  # Set the limits to avoid zero frequency and noisy upper freqs
        plt.xlabel('Time [s]')
        plt.title('Spectrogram of Accelerometer Data (Z-axis)')
        plt.show()


def save_spectrographs(spectrograms_list, t_list, f_list, output_dir='spectrograms'):

    os.makedirs(output_dir, exist_ok=True)

    #names = [f'walk_{i}' for i in range(46)]

    #filename = os.path.join(output_dir, name)

    # Loop through each spectrogram and plot each axis separately.
    for i, spectrogram in enumerate(spectrograms_list):

        filename = os.path.join(output_dir, f'{i}')
    
        # Separate the spectrogram for each axis.
        Sxx = spectrogram[:, :, 0]
        Syy = spectrogram[:, :, 1]
        Szz = spectrogram[:, :, 2]

        # Plot the spectrogram for each axis.
        plt.pcolormesh(t_list[i], f_list[i], Sxx, shading='auto', norm='log', cmap="hsv")
        #cmap = plt.colormaps['PiYG']
        #plt.ylabel('Frequency [Hz]')
        plt.ylim(0.1, 8)  # Set the limits to avoid zero frequency and noisy upper freqs
        plt.yscale('asinh')
        #plt.xlabel('Time [s]')
        #plt.title('Spectrogram of Accelerometer Data (X-axis)')
        plt.savefig(f'{filename}_x.png')
        plt.close()

        plt.pcolormesh(t_list[i], f_list[i], Syy, shading='auto', norm='log', cmap="hsv")
        #plt.ylabel('Frequency [Hz]')
        plt.yscale('asinh')
        plt.ylim(0.1, 8)  # Set the limits to avoid zero frequency and noisy upper freqs
        #plt.xlabel('Time [s]')
        #plt.title('Spectrogram of Accelerometer Data (Y-axis)')
        plt.savefig(f'{filename}_y.png')
        plt.close()

        plt.pcolormesh(t_list[i], f_list[i], Szz, shading='auto', norm='log', cmap="hsv")
        #plt.ylabel('Frequency [Hz]')
        plt.yscale('asinh')
        plt.ylim(0.1, 8)  # Set the limits to avoid zero frequency and noisy upper freqs
        #plt.xlabel('Time [s]')
        #plt.title('Spectrogram of Accelerometer Data (Z-axis)')
        plt.savefig(f'{filename}_z.png')
        plt.close()





def normalization(subject):
    num_transformer=make_pipeline(StandardScaler())
    num_transformer.fit(subject)
    return pd.DataFrame(num_transformer.transform(subject), columns=subject.columns)


def generate_df_list():
    df_list = []

    for walk in range(0,46):
        df = pd.read_csv(f"data2/{walk}.csv", names = ["x",
        "y", "z"], dtype=float)
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
        df = df.drop(columns='UNIX_time')
        df = df.reset_index(drop=True)
        df = normalization(df)

                #divides each df into 6ths
        for chunk in range(6):
                #adds first 6th to the output list
                df_c = df[:9600]
                df_list.append(df_c)
                #discards first 6th
                df = df.iloc[9600:]
    return df_list