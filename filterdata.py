import numpy as np
import pandas as pd
from scipy.special import erfc
from scipy import signal, fftpack
from scipy.fftpack import rfft, irfft, fftfreq
import ruptures as rpt

##################
### Parameters ###
##################

### Digital filter forward and backward to a signal
cutoff = 0.1
fs = 30                     # sampling frequency of the digital system
order = 5                   # order of the filter
btype = 'highpass'          # type of filter

### Savitzky-Golay filter
wl = 99                     # window-length
polyorder = 5               # order of the polynomial used to fit the samples

### FFT High freq
timestep = 70               # time step

### FFT bandpass
high = 0.0004               # upper limit of frequencies
low = 0.000001              # lower limit of frequencies

####################################
### Calculating rolling averages ###
####################################

def RollingAverage(array, 
                   window_size = 10):
    """
    Calculate the rolling/moving average of a numpy array, given the window-size.
    
    Parameters
    ----------
    array : numpy array
    window_size : int, default=10, window-size for the rolling average

    Returns
    ----------
    rollave : numpy array, rolling average of an array
    """    
    
    # Convert array of integers to pandas series
    numbers_series = pd.Series(array)

    # Get the window of series of observations of specified window size
    windows = numbers_series.rolling(window_size)

    # Create a series of moving averages of each window
    moving_averages = windows.mean()

    # Convert pandas series back to list
    moving_averages_list = moving_averages.tolist()

    # Remove null entries from the list
    rollave = moving_averages_list[window_size - 1:]
    
    # Change the series back to numpy array
    rollave = np.array(rollave)
    
    return rollave

#################
### Filtering ###
#################

# Digital filter forward and backward to a signal
def Filter_filtfilt(mag):#, cutoff, fs, order, btype):
    """
    Applies digital filter forward and backward to a signal.
    
    Parameters
    ----------
    mag : numpy array, magnetic field strength
    cutoff : int, cutoff frequency used to calculate normal cutoff
    fs : float, sampling frequency of the digital system
    order : int, order of the filter
    btype : string, type of filter, can be ‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’

    Returns
    ----------
    filtered signal : numpy array    
    """
  
    nyq = 0.5 * fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype, analog=False)
    
    filtfilt_mag = signal.filtfilt(b, a, mag, padlen=10)
    
    return filtfilt_mag
    
# Savitzky-Golay filter 
def Filter_savgol(mag):#, window_length, polyorder):
    """
    Applies Savitzky-Golay filter to a signal.
    
    Parameters
    ----------
    mag : numpy array, magnetic field strength
    window_length : int, length of the filter window
    polyorder : int, order of the polynomial used to fit the samples
    
    Returns
    ----------
    filtered signal : numpy array    
    """
    
    return signal.savgol_filter(mag, window_length, polyorder)

def Filter_ffthighfreq(mag):#, timestep):
    """
    Applies FFT high frequency filter to a signal.
    
    Parameters
    ----------
    mag : numpy array, magnetic field strength
    timestep : int, sample spacing (inverse of the sampling rate)
    
    Returns
    ----------
    filtered signal : numpy array    
    """
      
    # FFT of the signal
    mag_fft = fftpack.fft(mag)

    # Calculate power spectrum
    power = np.abs(mag_fft)**2

    # Corresponding frequencies
    sample_freq = fftpack.fftfreq(mag.size, d = timestep)

    # Find the peak frequency: concentrate on the positive values
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    # We now remove all the high frequencies and transform back from frequencies to signal.
    high_freq_fft = mag_fft.copy()
    high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    ffthighfreq_mag = fftpack.ifft(high_freq_fft)
    
    return ffthighfreq_mag, peak_freq

# FFT filter
def Filter_fftbandpass(mag):#, timestep, low, high):
    """
    Applies FFT bandpass filter to a signal.
    
    Parameters
    ----------
    mag : numpy array, magnetic field strength
    timestep : int, sample spacing (inverse of the sampling rate)
    low : int, lower limit of frequency
    high : int, upper limit of frequency
    
    Returns
    ----------
    filtered signal : numpy array    
    """
        
    W = fftfreq(mag.size, d = timestep)
    f_signal = rfft(mag)

    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    cut_f_signal[(W>high)] = 0           # High frequencies
    cut_f_signal[(W<low)] = 0            # Low frequencies

    fftbandpass_mag = irfft(cut_f_signal)
    
    return fftbandpass_mag
    
################
### Outliers ###
################

def Outliers(signal):
    """
    Find outliers. Points need to be evaluated using Chauvenet's criterion.
    
    Parameters
    ----------
    signal : numpy array

    Returns
    ----------
    signal without outliers : numpy array    
    
    """
    
    # Find outliers
    meansignal = np.mean(signal)                      # Mean of signal
    stdsignal = np.std(signal)                        # Standard deviation of signal
    chauvenet_crit = 1.0/(2*len(signal))              # Chauvenet's criterion
    residuals = abs(signal - meansignal) / stdsignal  # Distance of a value to mean in stdv's
    prob = erfc(residuals)                            # Area normal distribution    
    outliers = prob < chauvenet_crit                  # List of boolean. True means there is an outlier
    
    # Replace outlier with mean value of the signal
    for i in range(len(signal)):
        if outliers[i]:
            signal[i] = meansignal 
            
    return signal

##########################
### Standardize signal ###
##########################

def Standardize(signal):
    """
    Standardize (Z-score normalization) time-series signal.
    
    Parameters
    ----------
    signal : numpy array

    Returns
    ----------
    standardized signal : numpy array    
    
    """
    
    meansignal = np.mean(signal)                                   # Calculate the mean of signal
    stdsignal = np.std(signal)                                     # Calculate the standard deviation of signal
    standardized_signal = abs(signal - meansignal) / stdsignal     # Standardize signal (Z-score normalization)
    
    return standardized_signal

################
### DC Shift ###
################

def DC_Shift(mag, 
             bkps = 10):
    """
    Calculate change points (breaking points) in the signal, break it up to segments, 
    DC shift each segment to zero and return new array.
    
    Parameters
    ----------
    mag : numpy array, magnetic field strength measurements 
    bkps : int, default=10, max number of predicted breaking points

    Returns
    ----------
    DC shifted magnetic field strength data : numpy array    
    
    """
    
    if bkps != None:           # If there are change points in the signal
        # Change point detection
        model = "normal"  # "l2", "l1", "rbf", "linear", "normal", "ar"      # Segment model
        algo = rpt.Window(width = 40, model = model).fit(mag)      # Window sliding method to find breaking points
        my_bkps = algo.predict(n_bkps = bkps - 1)                  # Predict the location of breaking points in the signal
        #print("Breaking points = {}".format(my_bkps))   

        # Define segments using the breaking points
        start = 0                                                  # Starting point of the data set
        segmented_list = []                                        # Empty list for the segmented lists
        for i in range(len(mag)):                                  # Loop through the magnetic field strength measurements
            for p in my_bkps:                                      # But also loop through the breaking point list
                if i == p:                                         # If the index equals the breaking point
                    segment = mag[start:p]                         # Define a new segment from start point to the breaking point
                    segmented_list.append(segment)                 # Append this new segment to the segmented list
                    start = p                                      # Redefine the starting point
        last_segment = mag[start:]                                 # Define last segment
        segmented_list.append(last_segment)                        # Append last segment to the segmented list
        segmented_list = np.array(segmented_list,dtype=object)     # Convert list to numpy array

        maglength = len(mag)                                       # Length of data

        # DC Shift each segment to zero
        shifted_mag = []                                           # Empty list for the shifted field strength
        for s in segmented_list:                                   # Loop through each segment
            mean = np.mean(s)                                      # Mean of segment
            shifted_s = s - mean                                   # DC shift
            shifted_mag.append(shifted_s)                          # Append shifted signal to the shifted field strength list
        shifted_mag = np.array(shifted_mag,dtype=object)           # Convert list to numpy array
        shifted_mag = np.concatenate(shifted_mag,axis=0)           # Concatenate the whole sequence of arrays
    
    else:                   # If there are no change points in the signal                
        shifted_mag = mag - np.mean(mag)
    
    return shifted_mag