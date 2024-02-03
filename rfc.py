import csv
from scipy.signal import butter, filtfilt
from scipy import fft
import numpy as np
import os
from wettbewerb import load_references, get_3montages, EEGDataset
import mne
from scipy import signal as sig
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import ruptures as rpt
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import json
from sklearn.ensemble import RandomForestClassifier

"""
the whole training process is in the rfc_train.ipynb, the first four functions there are same with here
If you start with this file, you can just skip to extract_eeg_data() in rfc_train.ipynb 
Thanks
"""
#butter band pass filter
def butter_bandpass_filter(_signal, lowcut, highcut, freq, order=4):
    """
    build a butter bandpass filter
    :param _signal: eeg signal
    :param lowcut: lower bound of the filter
    :param highcut: upper bound of filter
    :freq: sampling frequency
    :order： order of the filter
    :return: first index of three continous 1 windows
    """
    nyquist = 0.5 * freq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_sig = filtfilt(b, a, _signal)
    return filtered_sig

#compute band power in a certrain range
def compute_band_power(sig, freq, freq_low, freq_high):
    """
    compute the band power in different frequency range
    :param sig: eeg signal
    :param freq: sampling frquency of the eeg signal
    :param freq_low: lower bound of the required wave frequency range
    :param freq_high: upper bound of the required wave frequency range
    :return: the power energy in range from freq_low to freq_high
    """
    spectrum = fft.fft(sig)
    frequencies = np.fft.fftfreq(len(sig), d=1/freq)
    freq_range = (frequencies >= freq_low) & (frequencies <= freq_high)
    power = np.sum(np.abs(spectrum[freq_range])**2) / len(freq_range)
    return power
    
# processing a single eeg signal  
def pre_processing(_channels, _data, fr):
    """
    precessing an EEG signal, extract features, including mean amplitude, max amplitude, standard deviation, and energy of delta, theta, alpha, beta wave
    :param _channels: channels of the eeg data
    :param _data: eeg data
    :param fr: sampling frequency of the eeg signal
    :return: extracted features in a form of numpy array
    """
    
    _montage, _montage_data, _is_missing = get_3montages(_channels, _data)
    # initialize the first three features
    amplitude_means = np.zeros(len(_montage))
    mean_amplitude = []
    amplitude_max = np.zeros(len(_montage))
    max_amplitude = []
    amplitude_std = np.zeros(len(_montage))
    std_amplitude = []
    
    band_power = np.zeros(4)

    for j, signal_name in enumerate(_montage):
        signal = _montage_data[j]
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fr, freqs=np.array([50., 100.]), #apply notch filter
                                                n_jobs=2, verbose=False)
        signal_filter = butter_bandpass_filter(_signal=signal, lowcut=0.5, #apply butter bandpass filter
                                                   highcut=70.0, freq=fr)

        # Calculate FFT 
        spectrum = fft.fft(signal_filter)   #fast fourier
        amplitude_means[j] = np.mean(np.abs(spectrum)) # three montage, store the mean amplitude of each monge
        amplitude_max[j] = np.max(np.abs(spectrum))# three montage, store the max amplitude of each monge
        amplitude_std[j] = np.std(np.abs(spectrum))# three montage, store the standard deviation of each monge

        # Compute band power
        band_power[0] += compute_band_power(signal_filter, fr, 0.5, 4) # energy of delta
        band_power[1] += compute_band_power(signal_filter, fr, 4, 8)# energy of theta
        band_power[2] += compute_band_power(signal_filter, fr, 8, 13)# energy of alpha
        band_power[3] += compute_band_power(signal_filter, fr, 13, 30)# energy of beta
        
    mean_amplitude.append(np.mean(amplitude_means))  #mean value of mean amplitude of three montages
    mean_amplitude = np.array(mean_amplitude)
    max_amplitude.append(np.mean(amplitude_max))#mean value  of max amplitude of three montages
    max_amplitude = np.array(max_amplitude)
    std_amplitude.append(np.mean(amplitude_std))#mean value  of standard deviation of three montages
    std_amplitude = np.array(std_amplitude)
    
    band_power /= len(_montage) # mean enegy of three montages
    features = np.concatenate((mean_amplitude, max_amplitude, std_amplitude, band_power), axis=0) # concatenat all 7 features to a np array
    #features_reshaped = features.reshape(1, -1)
    #feature_scaled = scaler.transform(features_reshaped)
    return features

def pad_eeg_signals(eeg_signals, target_length, fs=256):
    """
    padding EEG signal of different length to a common length
    :param eeg_signals: A EEG signal list with many signals, its length may differ
    :param target_length: target lenth of the signal
    :param fs: sampling frequency of the signal
    :return: List of padded signal
    """
    padded_signals = []
    target_samples = target_length * fs  # transform the target lenth to number of samples

    for signal in eeg_signals:
        if len(signal) < target_samples:
            # if signal length shorter than target length, padding 0
            padded_signal = np.pad(signal, (0, target_samples - len(signal)), 'constant')
        else:
            # if. signal length longer than. target length, cut them
            padded_signal = signal[:target_samples]
        padded_signals.append(padded_signal)
    padded_signals = np.array(padded_signals)
    
    return padded_signals

def find_first_triple_one(sequence):
    """
    determine the first index of three continous window, whose label is 1
    :param sequence: A list of binary numbers.
    :return: first index of three continous 1 windows
    """
    if len(sequence) < 3:  # less than three windows, return 0
        return 0
    # initialized sum of first three numbers
    window_sum = sequence[0] + sequence[1] + sequence[2]
    # is the first 3 number are all 1, return index 0
    if window_sum == 3:
        return 0
    for i in range(3, len(sequence)):
        # updata the sum
        window_sum += sequence[i] - sequence[i-3]
        # check if the sum is 3
        if window_sum == 3:
            return i - 2  # return index of first window with 1
    return 0  # if no three continous 1, return 0