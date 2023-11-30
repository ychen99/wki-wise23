from scipy import fft
import numpy as np
from scipy.signal import butter, filtfilt
from wettbewerb import load_references, get_3montages
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from wettbewerb import load_references, get_3montages
import mne
from scipy import signal as sig
from joblib import load
import ruptures as rpt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def butter_bandpass_filter(_signal, lowcut, highcut, freq, order=4):
    nyquist = 0.5 * freq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_sig = filtfilt(b, a, _signal)
    return filtered_sig


def compute_band_power(sig, freq, freq_low, freq_high):
    spectrum = fft.fft(sig)
    frequencies = np.fft.fftfreq(len(sig), d=1 / freq)
    freq_range = (frequencies >= freq_low) & (frequencies <= freq_high)
    power = np.sum(np.abs(spectrum[freq_range]) ** 2) / len(freq_range)
    return power


def pre_processing(_channels, _data, fr, scaler):
    _montage, _montage_data, _is_missing = get_3montages(_channels, _data)

    amplitude_means = np.zeros(len(_montage))
    band_power = np.zeros(4)

    for j, signal_name in enumerate(_montage):
        signal = _montage_data[j]
        signal_notch = mne.filter.notch_filter(x=signal, Fs=fr, freqs=np.array([50., 100.]),
                                               n_jobs=2, verbose=False)
        signal_filter = butter_bandpass_filter(_signal=signal_notch, lowcut=0.5,
                                               highcut=70.0, freq=fr)

        # Calculate FFT and mean amplitude
        spectrum = fft.fft(signal_filter)
        amplitude_means[j] = np.mean(np.abs(spectrum))

        # Compute band power
        band_power[0] += compute_band_power(signal_filter, fr, 0.5, 4)
        band_power[1] += compute_band_power(signal_filter, fr, 4, 8)
        band_power[2] += compute_band_power(signal_filter, fr, 8, 13)
        band_power[3] += compute_band_power(signal_filter, fr, 13, 30)

    band_power /= len(_montage)
    features = np.concatenate((amplitude_means, band_power), axis=0)
    features_reshaped = features.reshape(1, -1)
    feature_scaled = scaler.transform(features_reshaped)
    return feature_scaled