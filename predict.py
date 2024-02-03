# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Maurice Rohr, Dirk Schweickard
"""


import numpy as np
import json
import os
import rfc
from typing import List, Tuple, Dict, Any
from wettbewerb import get_3montages
from joblib import load
# Pakete aus dem Vorlesungsbeispiel
import mne
import ruptures as rpt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import butter, filtfilt
from scipy import fft
from scipy import signal as sig



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str, model_name : str='random_forest_model.joblib') -> Dict[str,Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  

    # Initialisiere Return (Ergebnisse)
    seizure_present = True # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 29.5   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.99 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 999999  # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0   # gibt die Unsicherheit bezüglich des Endes an (optional)

    # Hier könnt ihr euer vortrainiertes Modell laden (Kann auch aus verschiedenen Dateien bestehen)
    """model = load(model_name)
    rf_classifier_loaded = model['model']
    scaler_loaded = model['scaler']
    
    
    
    features = lstm.pre_processing(channels, data, fs, scaler_loaded)
    y_pred = rf_classifier_loaded.predict(features)
    seizure_present = y_pred > 0"""
    
    
    rf = load(model_name)
    padded_signal = rfc.pad_eeg_signals(data, 300, fs)
    scaler = StandardScaler()
    padded_signal = scaler.fit_transform(padded_signal)  
    window_size = 10
    step_size = 5
    window_samples = window_size * fs
    step_samples = step_size * fs
    window_features = []
        
    for start_idx in range(0, padded_signal.shape[1] - window_samples + 1, step_samples):
        window = padded_signal[:, start_idx:start_idx + window_samples]
        window_feature = rfc.pre_processing(channels, window, fs)
        window_features.append(window_feature)
    window_features = np.array(window_features)
    
    predictions = rf.predict(window_features)
    onset = 5 * rfc.find_first_triple_one(predictions)
    
    seizure_present = onset>0
    

    
    
#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit predction - Muss unverändert bleiben!
