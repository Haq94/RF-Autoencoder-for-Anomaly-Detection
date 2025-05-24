import numpy as np
import pywt
import scipy.signal

def iq_to_spectrogram(iq, fs=1.0, nperseg=128):
    f, t, Sxx = scipy.signal.spectrogram(np.abs(iq), fs=fs, nperseg=nperseg)
    return Sxx

def iq_to_scalogram(iq, fs=1.0, wavelet='morl', max_scale=128):
    data = np.abs(iq)
    scales = np.arange(1, max_scale)
    coeffs, _ = pywt.cwt(data, scales, wavelet, sampling_period=1/fs)
    scalogram = np.abs(coeffs)
    return scalogram
