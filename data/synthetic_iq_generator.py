# data/synthetic_iq_generator.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def generate_iq_sample(fs=1000, duration=1.0, anomaly=False):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    iq = np.exp(1j * 2 * np.pi * 100 * t)  # baseline carrier

    if anomaly:
        burst = np.exp(1j * 2 * np.pi * 400 * t)
        iq += burst * (np.abs(t - 0.5) < 0.05)  # burst near middle
    return iq

def batch_to_scalograms(iq_batch, transform_fn, fs=1000):
    return np.array([transform_fn(iq, fs) for iq in iq_batch])

if __name__ == "__main__":
    # Set path (for de-bugging)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    N_normal = 100
    N_anomalous = 20

    iq_normal = [generate_iq_sample(anomaly=False) for _ in range(N_normal)]
    iq_anomalous = [generate_iq_sample(anomaly=True) for _ in range(N_anomalous)]

    # Choose between FFT and Wavelet
    #
    #   - 'Spectrogram' -> FFT
    #   - 'Scalogram'   -> Wavelet (defaults to Morse Wavelet) 

    pre_processing_type = 'Spectrogram'

    if pre_processing_type == 'Spectrogram':

        from preprocessing.transform import iq_to_spectrogram

        X_normal = batch_to_scalograms(iq_normal, iq_to_spectrogram)
        X_anomalous = batch_to_scalograms(iq_anomalous, iq_to_spectrogram)

    elif pre_processing_type == 'Scalogram':

        from preprocessing.transform import iq_to_scalogram

        X_normal = batch_to_scalograms(iq_normal, iq_to_scalogram)
        X_anomalous = batch_to_scalograms(iq_anomalous, iq_to_scalogram)

    X_all = np.concatenate([X_normal, X_anomalous])
    
    # Get the absolute path to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Get the full path to the data folder
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'synthetic_data')
    os.makedirs(DATA_DIR, exist_ok=True)
        
    # Save scalograms
    np.save(os.path.join(DATA_DIR, pre_processing_type + '_all.npy'), X_all)
    np.save(os.path.join(DATA_DIR, pre_processing_type + '_normal.npy'), X_normal)  
    np.save(os.path.join(DATA_DIR, pre_processing_type + '_anomalous.npy'), X_anomalous)  

    print("Saved scalograms for training/testing.")

    # Output directory for plots
    PLOT_DIR = os.path.join(DATA_DIR, 'plots')
    os.makedirs(PLOT_DIR, exist_ok=True)

    samples_to_plot = [
        ("Normal", iq_normal[0], X_normal[0]),
        ("Anomalous", iq_anomalous[0], X_anomalous[0])
    ]

    for label, iq, scalogram in samples_to_plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

        # Plot real and imaginary parts
        axs[0].plot(iq.real, label='Real')
        axs[0].plot(iq.imag, label='Imag', alpha=0.7)
        axs[0].set_title(f"{label} IQ Signal (Real & Imag)")
        axs[0].legend()

        # Plot magnitude
        axs[1].plot(np.abs(iq))
        axs[1].set_title(f"{label} IQ Magnitude")

        # Plot spectrogram or scalogram
        img = axs[2].imshow(scalogram, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(img, ax=axs[2])
        axs[2].set_title(f"{label} {pre_processing_type}")


        plt.tight_layout()

        # Save plot as PNG
        plot_path = os.path.join(PLOT_DIR, f"{label.lower()}_{pre_processing_type.lower()}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")
