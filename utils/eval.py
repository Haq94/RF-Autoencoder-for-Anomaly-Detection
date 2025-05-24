import torch
import numpy as np

def reconstruction_error(original, reconstructed):
    """
    Compute MSE between original and reconstructed spectrogram or scalogram.
    """
    return np.mean((original - reconstructed) ** 2)

def evaluate_autoencoder(model, data_tensor):
    model.eval()
    with torch.no_grad():
        reconstructed = model(data_tensor).cpu().numpy()
    return reconstructed
