# test/test_pipeline.py
import os
import sys

# Set path (for de-bugging)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.Autoencoder import ConvAutoencoder
from utils.eval import reconstruction_error

def evaluate_model(model_path, test_data_path, threshold=None):
    data = np.load(test_data_path)  # shape: [N, H, W]
    data = data[:, None, :, :].astype(np.float32)  # [N, 1, H, W]
    data /= np.max(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.astype(np.float32)
    test_tensor = torch.from_numpy(data).to(torch.float32).to(device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(test_tensor).cpu().numpy()
    
    scores = [reconstruction_error(x.squeeze(), y.squeeze()) for x, y in zip(data, outputs)]
    
    if threshold is None:
        threshold = np.mean(scores) + 2 * np.std(scores)

    labels = [1 if s > threshold else 0 for s in scores]  # 1 = anomaly
    return scores, labels, threshold

if __name__ == "__main__":
        # Choose between FFT and Wavelet
    #
    #   - 'Spectrogram' -> FFT
    #   - 'Scalogram'   -> Wavelet (defaults to Morse Wavelet) 

    pre_processing_type = 'Scalogram'

    # Get the absolute path to the project root
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Get the full path to data
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'synthetic iq', pre_processing_type + '_all.npy')
    scores, labels, threshold = evaluate_model(
        model_path='checkpoints/autoencoder.pth',
        test_data_path='data/synthetic_iq/' + pre_processing_type + '_all.npy'
    )
    print(f"Threshold: {threshold:.4f}")
    print(f"Anomaly count: {sum(labels)} / {len(labels)}")
