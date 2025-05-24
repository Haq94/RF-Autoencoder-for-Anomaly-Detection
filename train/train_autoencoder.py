# train/train_autoencoder.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Set path (for de-bugging)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.Autoencoder import ConvAutoencoder

def load_data(file_path):
    data = np.load(file_path)  # shape: [N, H, W]
    data = data[:, None, :, :]  # add channel dim → [N, 1, H, W]
    data = data.astype(np.float32)
    data /= np.max(data)  # normalize to [0, 1]
    return torch.tensor(data)

def train_autoencoder(data_tensor, save_path, num_epochs=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(loader):.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Choose between FFT and Wavelet
    #
    #   - 'Spectrogram' -> FFT
    #   - 'Scalogram'   -> Wavelet (defaults to Morse Wavelet) 

    pre_processing_type = 'Scalogram'

    os.chdir(r"C:\Users\Omer\Documents\Python\rf anomaly detection")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/synthetic_iq/' + pre_processing_type + '_normal.npy')
    parser.add_argument('--save', type=str, default='checkpoints/' + pre_processing_type.lower() + '_autoencoder.pth')
    args = parser.parse_args()
    
    data = load_data(args.data)
    train_autoencoder(data, save_path=args.save)
