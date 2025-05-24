import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt  
from sklearn.metrics import roc_auc_score, roc_curve
import os  

from models.Autoencoder import ConvAutoencoder
from data.synthetic_iq_generator import generate_iq_sample
from data.synthetic_iq_generator import batch_to_scalograms
from utils.eval import reconstruction_error
from utils.eval import evaluate_autoencoder
from train.train_autoencoder import train_autoencoder
from test.test_pipeline import evaluate_model

def main():
    # -----------------------------
    # 1. Configuration
    # -----------------------------
    config = {
        "sample_rate": 1e3,
        "duration": 1.0,
        "Num_train": int(8e2),
        "Num_normal_test": int(1e2),
        "Num_anomaly_test": int(1e2),
        "transform": 'Spectrogram',
        "plots": True,
        "threshold": 1e-3,
        "batch_size": 64,
        "epochs": 20,
        "lr": 1e-3,
        "use_gpu": torch.cuda.is_available()
    }

    checkpoint_path = 'checkpoints/' + config['transform'].lower() + '_autoencoder.pth'

    device = torch.device("cuda" if config["use_gpu"] else "cpu")

    # -----------------------------
    # 2. Generate or Load Data
    # -----------------------------
    iq_train = [generate_iq_sample(fs=config["sample_rate"], duration=config["duration"], anomaly=False) 
    for _ in range(config["Num_train"])]

    iq_test = [generate_iq_sample(fs=config["sample_rate"], duration=config["duration"], anomaly=False) 
    for _ in range(config["Num_normal_test"])] + [generate_iq_sample(fs=config["sample_rate"], duration=config["duration"], anomaly=True) 
    for _ in range(config["Num_anomaly_test"])]

    labels_test = np.concatenate((np.zeros(config["Num_normal_test"]), np.ones(config["Num_normal_test"])))

    if config['plots']:

        # Plot IQ
        fig, axs = plt.subplots(2, 1)  # 2 rows, 1 column

        # Normal plot
        axs[0].plot(np.real(iq_test[0]), label='Real', color='blue')
        axs[0].plot(np.imag(iq_test[0]), label='Imag', color='red')
        axs[0].set_title('Normal')
        axs[0].legend()

        # Anomalie plot
        axs[1].plot(np.real(iq_test[config["Num_normal_test"]]), label='Real', color='blue')
        axs[1].plot(np.imag(iq_test[config["Num_normal_test"]]), label='Imag', color='red')
        axs[1].set_title('Anomalie')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
            
    # ------------------------------------------
    # 3. Generate Spectrogram/ Scalogram Images
    # ------------------------------------------
    if config['transform'] == 'Spectrogram':

        from preprocessing.transform import iq_to_spectrogram

        image_train = batch_to_scalograms(iq_train, iq_to_spectrogram, config["sample_rate"])
        image_test = batch_to_scalograms(iq_test, iq_to_spectrogram, config["sample_rate"])

    elif config['transform'] == 'Scalogram':

        from preprocessing.transform import iq_to_scalogram

        image_train = batch_to_scalograms(iq_train, iq_to_scalogram, config["sample_rate"])
        image_test = batch_to_scalograms(iq_test, iq_to_scalogram, config["sample_rate"])

    # ------------------------------------------
    # 4. Save Test Data and Labels
    # ------------------------------------------

    # Folder and file setup
    folder_name = "main_data"
    file_name = 'test_data.npy'
    test_data_path = os.path.join('data', folder_name, file_name)

    # Create the full folder path if it doesn't exist
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    # Save the file
    np.save(test_data_path, image_test)

    if config['plots']:

        # Create subplots
        fig, axs = plt.subplots(1, 2)

        # First heatmap
        im1 = axs[0].imshow(image_test[0], cmap='viridis')
        axs[0].set_title("Normal")
        plt.colorbar(im1, ax=axs[0])

        # Second heatmap
        im2 = axs[1].imshow(image_test[config["Num_normal_test"]], cmap='viridis')
        axs[1].set_title("Anomalie")
        plt.colorbar(im2, ax=axs[1])

        plt.tight_layout()
        plt.show()

    # -----------------------------
    # 5. Prepare PyTorch Dataset
    # -----------------------------
    image_train = torch.from_numpy(image_train).float() 
    image_train = image_train[:, None, :, :]  # add channel dim → [N, 1, H, W]

    # -----------------------------
    # 6. Train Autoencoder
    # -----------------------------
    train_autoencoder(image_train, checkpoint_path, config["epochs"], config["batch_size"], config["lr"])

    # -----------------------------
    # 7. Evaluate / Detect Anomalies
    # -----------------------------
    scores, labels, threshold = evaluate_model(checkpoint_path, test_data_path, config["threshold"])

    # -----------------------------
    # 8. Visualize and Define Metrics
    # -----------------------------

    # Classification Accuracy Percentage
    class_acc = 100*(labels_test == labels).mean()
    print(f"Classification Accuracy: {class_acc} %")
    print("ROC AUC: ", roc_auc_score(labels_test, scores))

    if config["plots"]:

        # Plot Scores
        plt.plot(scores)
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.xlabel("Training Sample Number")
        plt.ylabel("MSE")
        plt.title("MSE vs Training Sample Number")
        plt.legend()
        plt.grid(True)
        plt.show()

        # ROC Plot
        fpr, tpr, _ = roc_curve(labels_test, scores)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(labels_test, scores):.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
