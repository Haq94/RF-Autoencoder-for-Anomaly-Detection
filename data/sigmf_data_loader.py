import json
from pathlib import Path
import numpy as np
import hashlib
from sigmf import SigMFFile

def sigmf_data_loader(meta_path):
    """
    Load IQ samples and metadata from SigMF files.

    Args:
        meta_path (str or Path): Path to the .sigmf-meta JSON file.

    Returns:
        samples (np.ndarray): Complex64 IQ samples.
        metadata (dict): Loaded metadata dictionary.
        sample_rate (float or None): Sample rate in Hz if present.
        center_freq (float or None): Center frequency in Hz if present.
    """
    meta_path = Path(meta_path)
    data_path = meta_path.with_suffix(".sigmf-data")

    # Load metadata JSON
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Calculate and update SHA512 checksum
    def calculate_sha512(file_path):
        sha512 = hashlib.sha512()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha512.update(chunk)
        return sha512.hexdigest()

    correct_hash = calculate_sha512(data_path)

    if "captures" in metadata and len(metadata["captures"]) > 0:
        metadata["captures"][0]["core:checksum"] = f"sha512:{correct_hash}"
    else:
        raise ValueError("Metadata file missing 'captures' or checksum field to update.")

    # Save updated metadata back (optional, comment out if you don't want to overwrite)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Load IQ samples as complex64
    samples = np.fromfile(data_path, dtype=np.complex64)

    # Extract useful metadata fields (optional)
    sample_rate = metadata.get("global", {}).get("sample_rate", None)
    center_freq = metadata.get("global", {}).get("frequency", None)

    return samples, metadata, sample_rate, center_freq


# === Example usage ===
if __name__ == "__main__":
    meta_file = r"C:\Users\Omer\Downloads\Senior Perception Engineer HW Assigment\Senior Perception Engineer HW Assigment\data\sample2.sigmf-meta"
    samples, metadata, fs, fc = sigmf_data_loader(meta_file)

    print(f"Loaded {len(samples)} complex IQ samples")
    print(f"Sample rate: {fs} Hz")
    print(f"Center frequency: {fc} Hz")
