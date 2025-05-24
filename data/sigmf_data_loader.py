import json
from pathlib import Path
import numpy as np
import hashlib
from sigmf import SigMFFile

# === Step 1: Set paths to .sigmf-meta and .sigmf-data ===
meta_path = Path(r"C:\Users\Omer\Downloads\Senior Perception Engineer HW Assigment\Senior Perception Engineer HW Assigment\data\sample2.sigmf-meta")
data_path = meta_path.with_suffix(".sigmf-data")  # same name, .sigmf-data extension

# === Step 2: Load metadata ===
with open(meta_path, "r") as f:
    metadata = json.load(f)

sigmf_file = SigMFFile(metadata=metadata)

# === Step 3: Recalculate the SHA512 hash ===
def calculate_sha512(file_path):
    sha512 = hashlib.sha512()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            sha512.update(chunk)
    return sha512.hexdigest()

correct_hash = calculate_sha512(data_path)

# === Update the checksum in metadata manually ===
if "captures" in metadata and len(metadata["captures"]) > 0:
    # Replace the checksum with new hash
    metadata["captures"][0]["core:checksum"] = f"sha512:{correct_hash}"
else:
    raise ValueError("Metadata file does not contain 'captures' with checksum to update.")

# === Save the fixed metadata back ===
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Updated metadata with correct hash.")

# === Now load with SigMFFile ===
sigmf_file = SigMFFile(metadata=metadata)
sigmf_file.set_data_file(str(data_path))  # associate data file (no validation param)

# === Load samples ===
samples = np.fromfile(data_path, dtype=np.complex64)

print(f"✅ Loaded {len(samples)} complex samples")
print(f"Sample rate: {sigmf_file.get_global_field('sample_rate')} Hz")
print(f"Center frequency: {sigmf_file.get_global_field('frequency')} Hz")

# # === Step 4: Fix the hash in the metadata ===
# first_capture = sigmf_file.get_captures()
# data_file_name = data_path.name

# sigmf_file.set_data_file(Path(data_path))  # Associate the file
# sigmf_file.set_file_hash(correct_hash)    # Update the hash
# sigmf_file.set_file_size(data_path.stat().st_size)  # Ensure size is correct

# # === Step 5: Save fixed metadata ===
# with open(meta_path, "w") as f:
#     f.write(sigmf_file.dumps())
# print("✅ Metadata file hash fixed.")

# # === Step 6: Load samples ===
# samples = np.fromfile(data_path, dtype=np.complex64)
# print(f"✅ Loaded {len(samples)} complex samples")
# print(f"Sample rate: {sigmf_file.get_global_field('sample_rate')} Hz")
# print(f"Center frequency: {sigmf_file.get_global_field('frequency')} Hz")



