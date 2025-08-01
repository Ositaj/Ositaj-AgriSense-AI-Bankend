import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Setup
dataset_name = "emmarex/plantdisease"
download_path = "dataset"
zip_file_path = os.path.join(download_path, "plantdisease.zip")  # ✅ corrected here

# Create folder if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Authenticate and download
api = KaggleApi()
api.authenticate()
print("✅ Authenticated Kaggle API")

print(f"⬇️ Downloading {dataset_name}...")
api.dataset_download_files(dataset_name, path=download_path, unzip=False)
print("✅ Download complete!")

# Extract the ZIP file
print("📦 Extracting files...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)
print("✅ Extraction complete!")

# Cleanup
os.remove(zip_file_path)
print("🧹 Zip file removed!")

# Final check
print("✅ Dataset is ready at:", os.path.abspath(download_path))
