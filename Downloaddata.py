import os
import zipfile
import subprocess
import sys

DATASET_DOWLOAD_PATH = "uciml/sms-spam-collection-dataset"
ZIP_FILE = "sms-spam-collection-dataset.zip"

def download_dataset():
    # Check if kaggle.json exists in standard locations
    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print("Error: Kaggle API credentials not found.")
        print("Please place your kaggle.json file in", kaggle_dir)
        print("You can download it from your Kaggle account settings.")
        sys.exit(1)

    print(f"Downloading dataset {DATASET_DOWLOAD_PATH} from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_DOWLOAD_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to download dataset. Ensure kaggle package is installed and credentials are correct.")
        sys.exit(1)
        
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall("data")
        
    print("Dataset extracted to 'data' directory.")
    
    # Clean up zip file
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
        print(f"Removed temporary {ZIP_FILE} file.")

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    download_dataset()