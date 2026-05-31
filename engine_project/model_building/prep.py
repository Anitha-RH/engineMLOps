# for data manipulation
import pandas as pd
import os
# for train/test split
from sklearn.model_selection import train_test_split
# for hugging face authentication to upload files
from huggingface_hub import HfApi, hf_hub_download

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_REPO_ID = "AnithaRH/engine-predictive-maintenance"
DATASET_FILENAME = "engine_data.csv"

# Download the main dataset to a local path before reading with pandas
local_engine_dataset_path = hf_hub_download(
    repo_id=DATASET_REPO_ID,
    filename=DATASET_FILENAME,
    repo_type="dataset"
)
engine_dataset = pd.read_csv(local_engine_dataset_path)
print("Dataset loaded successfully.")

# Rename columns to remove spaces
RENAME_MAP = {
    "Engine rpm": "Engine_RPM",
    "Lub oil pressure": "Lub_Oil_Pressure",
    "Fuel pressure": "Fuel_Pressure",
    "Coolant pressure": "Coolant_Pressure",
    "lub oil temp": "Lub_Oil_Temperature",
    "Coolant temp": "Coolant_Temperature",
    "Engine Condition": "E
