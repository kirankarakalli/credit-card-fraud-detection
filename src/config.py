import os

# Get the root directory of your project (2 levels above this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "creditcard.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model directory
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Logs directory
LOG_DIR = os.path.join(ROOT_DIR, "logs")

# Train-test split settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
