
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_model_save_dir():
    return os.getenv("MODEL_SAVE_DIR", "models/")

def get_log_level():
    return os.getenv("LOG_LEVEL", "info")
