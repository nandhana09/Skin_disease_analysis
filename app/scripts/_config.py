from pathlib import Path

# Setting root directory path for easier refernce in script
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to various directories
MODEL_DIR = BASE_DIR / "models"
TRAIN_DIR = BASE_DIR / "dataset/train_set"
TEST_DIR = BASE_DIR / "dataset/test_set"
TEST_IMG_PATH = BASE_DIR / "img"
RESULT_DIR = BASE_DIR / "result"
