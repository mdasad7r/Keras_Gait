import os

# === Base Dataset Paths ===
# These should point to your local folders (relative paths)
BASE_PATH = "casia-b"
TRAIN_PATH = os.path.join(BASE_PATH, "train", "output")
TEST_PATH = os.path.join(BASE_PATH, "test", "output")

# === Image Parameters ===
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 1
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# === Sequence Parameters ===
SEQUENCE_LEN = 30  # Number of frames per sequence

# === Model Parameters ===
FEATURE_DIM = 256         # Output feature size from CNN per frame
TKAN_HIDDEN_DIM = 128     # Hidden size in TKAN layers
NUM_CLASSES = 124         # Number of subjects in CASIA-B
DROPOUT_RATE = 0.3        # Dropout rate between TKAN layers

# === Training Parameters ===
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# === Output Paths ===
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
LOG_DIR = os.path.join(BASE_PATH, "logs")

# === Ensure output directories exist ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
