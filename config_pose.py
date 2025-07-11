import os

# === Dataset Paths ===
BASE_PATH = "casia-b"
TRAIN_PATH = os.path.join(BASE_PATH, "train", "output")
TEST_PATH = os.path.join(BASE_PATH, "test", "output")

# === Input Sequence Parameters ===
SEQUENCE_LEN = 50              # Number of frames per sequence
POSE_FEATURE_DIM = 51          # 17 keypoints × 3 (x, y, confidence) per frame

# === Image Reshaping for CNNs ===
CNN_HEIGHT = 8                 # After padding to 64 and reshaping
CNN_WIDTH = 8
CNN_CHANNELS = 1               # For grayscale input (custom CNN)
RESNET_CHANNELS = 3            # For ResNet18/50 (requires 3 channels)

# === Model Parameters ===
FEATURE_DIM = 256              # Output dimension from each encoder
TKAN_HIDDEN_DIM = 256          # TKAN internal projection
NUM_CLASSES = 124              # Number of subjects in CASIA-B
DROPOUT_RATE = 0.3             # Dropout after encoders and TKAN

# === Training Parameters ===
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

# === Output & Logging ===
CHECKPOINT_DIR = os.path.join(BASE_PATH, "checkpoints")
LOG_DIR = os.path.join(BASE_PATH, "logs")

# === Ensure output directories exist ===
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
