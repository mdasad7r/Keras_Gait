"""import os
import numpy as np
import cv2

# Constants (optionally move to config.py)
IMAGE_SIZE = (64, 64)
TIME_STEPS = 50  # Trim or pad to fixed length

def load_sequence_images(seq_path):
    """
    Loads and processes silhouette frames from a sequence folder.
    Returns a (TIME_STEPS, 64, 64, 1) array.
    """
    frame_files = sorted(f for f in os.listdir(seq_path) if f.endswith('.png'))
    frames = []

    for fname in frame_files:
        img_path = os.path.join(seq_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0  # Normalize
        frames.append(img)

    # Pad or trim to TIME_STEPS
    if len(frames) >= TIME_STEPS:
        frames = frames[:TIME_STEPS]
    else:
        pad_len = TIME_STEPS - len(frames)
        frames.extend([np.zeros(IMAGE_SIZE, dtype=np.float32)] * pad_len)

    return np.stack(frames, axis=0)[..., np.newaxis]  # (T, H, W, 1)

def split_sequences_by_condition(subject_path):
    """
    Splits nm, bg, cl sequences into train/test per rule.
    Returns: {'train': [seq_names], 'test': [seq_names]}
    """
    sequences = {'nm': [], 'bg': [], 'cl': []}
    for seq_name in os.listdir(subject_path):
        if not os.path.isdir(os.path.join(subject_path, seq_name)):
            continue
        cond_type = seq_name.split('-')[0].lower()
        if cond_type in sequences:
            sequences[cond_type].append(seq_name)

    split = {'train': [], 'test': []}
    for cond, seq_list in sequences.items():
        seq_list.sort()
        if cond == 'nm':
            split['train'].extend(seq_list[:4])
            split['test'].extend(seq_list[4:6])
        elif cond in ['bg', 'cl']:
            split['train'].extend(seq_list[:1])
            split['test'].extend(seq_list[1:2])
    return split


def load_casia_dataset(split_dir):
    """
    Loads one split (train or test) from CASIA-B dataset with angles.
    Returns: X, y
    """
    X, y = [], []

    subject_dirs = sorted(os.listdir(split_dir))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(os.path.join(split_dir, d))]

    for subject_id, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(split_dir, subject_dir)

        for seq_name in os.listdir(subject_path):
            seq_path = os.path.join(subject_path, seq_name)
            if not os.path.isdir(seq_path):
                continue
            for angle in os.listdir(seq_path):
                angle_path = os.path.join(seq_path, angle)
                if not os.path.isdir(angle_path):
                    continue
                try:
                    sequence = load_sequence_images(angle_path)
                    X.append(sequence)
                    y.append(subject_id)
                except Exception as e:
                    print(f"❌ Failed to load {angle_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✅ Loaded split from {split_dir}: X={X.shape}, y={y.shape}")
    return X, y


    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"📦 Final shapes → X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"📦 Final shapes → X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, y_train, X_test, y_test


    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    print(f"📦 Final shapes → X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test, y_test
"""
import os
import numpy as np
import cv2

# === Constants ===
IMAGE_SIZE = (64, 64)
TIME_STEPS = 50  # Fixed length per sequence
CONDITIONS = ["nm", "bg", "cl"]

def load_sequence_images(seq_path):
    frame_files = sorted(f for f in os.listdir(seq_path) if f.endswith(".png"))
    frames = []

    for fname in frame_files:
        img_path = os.path.join(seq_path, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        img = img.astype(np.float32) / 255.0
        frames.append(img)

    # Pad or trim
    if len(frames) >= TIME_STEPS:
        frames = frames[:TIME_STEPS]
    else:
        pad_len = TIME_STEPS - len(frames)
        frames.extend([np.zeros(IMAGE_SIZE, dtype=np.float32)] * pad_len)

    return np.stack(frames, axis=0)[..., np.newaxis]  # (T, H, W, 1)


def is_nm_train(seq_name):
    # Only use nm-01 to nm-04 for training (normal walking)
    return seq_name.startswith("nm") and seq_name.split("-")[1] in ["01", "02", "03", "04"]


def load_training_split(split_dir):
    X, y = [], []
    subject_dirs = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))

    for subject_id, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(split_dir, subject_dir)

        for seq_name in os.listdir(subject_path):
            if not is_nm_train(seq_name):
                continue
            seq_path = os.path.join(subject_path, seq_name)
            for angle in os.listdir(seq_path):
                angle_path = os.path.join(seq_path, angle)
                try:
                    sequence = load_sequence_images(angle_path)
                    X.append(sequence)
                    y.append(subject_id)
                except Exception as e:
                    print(f"❌ Failed to load {angle_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"✅ Training data from {split_dir}: X={X.shape}, y={y.shape}")
    return X, y


def load_gallery_and_probe(split_dir):
    """
    Loads all sequences and categorizes by condition (nm, bg, cl).
    Useful for condition-wise evaluation.
    Returns:
        dict: {condition: (X, y)}
    """
    condition_data = {c: ([], []) for c in CONDITIONS}
    subject_dirs = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))

    for subject_id, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(split_dir, subject_dir)

        for seq_name in os.listdir(subject_path):
            seq_path = os.path.join(subject_path, seq_name)
            condition = seq_name[:2]
            if condition not in CONDITIONS:
                continue

            for angle in os.listdir(seq_path):
                angle_path = os.path.join(seq_path, angle)
                try:
                    sequence = load_sequence_images(angle_path)
                    condition_data[condition][0].append(sequence)
                    condition_data[condition][1].append(subject_id)
                except Exception as e:
                    print(f"❌ Failed to load {angle_path}: {e}")

    # Convert to numpy arrays
    for cond in CONDITIONS:
        X, y = condition_data[cond]
        condition_data[cond] = (np.array(X, dtype=np.float32), np.array(y, dtype=np.int32))
        print(f"✅ {cond.upper()} condition → X={X.shape}, y={y.shape}")

    return condition_data


def load_casia_dataset():
    """
    Loads training set (nm-01 to nm-04) and all test condition splits (nm, bg, cl).
    """
    base_path = "/content/Keras_Gait/casia-b"
    train_path = os.path.join(base_path, "train", "output")
    test_path = os.path.join(base_path, "test", "output")

    X_train, y_train = load_training_split(train_path)
    test_conditions = load_gallery_and_probe(test_path)

    print(f"📦 Final training → X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train, test_conditions
