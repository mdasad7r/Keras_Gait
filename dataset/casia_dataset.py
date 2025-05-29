import os
import numpy as np
import cv2

# Constants (optionally move to config.py)
IMAGE_SIZE = (64, 64)
TIME_STEPS = 30  # Trim or pad to fixed length

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
    Splits nm, bg, cl sequences into train/test per your rule.
    Returns: {'train': [seq_names], 'test': [seq_names]}
    """
    sequences = {'nm': [], 'bg': [], 'cl': []}
    for seq_name in os.listdir(subject_path):
        if not os.path.isdir(os.path.join(subject_path, seq_name)):
            continue
        parts = seq_name.split('-')
        if len(parts) < 2:
            continue
        cond = parts[1]
        if cond in sequences:
            sequences[cond].append(seq_name)

    split = {'train': [], 'test': []}
    for cond, seq_list in sequences.items():
        seq_list.sort()  # Ensure consistent ordering
        if cond == 'nm':
            split['train'].extend(seq_list[:4])
            split['test'].extend(seq_list[4:6])
        elif cond in ['bg', 'cl']:
            split['train'].extend(seq_list[:1])
            split['test'].extend(seq_list[1:2])
    return split

def load_casia_dataset(data_dir):
    """
    Loads the CASIA-B dataset with sequence-based splitting.
    Returns: X_train, y_train, X_test, y_test
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    subject_dirs = sorted(os.listdir(data_dir))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(os.path.join(data_dir, d))]
    
    for subject_id, subject_dir in enumerate(subject_dirs):  # Class IDs: 0 to 123
        subject_path = os.path.join(data_dir, subject_dir)
        split = split_sequences_by_condition(subject_path)

        for set_type in ['train', 'test']:
            for seq_name in split[set_type]:
                seq_path = os.path.join(subject_path, seq_name)
                sequence = load_sequence_images(seq_path)
                if set_type == 'train':
                    X_train.append(sequence)
                    y_train.append(subject_id)
                else:
                    X_test.append(sequence)
                    y_test.append(subject_id)

    # Convert to numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    return X_train, y_train, X_test, y_test
