import os
import numpy as np
import cv2
from tqdm import tqdm
from config import SEQUENCE_LEN


def load_sequence(sequence_path, img_size=(64, 64), max_frames=SEQUENCE_LEN):
    """
    Load a silhouette sequence from a view folder. Pads with zeros if fewer than max_frames.
    Returns: (T, H, W, 1)
    """
    frames = sorted(os.listdir(sequence_path))[:max_frames]
    sequence = []

    for frame_file in frames:
        frame_path = os.path.join(sequence_path, frame_file)
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        sequence.append(img)

    if not sequence:
        return None

    sequence = np.stack(sequence, axis=0)  # (T, H, W)

    # Pad if sequence is shorter than max_frames
    if sequence.shape[0] < max_frames:
        pad_len = max_frames - sequence.shape[0]
        pad_shape = (pad_len, *sequence.shape[1:])
        pad = np.zeros(pad_shape, dtype=sequence.dtype)
        sequence = np.concatenate([sequence, pad], axis=0)

    return np.expand_dims(sequence, axis=-1)  # (T, H, W, 1)


def load_casia_dataset_all(data_root="output", subject_ids=None, conditions=None,
                           img_size=(64, 64), max_frames=SEQUENCE_LEN):
    """
    Load CASIA-B dataset as sequences.

    Args:
        data_root: base directory of dataset (e.g., casia-b/train/output)
        subject_ids: optional list of subject IDs to include
        conditions: optional list of condition folder names (e.g., ["nm-01", "cl-02"])
        img_size: output image size
        max_frames: number of frames per sequence (defaults to config)

    Returns:
        X: np.array of shape (N, T, H, W, 1)
        y: np.array of subject IDs (N,)
    """
    X = []
    y = []
    subject_ids = set(subject_ids) if subject_ids else None
    conditions = set(conditions) if conditions else None

    subjects = sorted(os.listdir(data_root))
    for sid in tqdm(subjects, desc="Loading subjects"):
        if not sid.isdigit():
            continue
        sid_int = int(sid)
        if subject_ids and sid_int not in subject_ids:
            continue

        sid_path = os.path.join(data_root, sid)
        for condition in os.listdir(sid_path):
            if conditions and condition not in conditions:
                continue

            cond_path = os.path.join(sid_path, condition)
            for view in sorted(os.listdir(cond_path)):
                view_path = os.path.join(cond_path, view)
                if not os.path.isdir(view_path):
                    continue

                seq = load_sequence(view_path, img_size, max_frames)
                if seq is not None:
                    X.append(seq)
                    y.append(sid_int)

    X = np.array(X)
    y = np.array(y)
    return X, y

def load_dataset(train_path, test_path, sequence_len, image_size):
    """
    Wrapper to load train and test sets using config values.
    """
    X_train, y_train = load_casia_dataset_all(
        data_root=train_path,
        img_size=image_size,
        max_frames=sequence_len
    )
    X_test, y_test = load_casia_dataset_all(
        data_root=test_path,
        img_size=image_size,
        max_frames=sequence_len
    )
    return X_train, y_train, X_test, y_test
