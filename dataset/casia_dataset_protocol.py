import os
import numpy as np
import cv2
from tqdm import tqdm

def load_sequence(sequence_path, img_size=(64, 64), max_frames=30):
    """Load a sequence of silhouette frames from a view folder."""
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
    return np.expand_dims(sequence, axis=-1)  # (T, H, W, 1)

def load_casia_dataset_all(data_root="output", subject_ids=None, conditions=None,
                           img_size=(64, 64), max_frames=30):
    """
    Load CASIA-B sequences based on subject and condition.
    
    Returns:
        X: np.array (N, T, H, W, 1)
        y: np.array (N,)
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
            for view in os.listdir(cond_path):
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
