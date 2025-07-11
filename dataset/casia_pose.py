import os
import pickle
import numpy as np
from tqdm import tqdm
from config import SEQUENCE_LEN, POSE_FEATURE_DIM, CNN_HEIGHT, CNN_WIDTH


def load_pose_sequence(pkl_path, max_frames=SEQUENCE_LEN, pose_dim=POSE_FEATURE_DIM,
                       reshape_to=(CNN_HEIGHT, CNN_WIDTH), include_confidence=True):
    """
    Loads a single pose sequence from a .pkl file.
    Output: (T, H, W, 1)
    """
    with open(pkl_path, 'rb') as f:
        pose_seq = pickle.load(f)  # (T, 17, 3)

    pose_seq = np.array(pose_seq)

    if include_confidence:
        flat_seq = pose_seq.reshape(pose_seq.shape[0], -1)  # (T, 51)
    else:
        flat_seq = pose_seq[:, :, :2].reshape(pose_seq.shape[0], -1)  # (T, 34)

    # Pad or truncate to fixed SEQUENCE_LEN
    T = flat_seq.shape[0]
    if T < max_frames:
        pad = np.zeros((max_frames - T, flat_seq.shape[1]), dtype=flat_seq.dtype)
        flat_seq = np.concatenate([flat_seq, pad], axis=0)
    else:
        flat_seq = flat_seq[:max_frames]

    # Pad each frame to 64, then reshape to 8x8
    padded_seq = []
    for frame in flat_seq:
        if frame.shape[0] < 64:
            pad_len = 64 - frame.shape[0]
            frame = np.pad(frame, (0, pad_len))
        frame_2d = frame.reshape(reshape_to + (1,))  # (8, 8, 1)
        padded_seq.append(frame_2d)

    return np.array(padded_seq)  # shape: (T, 8, 8, 1)


def load_casia_pose_dataset(data_root, max_frames=SEQUENCE_LEN, pose_dim=POSE_FEATURE_DIM):
    """
    Loads all .pkl pose sequences from CASIA-B_HRNet structure.
    Returns:
        X: (N, T, H, W, 1)
        y: (N,)
    """
    X, y = [], []

    for sid in tqdm(sorted(os.listdir(data_root)), desc="Loading subjects"):
        sid_path = os.path.join(data_root, sid)
        if not os.path.isdir(sid_path) or not sid.isdigit():
            continue
        label = int(sid)

        for cond in os.listdir(sid_path):
            cond_path = os.path.join(sid_path, cond)
            if not os.path.isdir(cond_path):
                continue

            for seq in os.listdir(cond_path):
                seq_path = os.path.join(cond_path, seq)
                if not os.path.isdir(seq_path):
                    continue

                for file in os.listdir(seq_path):
                    if file.endswith(".pkl"):
                        pkl_path = os.path.join(seq_path, file)
                        try:
                            sequence = load_pose_sequence(pkl_path, max_frames=max_frames, pose_dim=pose_dim)
                            X.append(sequence)
                            y.append(label)
                        except Exception as e:
                            print(f"Error loading {pkl_path}: {e}")
                        break  # one .pkl per folder

    return np.array(X), np.array(y)
