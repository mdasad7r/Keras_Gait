import os
import pickle
import numpy as np
import tensorflow as tf
from config_pose import SEQUENCE_LEN, POSE_FEATURE_DIM, CNN_HEIGHT, CNN_WIDTH


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

    return np.array(padded_seq, dtype=np.float32)  # shape: (T, 8, 8, 1)


def casia_pose_generator(data_root, max_frames=SEQUENCE_LEN, pose_dim=POSE_FEATURE_DIM):
    """
    Generator that yields (pose_sequence, label) for each .pkl file
    """
    for sid in sorted(os.listdir(data_root)):
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
                            yield sequence, label
                        except Exception as e:
                            print(f"Error loading {pkl_path}: {e}")
                        break  # one .pkl per folder


def build_casia_pose_dataset(data_root, batch_size=16, shuffle=True):
    """
    Returns a tf.data.Dataset that yields (pose_sequence, label) batches lazily.
    """
    output_types = (tf.float32, tf.int32)
    output_shapes = ((SEQUENCE_LEN, CNN_HEIGHT, CNN_WIDTH, 1), ())

    ds = tf.data.Dataset.from_generator(
        lambda: casia_pose_generator(data_root),
        output_types=output_types,
        output_shapes=output_shapes
    )

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
