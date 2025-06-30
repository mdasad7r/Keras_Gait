import os
import numpy as np
import cv2

# === Constants ===
IMAGE_SIZE = (64, 64)
TIME_STEPS = 50

# Mapping folder prefixes to evaluation labels
CONDITIONS = {
    "nm": "NM#5-6",
    "bg": "BG#1-2",
    "cl": "CL#1-2"
}

def load_sequence_images(seq_path):
    """
    Loads all PNG images in a sequence folder, resizes them, normalizes,
    and pads/trims to fixed length TIME_STEPS.
    Returns: numpy array of shape (T, H, W, 1)
    """
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

def load_gallery_and_probe(split_dir):
    """
    Loads CASIA-B test sequences organized by:
        { "NM#5-6": { "0": (X, y), "18": (X, y), ... }, ... }

    Returns:
        dict of dicts: condition → view angle → (X, y)
    """
    condition_data = {v: {} for v in CONDITIONS.values()}
    subject_dirs = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))

    for subject_id, subject_dir in enumerate(subject_dirs):
        subject_path = os.path.join(split_dir, subject_dir)

        for seq_name in os.listdir(subject_path):
            condition_key = seq_name[:2]  # e.g., "nm", "bg", "cl"
            if condition_key not in CONDITIONS:
                continue

            condition_name = CONDITIONS[condition_key]
            seq_path = os.path.join(subject_path, seq_name)

            for view in os.listdir(seq_path):
                view_path = os.path.join(seq_path, view)
                try:
                    sequence = load_sequence_images(view_path)
                    if view not in condition_data[condition_name]:
                        condition_data[condition_name][view] = ([], [])
                    condition_data[condition_name][view][0].append(sequence)
                    condition_data[condition_name][view][1].append(subject_id)
                except Exception as e:
                    print(f"❌ Failed to load {view_path}: {e}")

    # Convert lists to numpy arrays
    for cond in condition_data:
        for view in condition_data[cond]:
            X, y = condition_data[cond][view]
            condition_data[cond][view] = (np.array(X, dtype=np.float32), np.array(y, dtype=np.int32))
            print(f"✅ {cond} - View {view}: X={len(X)}, y={len(y)}")

    return condition_data
