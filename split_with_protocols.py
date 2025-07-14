import os
import shutil

# === Source dataset path (adjust if needed) ===
SOURCE_PATH = "CASIA-B_SimCC"

# === Destination paths for split datasets ===
TRAIN_PATH = os.path.join("casia-b", "train", "output")
TEST_PATH = os.path.join("casia-b", "test", "output")

# === Protocol-defined subject IDs ===
TRAIN_SUBJECTS = [f"{i:03d}" for i in range(1, 75)]     # Subjects 001–074
TEST_SUBJECTS = [f"{i:03d}" for i in range(75, 125)]    # Subjects 075–124

# === Conditions to use for training/testing ===
TRAIN_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
TEST_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"]

def copy_pose_sequences(subjects, conditions, destination_root):
    for subject in subjects:
        for condition in conditions:
            src_cond_path = os.path.join(SOURCE_PATH, subject, condition)
            if not os.path.isdir(src_cond_path):
                continue

            for seq_folder in os.listdir(src_cond_path):  # e.g., 000, 001...
                src_seq_path = os.path.join(src_cond_path, seq_folder)
                if not os.path.isdir(src_seq_path):
                    continue

                # Final destination path
                dst_seq_path = os.path.join(destination_root, subject, condition, seq_folder)

                if not os.path.exists(dst_seq_path):
                    os.makedirs(os.path.dirname(dst_seq_path), exist_ok=True)
                    shutil.copytree(src_seq_path, dst_seq_path)
                else:
                    print(f"⚠ Skipping existing: {dst_seq_path}")

def main():
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)

    print("🚀 Starting CASIA-B dataset split...")

    copy_pose_sequences(TRAIN_SUBJECTS, TRAIN_CONDITIONS, TRAIN_PATH)
    copy_pose_sequences(TEST_SUBJECTS, TEST_CONDITIONS, TEST_PATH)

    print("✅ Dataset split complete.")
    print(f"✔ Train subjects: {len(TRAIN_SUBJECTS)} → {TRAIN_PATH}")
    print(f"✔ Test subjects : {len(TEST_SUBJECTS)} → {TEST_PATH}")

if __name__ == "__main__":
    main()
