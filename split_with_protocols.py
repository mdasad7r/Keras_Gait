import os
import shutil

# === Original unstructured dataset path ===
SOURCE_PATH = "CASIA_B/output"

# === Structured destination paths ===
TRAIN_PATH = "casia-b/train/output"
TEST_PATH = "casia-b/test/output"

# === Protocol-defined subject IDs ===
TRAIN_SUBJECTS = [f"{i:03d}" for i in range(1, 75)]     # 001–074
TEST_SUBJECTS = [f"{i:03d}" for i in range(75, 125)]    # 075–124

# === Training/Testing condition selection ===
TRAIN_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
TEST_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"]

def move_sequences(subjects, conditions, dest_base):
    os.makedirs(dest_base, exist_ok=True)

    for subject in subjects:
        src_subject = os.path.join(SOURCE_PATH, subject)
        dst_subject = os.path.join(dest_base, subject)
        os.makedirs(dst_subject, exist_ok=True)

        for cond in conditions:
            src_cond = os.path.join(src_subject, cond)
            if os.path.isdir(src_cond):
                dst_cond = os.path.join(dst_subject, cond)
                shutil.move(src_cond, dst_cond)

def main():
    move_sequences(TRAIN_SUBJECTS, TRAIN_CONDITIONS, TRAIN_PATH)
    move_sequences(TEST_SUBJECTS, TEST_CONDITIONS, TEST_PATH)

    print("✅ CASIA-B protocol dataset split complete.")
    print(f"✔ Train subjects: {len(TRAIN_SUBJECTS)} → {TRAIN_PATH}")
    print(f"✔ Test subjects : {len(TEST_SUBJECTS)} → {TEST_PATH}")

if __name__ == "__main__":
    main()
