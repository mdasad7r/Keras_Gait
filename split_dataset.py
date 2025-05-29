import os
import shutil

# === Source dataset path (unstructured original output) ===
SOURCE_PATH = "casia-b/output"

# === Destination paths ===
TRAIN_PATH = "casia-b/train/output"
TEST_PATH = "casia-b/test/output"

# Create output directories
os.makedirs(TRAIN_PATH, exist_ok=True)
os.makedirs(TEST_PATH, exist_ok=True)

# === Split logic ===
TRAIN_CONDITIONS = {
    "nm": ["nm-01", "nm-02", "nm-03", "nm-04"],
    "bg": ["bg-01"],
    "cl": ["cl-01"]
}

TEST_CONDITIONS = {
    "nm": ["nm-05", "nm-06"],
    "bg": ["bg-02"],
    "cl": ["cl-02"]
}

def move_sequences(subject, conditions, source_base, dest_base):
    subject_source = os.path.join(source_base, subject)
    subject_dest = os.path.join(dest_base, subject)
    os.makedirs(subject_dest, exist_ok=True)

    for cond_type, cond_names in conditions.items():
        for cond_name in cond_names:
            seq_folder = os.path.join(subject_source, cond_name)
            if os.path.isdir(seq_folder):
                dest_folder = os.path.join(subject_dest, cond_name)
                shutil.move(seq_folder, dest_folder)

def main():
    subjects = sorted(os.listdir(SOURCE_PATH))
    for subject in subjects:
        move_sequences(subject, TRAIN_CONDITIONS, SOURCE_PATH, TRAIN_PATH)
        move_sequences(subject, TEST_CONDITIONS, SOURCE_PATH, TEST_PATH)

    print("✅ Dataset split complete.")
    print(f"Train sequences saved to: {TRAIN_PATH}")
    print(f"Test sequences saved to:  {TEST_PATH}")

if __name__ == "__main__":
    main()
