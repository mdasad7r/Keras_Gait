import os
import numpy as np
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
    EarlyStopping, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight

import config
from model.resnet50_tkan import build_model
from dataset.casia_dataset_all import load_casia_dataset_all

# === Protocol-specific split ===
TRAIN_SUBJECTS = list(range(1, 75))      # Subjects 001–074
TEST_SUBJECTS = list(range(75, 125))     # Subjects 075–124

TRAIN_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
GALLERY_CONDITIONS = ["nm-01", "nm-02", "nm-03", "nm-04"]
PROBE_CONDITIONS = ["nm-05", "nm-06", "bg-01", "bg-02", "cl-01", "cl-02"]

# === Load datasets ===
X_train, y_train = load_casia_dataset_all(subject_ids=TRAIN_SUBJECTS, conditions=TRAIN_CONDITIONS)
X_gallery, y_gallery = load_casia_dataset_all(subject_ids=TEST_SUBJECTS, conditions=GALLERY_CONDITIONS)
X_probe, y_probe = load_casia_dataset_all(subject_ids=TEST_SUBJECTS, conditions=PROBE_CONDITIONS)

print(f"✅ Train: {X_train.shape}, {len(np.unique(y_train))} classes")
print(f"🎯 Gallery: {X_gallery.shape}, Probes: {X_probe.shape}")

# === Build model ===
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=[SparseCategoricalAccuracy()]
)
model.summary()

# === Class weights ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# === Callbacks ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
steps_per_epoch = len(X_train) // config.BATCH_SIZE
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(config.CHECKPOINT_DIR, "epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq=5 * steps_per_epoch
    ),
    TensorBoard(log_dir=os.path.join(config.LOG_DIR, f"{timestamp}_resnet50_tkan")),
    CSVLogger(os.path.join(config.LOG_DIR, "training_log_resnet50.csv"), append=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# === Train ===
model.fit(
    X_train, y_train,
    validation_data=(X_gallery, y_gallery),  # Optional: monitor gallery val
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    shuffle=True
)

# === Evaluate (probe vs gallery) ===
print("🔍 Evaluating on probe set:")
model.evaluate(X_probe, y_probe)
