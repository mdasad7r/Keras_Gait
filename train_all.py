import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight

# === Load configs and model ===
import config
from model.cnn_tkan import build_model  # 🔄 updated import
from dataset.casia_dataset_all import load_casia_dataset  # should support train_conditions param

# === Load dataset: Train on nm#01-04, bg#01, cl#01 ===
train_conditions = ["nm-01", "nm-02", "nm-03", "nm-04", "bg-01", "cl-01"]
X_train, y_train, test_conditions = load_casia_dataset(train_conditions=train_conditions)

print(f"✅ Loaded training data: {X_train.shape}")
print(f"📂 Test conditions available: {list(test_conditions.keys())}")

# === Build and compile model ===
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=[SparseCategoricalAccuracy()]
)
model.summary()

# === Compute class weights ===
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# === Resume from checkpoint if needed ===
initial_epoch = 0
#latest_ckpt = tf.train.latest_checkpoint(config.CHECKPOINT_DIR)
#if latest_ckpt:
#    print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
#    model.load_weights(latest_ckpt)
#    initial_epoch = int(os.path.basename(latest_ckpt).split("_")[-1].replace(".keras", ""))

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

# === Prepare validation set ===
val_nm = test_conditions.get("nm")
val_bg = test_conditions.get("bg")
val_cl = test_conditions.get("cl")

if not (val_nm and val_bg and val_cl):
    raise ValueError("Missing one or more test conditions: expected nm, bg, cl.")

X_val = np.concatenate([val_nm[0], val_bg[0], val_cl[0]], axis=0)
y_val = np.concatenate([val_nm[1], val_bg[1], val_cl[1]], axis=0)
print(f"🧪 Validation set shape: {X_val.shape}, Labels: {y_val.shape}")

# === Train the model ===
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    initial_epoch=initial_epoch,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    shuffle=True
)

# === Final evaluation ===
model.evaluate(X_val, y_val)
