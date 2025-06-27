import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger, Callback
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# === Load configs and model parts ===
import config
from model.cnn_tkan import build_model
from dataset.casia_dataset import load_casia_dataset

# === Load dataset (only training data: nm-01 to nm-04) ===
X_train, y_train, _ = load_casia_dataset()  # test_conditions ignored here
print(f"✅ Loaded training data: {X_train.shape}")

# === Create validation split from training data ===
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=42
)
print(f"🧪 Validation split created: X_val={X_val.shape}, y_val={y_val.shape}")

# === Build model ===
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

# === Resume from checkpoint (optional) ===
initial_epoch = 0
# latest_ckpt = tf.train.latest_checkpoint(config.CHECKPOINT_DIR)
# if latest_ckpt:
#     print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
#     model.load_weights(latest_ckpt)
#     initial_epoch = int(os.path.basename(latest_ckpt).split("_")[1])
# else:
#     print("🚨 No checkpoint found — starting fresh training.")

# === Custom callback to save every 10 epochs ===
#from tensorflow.keras.callbacks import Callback

class EpochModCallback(Callback):
    def __init__(self, checkpoint_callback, every_n_epochs=10):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.every_n_epochs = every_n_epochs

    def set_model(self, model):
        # Do not assign to self.model — just forward to the checkpoint
        self.checkpoint_callback.set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            self.checkpoint_callback.on_epoch_end(epoch, logs)


# === Callbacks ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "epoch_{epoch:02d}_val{val_loss:.4f}.keras")

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_freq='epoch',  # must be 'epoch' if using val_loss
    verbose=1
)

callbacks = [
    EpochModCallback(model_checkpoint, every_n_epochs=10),
    TensorBoard(log_dir=os.path.join(config.LOG_DIR, f"{timestamp}_tkan")),
    CSVLogger(os.path.join(config.LOG_DIR, "training_log.csv"), append=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# === Train ===
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

# === Final evaluation on validation set ===
print("\n✅ Final evaluation on internal validation split:")
model.evaluate(X_val, y_val)
