import os
import numpy as np
import tensorflow as tf
from datetime import datetime

from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras import mixed_precision

from resnet import build_resnet_encoder
from tkan import build_tkan_classifier
import config  # <-- your config file

# === Enable mixed precision training ===
mixed_precision.set_global_policy("mixed_float16")

# === Load data ===
def load_data(path):
    X = np.load(os.path.join(path, "X.npy"))
    y = np.load(os.path.join(path, "y.npy"))
    return X, y

X_train, y_train = load_data(config.TRAIN_PATH)
X_test, y_test = load_data(config.TEST_PATH)

print(f"✅ Loaded train: {X_train.shape}, test: {X_test.shape}")

# === Build model ===
def build_model():
    input_shape = (
        config.SEQUENCE_LEN,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    )

    sequence_input = tf.keras.Input(shape=input_shape, name="sequence_input")

    encoder = build_resnet_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM,
        use_pretrained=False  # ← training from scratch for silhouettes
    )

    x = tf.keras.layers.TimeDistributed(encoder, name="time_distributed")(sequence_input)
    output = build_tkan_classifier(x, num_classes=config.NUM_CLASSES)

    model = tf.keras.Model(inputs=sequence_input, outputs=output, name="Resnet_TKAN_GaitRecognizer")
    return model

model = build_model()
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=[SparseCategoricalAccuracy()]
)
model.summary()

# === Compute class weights ===
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# === Checkpoint resume ===
initial_epoch = 0
latest_ckpt = tf.train.latest_checkpoint(config.CHECKPOINT_DIR)
if latest_ckpt:
    print(f"🔄 Resuming from {latest_ckpt}")
    model.load_weights(latest_ckpt)
    initial_epoch = int(os.path.basename(latest_ckpt).split("_")[-1].replace(".keras", ""))

# === Callbacks ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
steps_per_epoch = len(X_train) // config.BATCH_SIZE

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(config.CHECKPOINT_DIR, "epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq=5 * steps_per_epoch  # ✅ save every 5 epochs
    ),
    TensorBoard(log_dir=os.path.join(config.LOG_DIR, timestamp)),
    CSVLogger(os.path.join(config.LOG_DIR, "training_log.csv"), append=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# === Train ===
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    initial_epoch=initial_epoch,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    shuffle=True
)
