"""import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

from models.cnn_encoder import build_cnn_encoder
from models.tkan import build_tkan_block
from dataset.casia_dataset import load_casia_dataset_colab
from config import (
    IMAGE_SIZE, IMAGE_CHANNELS, SEQUENCE_LEN, FEATURE_DIM,
    TKAN_HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CHECKPOINT_DIR, LOG_DIR
)

import os

def build_full_model():
    """
    Combines CNN + TimeDistributed + TKAN + Softmax output.
    """
    input_shape = (SEQUENCE_LEN, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS)
    inputs = layers.Input(shape=input_shape, name="sequence_input")

    # Frame-level feature extraction
    cnn_encoder = build_cnn_encoder(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_CHANNELS), feature_dim=FEATURE_DIM)
    x = layers.TimeDistributed(cnn_encoder)(inputs)

    # TKAN sequence classification
    tkan_model = build_tkan_block(
        input_shape=(SEQUENCE_LEN, FEATURE_DIM),
        feature_dim=TKAN_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    )
    outputs = tkan_model(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_TKAN_GaitRecognizer")
    return model

def main():
    print("🔄 Loading dataset...")
    X_train, y_train, X_test, y_test = load_casia_dataset_colab()

    print(f"✅ Data loaded: {X_train.shape} train, {X_test.shape} test")

    print("🧠 Building model...")
    model = build_full_model()
    model.summary()

    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    earlystop_cb = callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
    tensorboard_cb = callbacks.TensorBoard(log_dir=LOG_DIR)

    print("🚀 Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_cb, earlystop_cb, tensorboard_cb],
        shuffle=True
    )

    print("✅ Training complete. Evaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"🧪 Test Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import (
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Optional: Enable mixed precision
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

# === Load configs and modules ===
import config
from model.cnn_encoder import build_cnn_encoder
from model.tkan import build_tkan_classifier
from dataset.casia_dataset import load_casia_dataset

# === Load filtered dataset ===
X_train, y_train, test_conditions = load_casia_dataset()
print(f"✅ Loaded train: {X_train.shape}, test conditions: {list(test_conditions.keys())}")

# === Build model ===
def build_model():
    input_shape = (
        config.SEQUENCE_LEN,
        config.IMAGE_HEIGHT,
        config.IMAGE_WIDTH,
        config.IMAGE_CHANNELS
    )
    sequence_input = tf.keras.Input(shape=input_shape, name="sequence_input")

    encoder = build_cnn_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )

    x = tf.keras.layers.TimeDistributed(encoder)(sequence_input)

    tkan_model = build_tkan_classifier(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        feature_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )

    output = tkan_model(x)
    return tf.keras.Model(inputs=sequence_input, outputs=output, name="CNN_TKAN_GaitRecognizer")

# === Compile model ===
model = build_model()
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=[SparseCategoricalAccuracy()]
)
model.summary()

# === Class weighting ===
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
else:
    print("🚨 Resetting: Starting training from scratch!")

# === Callbacks ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
steps_per_epoch = len(X_train) // config.BATCH_SIZE

callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(config.CHECKPOINT_DIR, "epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_freq=5 * steps_per_epoch
    ),
    TensorBoard(log_dir=os.path.join(config.LOG_DIR, timestamp)),
    CSVLogger(os.path.join(config.LOG_DIR, "training_log.csv"), append=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# === Build validation set (using test nm as val) ===
X_val, y_val = test_conditions["nm"]
print(f"🧪 Using {X_val.shape} NM sequences for validation")

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

# === Evaluate final model ===
model.evaluate(X_val, y_val)
