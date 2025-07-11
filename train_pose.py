import tensorflow as tf
import numpy as np
import os

from config_pose import (
    TRAIN_PATH, TEST_PATH,
    SEQUENCE_LEN, POSE_FEATURE_DIM,
    CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CHECKPOINT_DIR, LOG_DIR
)

from dataset.casia_pose import load_casia_pose_dataset
from model.pose_model_ensemble import build


def preprocess_for_model(X):
    """
    Normalize and convert to tensor.
    X shape: (N, T, H, W, 1)
    """
    X = X.astype('float32') / 255.0  # Normalize pose values
    return tf.convert_to_tensor(X)


def prepare_dataset(X, y, shuffle=True):
    """
    Converts numpy arrays to tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


# === Custom Callback to Save Every 10 Epochs ===
class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            filename = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.h5")
            self.model.save(filename)
            print(f"✅ Saved model at {filename}")


def main():
    print("=== Loading Dataset ===")
    X_train, y_train = load_casia_pose_dataset(TRAIN_PATH)
    X_test, y_test = load_casia_pose_dataset(TEST_PATH)

    X_train = preprocess_for_model(X_train)  # (N, T, 8, 8, 1)
    X_test = preprocess_for_model(X_test)

    train_ds = prepare_dataset(X_train, y_train, shuffle=True)
    test_ds = prepare_dataset(X_test, y_test, shuffle=False)

    print("=== Building Model ===")
    model = build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # === Callbacks ===
    checkpoint_cb = EpochSaver()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    print("=== Training ===")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, tensorboard_cb, earlystop_cb]
    )

    print("=== Evaluating Final Model ===")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"✅ Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
