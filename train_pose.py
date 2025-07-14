import os
import tensorflow as tf
from config_pose import *
from dataset.casia_pose import load_casia_pose_dataset
from model.pose_model_ensemble import build


def preprocess_for_model(X):
    X = X.astype('float32') / 255.0
    return tf.convert_to_tensor(X)

def prepare_dataset(X, y, batch_size=BATCH_SIZE, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.h5")
            self.model.save(path)
            print(f"💾 Model saved at: {path}")

def main():
    print("📦 Loading dataset...")
    X_train, y_train = load_casia_pose_dataset(TRAIN_PATH)
    X_test, y_test = load_casia_pose_dataset(TEST_PATH)

    print("🧼 Preprocessing...")
    X_train = preprocess_for_model(X_train)
    X_test = preprocess_for_model(X_test)

    print("📊 Creating tf.data.Dataset...")
    train_ds = prepare_dataset(X_train, y_train)
    test_ds = prepare_dataset(X_test, y_test, shuffle=False)

    print("🏗 Building model...")
    model = build()

    print("🧪 Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🚀 Starting training...")
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[
            EpochSaver(),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
        ]
    )

    print("✅ Evaluating final model...")
    loss, acc = model.evaluate(test_ds)
    print(f"🎯 Final Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
