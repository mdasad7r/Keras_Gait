import os
import tensorflow as tf
from config_pose import *
from dataset.casia_pose import build_casia_pose_dataset
from model.pose_model_ensemble import build


class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.h5")
            self.model.save(path)
            print(f"💾 Model saved at: {path}")


def main():
    print("📦 Building lazy-loading datasets...")
    train_ds = build_casia_pose_dataset(TRAIN_PATH, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = build_casia_pose_dataset(TEST_PATH, batch_size=BATCH_SIZE, shuffle=False)

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
