import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from model.ensemble_tkan import build
from casia_dataset_protocol import load_dataset
import config

# === Strategy (optional for Colab TPU/GPU) ===
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
model = build()
model.summary()

# === Load Dataset ===
print("🔄 Loading dataset ...")
X_train, y_train, X_test, y_test = load_dataset(
    train_path=config.TRAIN_PATH,
    test_path=config.TEST_PATH,
    sequence_len=config.SEQUENCE_LEN,
    image_size=config.IMAGE_SIZE
)

print(f"✅ Train shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"✅ Test shape : {X_test.shape}, Labels: {y_test.shape}")

# === Compute Class Weights ===
classes = list(range(config.NUM_CLASSES))
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# === Compile Model ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# === Callbacks ===
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(config.CHECKPOINT_DIR, 'ensemble_tkan_best.h5'),
    monitor='val_sparse_categorical_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

tensorboard_cb = TensorBoard(log_dir=config.LOG_DIR)

# === Train ===
print("🚀 Starting training ...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_cb, tensorboard_cb],
    verbose=1
)

# === Evaluate ===
print("✅ Evaluating best model ...")
model.load_weights(os.path.join(config.CHECKPOINT_DIR, 'ensemble_tkan_best.h5'))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Final Test Accuracy: {test_acc:.4f}")
