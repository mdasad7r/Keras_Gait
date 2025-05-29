import tensorflow as tf
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
