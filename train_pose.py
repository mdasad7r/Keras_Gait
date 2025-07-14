import os
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D,
    Concatenate, TimeDistributed, Lambda
)
from keras.applications import ResNet50, ResNet101
from keras import layers
from tkan import TKAN
from config_pose import (
    TRAIN_PATH, TEST_PATH,
    SEQUENCE_LEN, POSE_FEATURE_DIM,
    CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    CHECKPOINT_DIR, LOG_DIR,
    FEATURE_DIM, TKAN_HIDDEN_DIM, NUM_CLASSES, DROPOUT_RATE
)
from dataset.casia_pose import load_casia_pose_dataset


# === Utility Functions ===
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


# === Model Building ===
def build_custom_cnn_encoder(input_shape=(8, 8, 1), feature_dim=FEATURE_DIM):
    print("🧱 Building Custom CNN Encoder...")
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(feature_dim, activation='relu')(x)
    return Model(inp, x, name="custom_cnn_encoder")

def build_spatial_encoder(base_model_class, input_shape=(8, 8, 3), projection_dim=FEATURE_DIM):
    print(f"🧱 Building Spatial Encoder for {base_model_class.__name__}...")
    inp = Input(shape=input_shape)
    x = Lambda(lambda x: tf.image.resize(x, (32, 32)))(inp)
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(projection_dim)(x)
    return Model(inp, x)

def build_model(sequence_len=SEQUENCE_LEN,
                cnn_input_shape=(CNN_HEIGHT, CNN_WIDTH, CNN_CHANNELS),
                resnet_input_shape=(CNN_HEIGHT, CNN_WIDTH, 3),
                feature_dim=FEATURE_DIM,
                tkan_hidden_dim=TKAN_HIDDEN_DIM,
                num_classes=NUM_CLASSES,
                dropout_rate=DROPOUT_RATE):

    print("🚧 Building Final Model Architecture...")
    seq_input = Input(shape=(sequence_len, *cnn_input_shape), name="input_sequence")
    resnet_input = Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1))(seq_input)

    custom_cnn = build_custom_cnn_encoder(input_shape=cnn_input_shape)
    resnet101 = build_spatial_encoder(ResNet101, input_shape=resnet_input_shape)
    resnet50 = build_spatial_encoder(ResNet50, input_shape=resnet_input_shape)

    print("🔁 Applying TimeDistributed...")
    cnn_out = TimeDistributed(custom_cnn)(seq_input)
    r101_out = TimeDistributed(resnet101)(resnet_input)
    r50_out = TimeDistributed(resnet50)(resnet_input)

    print("🔗 Concatenating outputs...")
    fused = Concatenate()([cnn_out, r101_out, r50_out])
    fused_proj = Dense(feature_dim)(fused)

    print("🧠 Passing through TKAN...")
    x = TKAN(
        units=tkan_hidden_dim,
        sub_kan_configs=[0, 1, 2, 3, 4],
        return_sequences=True,
        use_bias=True
    )(fused_proj)

    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax', name="identity_output")(x)

    print("✅ Model built.")
    return Model(seq_input, out, name="ensemble_pose_tkan_model")


# === Training Entry Point ===
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
    model = build_model()

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
