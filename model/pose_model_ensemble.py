from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D,
    Concatenate, TimeDistributed, Lambda
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
import tensorflow as tf
from tkan import TKAN
import config

# === Optional: Replace with your ResNet18 source ===
from keras_resnet.models import ResNet18


# === Custom CNN encoder ===
def build_custom_cnn_encoder(input_shape=(8, 8, 1), feature_dim=config.FEATURE_DIM):
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


# === Pretrained ResNet encoder ===
def build_spatial_encoder(base_model_class, input_shape=(8, 8, 3), projection_dim=config.FEATURE_DIM):
    inp = Input(shape=input_shape)
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(projection_dim)(x)
    return Model(inp, x)


# === Final model ===
def build_model(sequence_len=config.SEQUENCE_LEN,
                pose_feature_dim=config.POSE_FEATURE_DIM,
                cnn_input_shape=(8, 8, 1),
                resnet_input_shape=(8, 8, 3),
                feature_dim=config.FEATURE_DIM,
                tkan_hidden_dim=config.TKAN_HIDDEN_DIM,
                num_classes=config.NUM_CLASSES,
                dropout_rate=config.DROPOUT_RATE):
    """
    Model: Custom CNN + ResNet18 + ResNet50 -> TKAN
    Input shape: (T, 8, 8, 1)
    """

    # Input: (T, 8, 8, 1) — per-frame grayscale pseudo-image
    seq_input = Input(shape=(sequence_len, *cnn_input_shape), name="input_sequence")

    # Expand to 3 channels for ResNet
    resnet_input = Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1))(seq_input)

    # === Encoders ===
    custom_cnn = build_custom_cnn_encoder(input_shape=cnn_input_shape)
    resnet18 = build_spatial_encoder(ResNet18, input_shape=resnet_input_shape)
    resnet50 = build_spatial_encoder(ResNet50, input_shape=resnet_input_shape)

    # === Apply TimeDistributed ===
    cnn_out = TimeDistributed(custom_cnn)(seq_input)
    r18_out = TimeDistributed(resnet18)(resnet_input)
    r50_out = TimeDistributed(resnet50)(resnet_input)

    # === Concatenate encoder outputs ===
    fused = Concatenate()([cnn_out, r18_out, r50_out])  # shape: (T, 768)
    fused_proj = Dense(feature_dim)(fused)              # reduce to (T, 256)

    # === TKAN ===
    x = TKAN(
        output_dim=tkan_hidden_dim,
        sub_kan_configs=[0, 1, 2, 3, 4],
        return_sequences=True,
        use_bias=True
    )(fused_proj)

    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax', name="identity_output")(x)

    return Model(seq_input, out, name="ensemble_pose_tkan_model")


# Entrypoint for training
def build():
    return build_model()
