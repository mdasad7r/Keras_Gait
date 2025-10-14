import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Switch to TensorFlow backend to avoid JAX XLA issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN for consistency

import tensorflow as tf
tf.config.optimizer.set_jit(False)  # Disable XLA/JIT compilation to fix transpose_grad error

from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, TimeDistributed, GlobalAveragePooling1D
)
from tkan import TKAN
import config

def build_cnn_encoder(input_shape=(64, 64, 1), feature_dim=256):
    """
    Builds a CNN encoder to process a single gait frame.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(feature_dim, activation='relu')(x)
    model = models.Model(inputs, x, name="cnn_encoder")
    return model

def build_tkan_block(input_shape, feature_dim, num_classes, dropout_rate):
    """
    Builds the optimized temporal TKAN-based classification block.
    Optimizations:
    - Single TKAN layer with simplified sub_kan_configs (2 splines, orders 3-4, grid_size 10-12) to reduce complexity and avoid XLA issues.
    - BatchNormalization for input stability.
    - Light L2 regularization (0.001).
    - Single Dropout after pooling.
    - Matches feature_dim to 256 for CNN consistency.

    Args:
        input_shape: tuple, e.g., (50, 256)
        feature_dim: internal TKAN hidden size (256)
        num_classes: 124 for CASIA-B
        dropout_rate: 0.1-0.2

    Returns:
        Keras Model
    """
    inputs = Input(shape=input_shape, name="tkan_input")
    x = BatchNormalization()(inputs)
    x = TKAN(
        feature_dim,
        sub_kan_configs=[
            {'spline_order': 3, 'grid_size': 10},
            {'spline_order': 4, 'grid_size': 12}
        ],
        return_sequences=True,
        use_bias=True,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax', name="identity_output")(x)
    return models.Model(inputs, outputs, name="tkan_classifier")

def build_model():
    """
    Full end-to-end gait recognition model using CNN + Optimized TKAN.
    """
    sequence_input = Input(
        shape=(config.SEQUENCE_LEN, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="sequence_input"
    )
    cnn_encoder = build_cnn_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )
    x = TimeDistributed(cnn_encoder)(sequence_input)  # shape: (batch, 50, 256)
    x = BatchNormalization()(x)
    tkan_block = build_tkan_block(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        feature_dim=config.FEATURE_DIM,  # 256
        num_classes=config.NUM_CLASSES,  # 124
        dropout_rate=config.DROPOUT_RATE
    )
    output = tkan_block(x)
    return models.Model(inputs=sequence_input, outputs=output, name="CNN_TKAN_GaitModel")
