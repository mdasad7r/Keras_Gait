from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, TimeDistributed, LSTM, GlobalAveragePooling1D
)
import config

def build_cnn_encoder(input_shape=(64, 64, 1), feature_dim=256):
    """
    Builds a CNN encoder to process a single gait frame.

    Args:
        input_shape: shape of one input frame (H, W, C).
        feature_dim: output dimensionality of the CNN per frame.

    Returns:
        Keras Model that outputs a feature vector per frame.
    """
    inputs = Input(shape=input_shape)

    # Conv Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # Conv Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)

    # Final dense projection
    x = Dense(feature_dim, activation='relu')(x)

    model = models.Model(inputs, x, name="cnn_encoder")
    return model

def build_lstm_block(input_shape, hidden_dim, num_classes, dropout_rate):
    """
    Builds the temporal LSTM-based classification block.

    Args:
        input_shape: tuple, e.g., (time_steps, feature_dim)
        hidden_dim: LSTM hidden state dimensionality
        num_classes: output class count (e.g., CASIA-B identities)
        dropout_rate: dropout probability

    Returns:
        Keras Model that maps a sequence of vectors to a class score
    """
    inputs = Input(shape=input_shape, name="lstm_input")

    # LSTM layers for temporal modeling
    x = LSTM(hidden_dim, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(hidden_dim // 2)(x)  # Second LSTM to reduce dimensionality
    x = Dropout(dropout_rate)(x)

    # Final classification
    outputs = Dense(num_classes, activation='softmax', name="identity_output")(x)

    return models.Model(inputs, outputs, name="lstm_classifier")

def build_model():
    """
    Full end-to-end gait recognition model using CNN + LSTM.

    Returns:
        Keras Model.
    """
    sequence_input = Input(
        shape=(config.SEQUENCE_LEN, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="sequence_input"
    )

    # CNN encoder
    cnn_encoder = build_cnn_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )
    cnn_features = TimeDistributed(cnn_encoder)(sequence_input)  # Shape: (batch, 50, 256)

    # LSTM block for temporal modeling
    lstm_block = build_lstm_block(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        hidden_dim=config.TKAN_HIDDEN_DIM,  # Reusing TKAN_HIDDEN_DIM for consistency
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    output = lstm_block(cnn_features)

    return models.Model(inputs=sequence_input, outputs=output, name="CNN_LSTM_GaitModel")
