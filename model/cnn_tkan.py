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


def build_tkan_block(input_shape, feature_dim, num_classes, dropout_rate):
    """
    Builds the temporal TKAN-based classification block.

    Args:
        input_shape: tuple, e.g. (time_steps, feature_dim)
        feature_dim: internal TKAN hidden size
        num_classes: output class count (e.g. CASIA-B identities)
        dropout_rate: dropout probability

    Returns:
        Keras Model that maps a sequence of vectors to a class score
    """
    inputs = Input(shape=input_shape, name="tkan_input")

    x = TKAN(
        feature_dim,
        sub_kan_configs=['bspline_0', 'bspline_1', 'bspline_2', 'bspline_3', 'bspline_4'],
        return_sequences=True,
        use_bias=True
    )(inputs)

    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(num_classes, activation='softmax', name="identity_output")(x)
    return models.Model(inputs, outputs, name="tkan_classifier")


def build_model():
    """
    Full end-to-end gait recognition model using CNN + TKAN.

    Returns:
        Compiled Keras Model.
    """
    sequence_input = Input(
        shape=(config.SEQUENCE_LEN, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="sequence_input"
    )

    # CNN applied per frame
    cnn_encoder = build_cnn_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )
    x = TimeDistributed(cnn_encoder)(sequence_input)  # shape: (batch, time, feature_dim)

    # TKAN block for temporal modeling
    tkan_block = build_tkan_block(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        feature_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    output = tkan_block(x)

    return models.Model(inputs=sequence_input, outputs=output, name="CNN_TKAN_GaitModel")
