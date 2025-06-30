from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Dense, Dropout, TimeDistributed, GlobalAveragePooling1D
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tkan import TKAN
import config


def build_resnet_encoder(input_shape=(64, 64, 1), feature_dim=256):
    """
    Builds a ResNet-based encoder adapted for grayscale gait frames.

    Args:
        input_shape: shape of one input frame (H, W, C).
        feature_dim: output dimensionality of the encoder per frame.

    Returns:
        Keras Model that outputs a feature vector per frame.
    """
    inputs = Input(shape=input_shape)

    # Convert grayscale to 3-channel if needed
    if input_shape[-1] == 1:
        x = layers.Concatenate()([inputs, inputs, inputs])  # shape: (H, W, 3)
    else:
        x = inputs

    # Use ResNet50 as a deep encoder (you can swap in ResNet18 manually if needed)
    base_model = ResNet50(include_top=False, weights=None, input_tensor=x, pooling='avg')

    # Dense projection layer to match feature_dim
    x = base_model.output
    x = Dense(feature_dim, activation='relu')(x)

    return Model(inputs=inputs, outputs=x, name="resnet_encoder")


def build_tkan_block(input_shape, feature_dim, num_classes, dropout_rate):
    """
    TKAN-based temporal classifier block.

    Args:
        input_shape: (time_steps, feature_dim)
        feature_dim: TKAN hidden size
        num_classes: total identity classes
        dropout_rate: dropout for regularization

    Returns:
        A Keras Model.
    """
    inputs = Input(shape=input_shape, name="tkan_input")

    x = TKAN(
        feature_dim,
        sub_kan_configs=[0, 1, 2, 3, 4],
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
    Constructs the full ResNet-TKAN gait recognition model.

    Returns:
        A compiled Keras Model.
    """
    sequence_input = Input(
        shape=(config.SEQUENCE_LEN, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="sequence_input"
    )

    # Frame-level ResNet encoder
    resnet_encoder = build_resnet_encoder(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM
    )
    x = TimeDistributed(resnet_encoder)(sequence_input)  # shape: (batch, time, feature_dim)

    # TKAN block
    tkan_block = build_tkan_block(
        input_shape=(config.SEQUENCE_LEN, config.FEATURE_DIM),
        feature_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    output = tkan_block(x)

    return models.Model(inputs=sequence_input, outputs=output, name="ResNet_TKAN_GaitModel")
