from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D,
    Concatenate, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201, ResNet50
from tkan import TKAN
import config

# Your custom CNN encoder (copied from cnn_encoder.py)
def build_custom_cnn_encoder(input_shape=(71, 71, 1), feature_dim=256):
    inp = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(feature_dim, activation='relu')(x)

    return Model(inp, x, name="custom_cnn_encoder")


def build_spatial_encoder(base_model_class, input_shape=(71, 71, 1), projection_dim=256):
    inp = Input(shape=input_shape)
    base = base_model_class(include_top=False, weights=None, input_shape=input_shape)(inp)
    x = GlobalAveragePooling2D()(base)
    x = Dense(projection_dim)(x)
    return Model(inp, x)


def build_model(input_shape=(50, 71, 71, 1),
                feature_dim=256,
                tkan_hidden_dim=256,
                num_classes=124,
                dropout_rate=0.3):
    """
    Builds the ensemble model: DenseNet + ResNet + Custom CNN + TKAN
    """

    seq_input = Input(shape=input_shape, name="input_sequence")

    # Spatial encoders
    densenet_encoder = build_spatial_encoder(DenseNet201)
    resnet_encoder   = build_spatial_encoder(ResNet50)
    custom_encoder   = build_custom_cnn_encoder()

    # TimeDistributed encoders
    dense_out  = TimeDistributed(densenet_encoder)(seq_input)
    resnet_out = TimeDistributed(resnet_encoder)(seq_input)
    custom_out = TimeDistributed(custom_encoder)(seq_input)

    # Concatenate outputs
    fused = Concatenate()([dense_out, resnet_out, custom_out])  # shape: (batch, T, 768)
    fused_proj = Dense(feature_dim)(fused)  # Project to consistent dim

    # TKAN temporal modeling
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

    return Model(seq_input, out, name="Ensemble_Dense_Res_Custom_TKAN")


# Build function used in training script
def build():
    return build_model(
        input_shape=(config.SEQUENCE_LEN, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM,
        tkan_hidden_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
