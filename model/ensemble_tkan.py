from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D,
    Concatenate, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201, ResNet50, Xception
from tkan import TKAN
import config


def build_spatial_encoder(base_model_class, input_shape=(64, 64, 1), projection_dim=256):
    inp = Input(shape=input_shape)
    x = base_model_class(include_top=False, weights=None, input_shape=input_shape)(inp)
    x = GlobalAveragePooling2D()(x)
    x = Dense(projection_dim)(x)
    return Model(inp, x)

def build_model(input_shape=(50, 64, 64, 1),
                feature_dim=256,
                tkan_hidden_dim=256,
                num_classes=124,
                dropout_rate=0.3):
    """
    Builds the full ensemble model: 3 CNNs + fusion + TKAN
    """

    # Input: sequence of silhouette frames
    seq_input = Input(shape=input_shape, name="input_sequence")

    # Build spatial encoders (grayscale support)
    densenet = build_spatial_encoder(DenseNet201)
    resnet   = build_spatial_encoder(ResNet50)
    xcept    = build_spatial_encoder(Xception)

    # Apply to sequence
    dense_out = TimeDistributed(densenet)(seq_input)  # (batch, T, 256)
    resnet_out = TimeDistributed(resnet)(seq_input)
    xcept_out = TimeDistributed(xcept)(seq_input)

    # Fuse
    fused = Concatenate()([dense_out, resnet_out, xcept_out])  # (batch, T, 768)
    fused_proj = Dense(feature_dim)(fused)  # (batch, T, 256)

    # TKAN
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

    return Model(seq_input, out, name="ensemble_tkan")


# Optional: expose entrypoint
def build():
    return build_model(
        input_shape=(config.SEQUENCE_LEN, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        feature_dim=config.FEATURE_DIM,
        tkan_hidden_dim=config.TKAN_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
