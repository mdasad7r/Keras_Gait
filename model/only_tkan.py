from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling1D
)
from tkan import TKAN
import config


def build_pose_tkan_model(input_shape=(config.SEQUENCE_LEN, config.POSE_FEATURE_DIM),
                          tkan_hidden_dim=config.TKAN_HIDDEN_DIM,
                          num_classes=config.NUM_CLASSES,
                          dropout_rate=config.DROPOUT_RATE):
    """
    Build TKAN-based model for pose sequence classification.
    Input: sequence of (T, F) pose features.
    Output: softmax classification over gait identities.
    """

    seq_input = Input(shape=input_shape, name="input_sequence")

    x = TKAN(
        output_dim=tkan_hidden_dim,
        sub_kan_configs=[0, 1, 2, 3, 4],
        return_sequences=True,
        use_bias=True
    )(seq_input)

    x = Dropout(dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    out = Dense(num_classes, activation='softmax', name="identity_output")(x)

    model = Model(seq_input, out, name="pose_tkan_model")
    return model


# Optional: default entrypoint
def build():
    return build_pose_tkan_model()
