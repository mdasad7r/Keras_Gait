from tensorflow.keras import layers, models
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling1D
from tkan import TKAN  # Make sure the TKAN package is installed: pip install tkan

# ========================
# CNN Encoder
# ========================
def build_cnn_encoder(input_shape=(64, 64, 1), feature_dim=128):
    inputs = layers.Input(shape=input_shape)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Final Dense projection
    x = layers.Dense(feature_dim, activation='relu')(x)

    return models.Model(inputs, x, name='cnn_encoder')

# ========================
# Full Model: CNN + TKAN
# ========================
def build_gait_model(
    time_steps=30, 
    frame_shape=(64, 64, 1), 
    feature_dim=128, 
    num_classes=74, 
    dropout_rate=0.3
):
    """
    Constructs the full gait recognition model using CNN + TKAN.
    """

    # Input: sequence of frames
    video_input = layers.Input(shape=(time_steps, *frame_shape), name="video_input")

    # CNN per frame
    cnn_encoder = build_cnn_encoder(input_shape=frame_shape, feature_dim=feature_dim)
    x = TimeDistributed(cnn_encoder)(video_input)  # shape: (batch, time_steps, feature_dim)

    # TKAN Temporal modeling
    x = TKAN(
        feature_dim,
        sub_kan_configs=['bspline_0', 'bspline_1', 'bspline_2', 'bspline_3', 'bspline_4'],
        return_sequences=True,
        use_bias=True
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Temporal pooling (preserve full sequence info)
    x = GlobalAveragePooling1D()(x)

    # Final classifier
    x = layers.Dropout(dropout_rate)(x)
    identity_output = layers.Dense(num_classes, activation='softmax', name="identity_output")(x)

    model = models.Model(inputs=video_input, outputs=identity_output, name="Gait_TKAN_Model")
    return model
