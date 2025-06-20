import tensorflow as tf
from tensorflow.keras import layers, models

def build_their_cnn(input_shape=(64, 64, 1), feature_dim=256):
    """
    Builds the CNN described in the 'Gait Recognition Using CNN' paper.
    This CNN is used to extract spatial features from individual silhouette frames.

    Args:
        input_shape (tuple): Shape of input frame, e.g. (64, 64, 1)
        feature_dim (int): Dimension of final embedding vector

    Returns:
        tf.keras.Model: CNN model that maps a frame to a feature vector
    """
    inputs = layers.Input(shape=input_shape, name="frame_input")

    # Conv Layer 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Conv Layer 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Conv Layer 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and dense to get feature vector
    x = layers.Flatten()(x)
    x = layers.Dense(feature_dim, activation='relu', name="cnn_embedding")(x)

    model = models.Model(inputs=inputs, outputs=x, name="their_cnn_encoder")
    return model
