from tensorflow.keras import layers, models

def build_cnn_encoder(input_shape=(64, 64, 1), feature_dim=256):
    """
    Builds a CNN to extract features from silhouette frames.
    
    Args:
        input_shape: Tuple. Shape of input silhouette (H, W, C).
        feature_dim: Int. Size of the output feature vector.

    Returns:
        Keras Model.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(feature_dim, activation='relu')(x)

    model = models.Model(inputs, x, name='cnn_encoder')
    return model
  
