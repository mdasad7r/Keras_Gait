from tensorflow.keras import layers, models

def build_cnn_encoder(input_shape=(64, 64, 1), feature_dim=256):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2 (2 layers)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3 (2 layers)
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Final dense projection
    x = layers.Dense(feature_dim)(x)
    x = layers.Activation('sigmoid')(x)  # Final sigmoid (as requested)

    model = models.Model(inputs, x, name='cnn_encoder')
    return model
