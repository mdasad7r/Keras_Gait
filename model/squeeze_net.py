import tensorflow as tf
from tensorflow.keras import layers, models

def fire_module(x, squeeze_channels, expand_channels, name):
    squeeze = layers.Conv2D(squeeze_channels, (1, 1), activation='relu', padding='same', name=f"{name}_squeeze")(x)
    expand1x1 = layers.Conv2D(expand_channels, (1, 1), activation='relu', padding='same', name=f"{name}_expand1x1")(squeeze)
    expand3x3 = layers.Conv2D(expand_channels, (3, 3), activation='relu', padding='same', name=f"{name}_expand3x3")(squeeze)
    return layers.Concatenate(name=f"{name}_concat")([expand1x1, expand3x3])

def build_squeezenet_encoder(input_shape=(64, 64, 1), feature_dim=256):
    """
    SqueezeNet-style encoder for silhouette-based gait input.

    Args:
        input_shape (tuple): Input image shape (H, W, C), e.g., (64, 64, 1)
        feature_dim (int): Output embedding size (default: 256)

    Returns:
        tf.keras.Model: Keras encoder model
    """

    input_tensor = layers.Input(shape=input_shape, name="squeeze_input")

    # Convert grayscale to 3 channels
    x = layers.Lambda(lambda x: tf.repeat(x, repeats=3, axis=-1), name="repeat_channels")(input_tensor)
    x = layers.Rescaling(1.0 / 255.0, name="normalize")(x)

    # Stem
    x = layers.Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu', name="conv1")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="maxpool1")(x)

    # Fire modules
    x = fire_module(x, 16, 64, name="fire2")
    x = fire_module(x, 16, 64, name="fire3")
    x = fire_module(x, 32, 128, name="fire4")
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="maxpool4")(x)

    x = fire_module(x, 32, 128, name="fire5")
    x = fire_module(x, 48, 192, name="fire6")
    x = fire_module(x, 48, 192, name="fire7")
    x = fire_module(x, 64, 256, name="fire8")
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Output feature vector
    x = layers.Dense(feature_dim, activation='sigmoid', name="squeeze_embedding")(x)

    model = models.Model(inputs=input_tensor, outputs=x, name="squeezenet_encoder")
    return model
