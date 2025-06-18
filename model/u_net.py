import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return x

def encoder_block(x, filters, name):
    f = conv_block(x, filters, name)
    p = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(f)
    return f, p

def build_unet_encoder(input_shape=(64, 64, 1), feature_dim=256):
    """
    U-Net style encoder for silhouette spatial features.

    Args:
        input_shape (tuple): e.g. (64, 64, 1)
        feature_dim (int): Output feature size per frame

    Returns:
        tf.keras.Model: encoder model
    """

    input_tensor = layers.Input(shape=input_shape, name="unet_input")

    # 1. Expand to 3 channels (optional)
    x = layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), name="repeat_channels")(input_tensor)
    x = layers.Rescaling(1.0 / 255.0, name="normalize")(x)

    # 2. Encoder
    f1, p1 = encoder_block(x, 32, "enc1")
    f2, p2 = encoder_block(p1, 64, "enc2")
    f3, p3 = encoder_block(p2, 128, "enc3")

    # Bottleneck
    bn = conv_block(p3, 256, "bottleneck")

    # Optional: decoder (can be removed if not used)
    # f4 = layers.UpSampling2D((2, 2))(bn)
    # f4 = layers.Concatenate()([f4, f3])
    # f4 = conv_block(f4, 128, "dec3")

    # Global feature
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(bn)
    x = layers.Dense(feature_dim, activation='sigmoid', name="unet_embedding")(x)

    model = models.Model(inputs=input_tensor, outputs=x, name="unet_encoder")
    return model
