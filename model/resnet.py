#code of resnet(this is not pretrained but i will train it for silhoutte as pretrained wants rgb but dataset has silhoutte)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def build_resnet_encoder(
    input_shape=(64, 64, 1),
    feature_dim=256,
    use_pretrained=False,
    normalize_input=True,
    add_batch_norm=True,
    dropout_rate=0.3
):
    """
    Builds a ResNet-based encoder for silhouette input.

    Args:
        input_shape (tuple): (H, W, C) e.g., (64, 64, 1)
        feature_dim (int): Size of final embedding vector
        use_pretrained (bool): Load ImageNet weights if True
        normalize_input (bool): Whether to normalize input to [-1, 1]
        add_batch_norm (bool): Whether to add BatchNorm after ResNet
        dropout_rate (float): Dropout rate after ResNet

    Returns:
        model (tf.keras.Model): Keras encoder model
    """

    def preprocess_silhouette(x):
        # Convert 1-channel grayscale to 3-channel RGB
        x = tf.repeat(x, repeats=3, axis=-1)
        if normalize_input:
            x = (x - 0.5) / 0.5  # Scale to [-1, 1]
        return x

    input_tensor = layers.Input(shape=input_shape, name="resnet_input")
    x = layers.Lambda(preprocess_silhouette, name="preprocess")(input_tensor)

    # Build ResNet50 (no top)
    resnet_base = ResNet50(
        include_top=False,
        weights='imagenet' if use_pretrained else None,
        input_tensor=x,
        pooling='avg'
    )

    # Set trainability
    resnet_base.trainable = not use_pretrained  # train if not pretrained

    x = resnet_base.output

    if add_batch_norm:
        x = layers.BatchNormalization(name="resnet_bn")(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="resnet_dropout")(x)

    x = layers.Dense(feature_dim, activation='sigmoid', name="resnet_embedding")(x)

    model = models.Model(inputs=input_tensor, outputs=x, name="resnet_encoder")
    return model
