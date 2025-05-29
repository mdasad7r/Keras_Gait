from tensorflow.keras import layers, models
from tkan import TKAN

def build_tkan_block(input_shape, feature_dim=128, num_classes=None):
    """
    Builds a TKAN-based temporal model.

    Args:
        input_shape: Tuple. Shape of input tensor (time_steps, feature_dim).
        feature_dim: Int. Hidden size for TKAN layers.
        num_classes: Optional. If given, adds classification head.

    Returns:
        Keras Model.
    """
    inputs = layers.Input(shape=input_shape)

    # TKAN temporal modeling
    x = TKAN(feature_dim, sub_kan_configs=['relu'] * 5, return_sequences=True)(inputs)
    x = layers.Dropout(0.3)(x)
    x = TKAN(feature_dim, sub_kan_configs=['relu'] * 5, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    if num_classes is not None:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        outputs = x  # Feature vector output

    model = models.Model(inputs, outputs, name='tkan_block')
    return model
