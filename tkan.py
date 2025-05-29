from tensorflow.keras import layers, models
from tkan import TKAN

def build_tkan_block(input_shape, feature_dim=128, num_classes=74, dropout_rate=0.3):
    """
    Builds a TKAN-based temporal classification model for gait recognition.

    Args:
        input_shape: tuple. Shape of the input (time_steps, feature_dim).
        feature_dim: int. Hidden size for TKAN layers.
        num_classes: int. Number of identity classes for CASIA-B (default: 74).
        dropout_rate: float. Dropout rate to prevent overfitting.

    Returns:
        Keras Model instance.
    """
    inputs = layers.Input(shape=input_shape, name="tkan_input")

    # First TKAN layer (returns sequence)
    x = TKAN(feature_dim, sub_kan_configs=['relu'] * 5, return_sequences=True, use_bias=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)

    # Second TKAN layer (summarizes sequence)
    x = TKAN(feature_dim, sub_kan_configs=['relu'] * 5, return_sequences=False, use_bias=True)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax', name="identity_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="tkan_classifier")
    return model
