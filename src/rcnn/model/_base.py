"""Pre-trained base models."""
import tensorflow as tf


def get_vgg16(h: int, w: int) -> tf.keras.Model:
    """Get VGG16 model.

    Args:
        h (int): Height of input image.
        w (int): Width of input image.

    Returns:
        tf.keras.Model: pre-trained VGG16 model without top layers for feature
            extraction.
    """
    mdl = tf.keras.applications.vgg16.VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(h, w, 3),
    )
    # Freeze all layers
    for layer in mdl.layers:
        layer.trainable = False
    return mdl
