"""delta-box."""
import tensorflow as tf


def dx(delta: tf.Tensor) -> tf.Tensor:
    """Get delta x coordinate of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 0]


def dy(delta: tf.Tensor) -> tf.Tensor:
    """Get delta y coordinate of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 1]


def dw(delta: tf.Tensor) -> tf.Tensor:
    """Get delta width of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 2]


def dh(delta: tf.Tensor) -> tf.Tensor:
    """Get delta height of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 3]
