"""Risk functions."""
import tensorflow as tf


def risk_class(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Risk function for classification."""
    mask = tf.greater(y_true, -1)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def risk_local(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Risk function for localization."""
    mask = tf.greater(y_true, -1)
    loss = tf.keras.losses.mse(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def risk_iou(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Risk function for IoU."""
    mask = tf.greater(y_true, -1)
    loss = 1 - iou(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def smooth_l1(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Smooth L1 loss."""
    mask = tf.greater(y_true, -1)
    loss = tf.keras.losses.huber(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
