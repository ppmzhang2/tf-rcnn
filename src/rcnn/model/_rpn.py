"""Region Proposal Network (RPN)."""
import tensorflow as tf

from rcnn import cfg

init_he = tf.keras.initializers.he_normal()
reg_l2 = tf.keras.regularizers.l2(0.0005)


def rpn(x: tf.Tensor) -> tf.Tensor:
    """Region Proposal Network (RPN).

    Args:
        x (tf.Tensor): The input tensor, usually the output of a backbone
            network.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: predicted deltas and labels.
            - deltas: (n_batch, N_ANCHOR * H_FM * W_FM, 4)
            - labels: (n_batch, N_ANCHOR * H_FM * W_FM, 1)
    """

    def flatten_dlt(x: tf.Tensor) -> tf.Tensor:
        """Flatten the tensor to (n_batch, N_ANCHOR * H_FM * W_FM, 4)."""
        n_batch = tf.shape(x)[0]
        return tf.reshape(x, (n_batch, -1, 4))

    def flatten_lbl(x: tf.Tensor) -> tf.Tensor:
        """Flatten the tensor to (n_batch, N_ANCHOR * H_FM * W_FM, 1)."""
        n_batch = tf.shape(x)[0]
        return tf.reshape(x, (n_batch, -1, 1))

    # feature map shared by the two heads
    fm = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        kernel_initializer=init_he,
        padding="same",
        activation="relu",
        name="rpn_conv",
    )(x)

    # predicted deltas
    dlt = tf.keras.layers.Conv2D(
        cfg.N_ANCHOR * 4,
        (1, 1),
        kernel_initializer=init_he,
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_dlt",
    )(fm)

    # predicted labels
    lbl = tf.keras.layers.Conv2D(
        cfg.N_ANCHOR,
        (1, 1),
        kernel_initializer=init_he,
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_cls",
    )(fm)

    # flatten the tensors
    # use the Lambda layer as dynmaic shape is not supported in tf.reshape
    dlt_flat = tf.keras.layers.Lambda(flatten_dlt, name="flatten_dlt")(dlt)
    lbl_flat = tf.keras.layers.Lambda(flatten_lbl, name="flatten_cls")(lbl)
    return dlt_flat, lbl_flat
