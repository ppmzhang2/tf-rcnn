"""Region Proposal Network (RPN)."""
import tensorflow as tf

from rcnn import cfg

SEED_INIT = 42

reg_l2 = tf.keras.regularizers.l2(0.0005)
# Derive three unique seeds from the initial seed
seed_cnn_fm = hash(SEED_INIT) % (2**32)
seed_cnn_dlt = hash(seed_cnn_fm) % (2**32)
seed_cnn_lbl = hash(seed_cnn_dlt) % (2**32)


def rpn(fm: tf.Tensor, h_fm: int, w_fm: int) -> tf.Tensor:
    """Region Proposal Network (RPN).

    Args:
        fm (tf.Tensor): feature map from the backbone.
        h_fm (int): The height of the feature map.
        w_fm (int): The width of the feature map.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: predicted deltas and labels.
            - deltas: (n_batch, N_ANCHOR * H_FM * W_FM, 4)
            - labels: (n_batch, N_ANCHOR * H_FM * W_FM, 1)
    """
    # processed feature map (n_batch, h_fm, w_fm, 512)
    fm = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_fm),
        padding="same",
        activation="relu",
        name="rpn_conv",
    )(fm)

    # predicted deltas
    dlt = tf.keras.layers.Conv2D(
        cfg.N_ANCHOR * 4,
        (1, 1),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_dlt),
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_dlt",
    )(fm)  # (n_batch, h_fm, w_fm, N_ANCHOR * 4)

    # predicted labels
    lbl = tf.keras.layers.Conv2D(
        cfg.N_ANCHOR,
        (1, 1),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_lbl),
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_cls",
    )(fm)  # (n_batch, h_fm, w_fm, N_ANCHOR)

    # flatten the tensors
    # cannot simply use tf.reshape because the shape is dynamic
    dlt_flat = tf.reshape(dlt, (-1, h_fm * w_fm * cfg.N_ANCHOR, 4))
    lbl_flat = tf.reshape(lbl, (-1, h_fm * w_fm * cfg.N_ANCHOR, 1))
    return dlt_flat, lbl_flat
