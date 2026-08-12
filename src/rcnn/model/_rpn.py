"""Region Proposal Network (RPN)."""
import tensorflow as tf

from rcnn import anchor
from rcnn import cfg

SEED_INIT = 42
MASK_AC = tf.constant(anchor.MASK_RPNAC, dtype=tf.float32)  # (N_ANCHOR,)
N_VAL_AC = anchor.N_RPNAC  # number of valid anchors, also the dim of axis=1

reg_l2 = tf.keras.regularizers.l2(0.0005)
# Derive three unique seeds from the initial seed
seed_cnn_fm = hash(SEED_INIT) % (2**32)
seed_cnn_dlt = hash(seed_cnn_fm) % (2**32)
seed_cnn_lbl = hash(seed_cnn_dlt) % (2**32)


def rpn(
    fm: tf.Tensor,
    h_fm: int,
    w_fm: int,
) -> tuple[tf.Tensor, tf.Tensor, list[tf.keras.layers.Layer]]:
    """Region Proposal Network (RPN).

    Args:
        fm (tf.Tensor): feature map from the backbone.
        h_fm (int): The height of the feature map.
        w_fm (int): The width of the feature map.

    Returns:
        tuple[tf.Tensor, tf.Tensor, list[tf.keras.layers.Layer]]: predicted
        deltas, labels, and the RPN layers for weight freezing.
            - deltas: (n_batch, N_VAL_AC, 4)
            - labels: (n_batch, N_VAL_AC, 1)
    """
    # TBD: residual connections?
    layer_drop_entr = tf.keras.layers.Dropout(
        cfg.R_DROP,
        name="rpn_dropout_entr",
    )  # entrance dropout
    layer_conv_share = tf.keras.layers.Conv2D(
        cfg.SIZE_IMG,
        (3, 3),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_fm),
        kernel_regularizer=reg_l2,
        padding="same",
        activation=None,
        name="rpn_share",
    )  # shared convolutional layer
    layer_gn_share = tf.keras.layers.GroupNormalization()
    layer_relu_share = tf.keras.layers.Activation("relu")
    layer_drop_share = tf.keras.layers.Dropout(cfg.R_DROP)

    # (n_batch, h_fm, w_fm, SIZE_IMG)
    fm = layer_drop_entr(fm)
    fm = layer_conv_share(fm)
    fm = layer_gn_share(fm)
    fm = layer_relu_share(fm)
    fm = layer_drop_share(fm)

    # deltas
    layer_conv_dlt = tf.keras.layers.Conv2D(
        cfg.N_ANCHOR * 4,
        (1, 1),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_dlt),
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_dlt",
    )
    layer_bn_dlt = tf.keras.layers.BatchNormalization()
    layer_drop_dlt = tf.keras.layers.Dropout(cfg.R_DROP)
    # (n_batch, h_fm, w_fm, N_ANCHOR * 4)
    dlt = layer_conv_dlt(fm)
    dlt = layer_bn_dlt(dlt)
    dlt = layer_drop_dlt(dlt)

    # logits objectness score
    layer_conv_log = tf.keras.layers.Conv2D(
        cfg.N_ANCHOR,
        (1, 1),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_lbl),
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_log",
    )
    layer_bn_log = tf.keras.layers.BatchNormalization()
    layer_drop_log = tf.keras.layers.Dropout(cfg.R_DROP)
    # (n_batch, h_fm, w_fm, N_ANCHOR)
    log = layer_conv_log(fm)
    log = layer_bn_log(log)
    log = layer_drop_log(log)

    # flatten the tensors
    # shape: (B, H_FM * W_FM * N_ANCHOR, 4) and (B, H_FM * W_FM * N_ANCHOR, 1)
    dlt_flat = tf.reshape(dlt, (-1, h_fm * w_fm * cfg.N_ANCHOR, 4))
    log_flat = tf.reshape(log, (-1, h_fm * w_fm * cfg.N_ANCHOR, 1))

    # Get valid labels and deltas based on valid anchor masks.
    # shape: (B, N_VAL_AC, 4) and (B, N_VAL_AC, 1)
    dlt_val = tf.boolean_mask(dlt_flat, MASK_AC == 1, axis=1)
    log_val = tf.boolean_mask(log_flat, MASK_AC == 1, axis=1)

    # for shape inference
    return (
        tf.reshape(dlt_val, (-1, N_VAL_AC, 4)),
        tf.reshape(log_val, (-1, N_VAL_AC, 1)),
        [
            layer_drop_entr,
            layer_conv_share,
            layer_gn_share,
            layer_relu_share,
            layer_drop_share,
            layer_conv_dlt,
            layer_bn_dlt,
            layer_drop_dlt,
            layer_conv_log,
            layer_bn_log,
            layer_drop_log,
        ],
    )
