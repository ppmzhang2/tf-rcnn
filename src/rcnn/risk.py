"""Risk functions."""
import tensorflow as tf

from rcnn import cfg


def risk_rpn_reg(
    bx_prd: tf.Tensor,
    bx_tgt: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """Delta / RoI regression risk for RPN with smooth L1 loss.

    `bx` values cannot be `inf` or `-inf` otherwise the loss will be `nan`.

    Args:
        bx_prd (tf.Tensor): predicted box (N_ac, 4); could be predicted deltas
            or RoIs
        bx_tgt (tf.Tensor): target box (N_ac, 4); could be target deltas or
            ground truth boxes
        mask (tf.Tensor): 0/1 object mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    fn_huber = tf.keras.losses.Huber(reduction="none")
    prd = tf.boolean_mask(bx_prd, mask)  # (N_obj, 4)
    tgt = tf.boolean_mask(bx_tgt, mask)  # (N_obj, 4)
    loss = fn_huber(tgt, prd)  # (N_obj,)
    # print(prd[:5])  # normal
    # print(tgt[:5])  # dw, dh = -inf
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + cfg.EPS)


def risk_rpn_obj(logits: tf.Tensor, mask_obj: tf.Tensor) -> tf.Tensor:
    """Objectness classification risk for RPN with binary cross entropy loss.

    Args:
        logits (tf.Tensor): predicted logits (N_ac, 1)
        mask_obj (tf.Tensor): 0/1 object mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction="none",
    )
    prd = tf.boolean_mask(logits, mask_obj)  # (N_obj, 1)
    tgt = tf.ones_like(prd)
    loss = bce(tgt, prd)  # (N_obj,)
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask_obj) + cfg.EPS)


def risk_rpn_bkg(logits: tf.Tensor, mask_bkg: tf.Tensor) -> tf.Tensor:
    """Background classification risk for RPN with binary cross entropy loss.

    Args:
        logits (tf.Tensor): predicted logits (N_ac, 1)
        mask_bkg (tf.Tensor): 0/1 background mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction="none",
    )
    prd = tf.boolean_mask(logits, mask_bkg)  # (N_bkg, 1)
    tgt = tf.zeros_like(prd)  # (N_bkg, 1)
    loss = bce(tgt, prd)  # (N_bkg,)
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask_bkg) + cfg.EPS)


@tf.function
def risk_rpn(
    bx_roi: tf.Tensor,
    bx_tgt: tf.Tensor,
    logits: tf.Tensor,
    mask_obj: tf.Tensor,
    mask_bkg: tf.Tensor,
) -> tf.Tensor:
    """RPN loss.

    Args:
        bx_roi (tf.Tensor): RoI boxes (B, N_ac, 4)
        bx_tgt (tf.Tensor): ground truth boxes matching RoIs (B, N_ac, 4)
        logits (tf.Tensor): predicted logits (B, N_ac, 1)
        mask_obj (tf.Tensor): 0/1 object mask (B, N_ac)
        mask_bkg (tf.Tensor): 0/1 background mask (B, N_ac)

    Returns:
        tf.Tensor: risk (scalar)
    """
    loss_reg = tf.map_fn(
        lambda x: risk_rpn_reg(x[0], x[1], x[2]),
        (bx_roi, bx_tgt, mask_obj),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )  # (B,)
    loss_obj = tf.map_fn(
        lambda x: risk_rpn_obj(x[0], x[1]),
        (logits, mask_obj),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )  # (B,)
    loss_bkg = tf.map_fn(
        lambda x: risk_rpn_bkg(x[0], x[1]),
        (logits, mask_bkg),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )  # (B,)
    return tf.reduce_mean(loss_reg + loss_obj + loss_bkg)
