"""Region of Interest (RoI) Block."""
import tensorflow as tf

from rcnn import anchor

# Buffer to clip the RoIs. Defaults to 1e-1.
BUFFER = 1e-1
# valid anchors
AC_VAL = tf.constant(anchor.RPNAC, dtype=tf.float32)  # (N_VAL_AC, 4)
N_VAL_AC = anchor.N_RPNAC  # number of valid anchors, also the dim of axis=1


def roi(
    rpn_log: tf.Tensor,
    rpn_dlt: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Region of Interest (RoI) Block.

    Args:
        rpn_log (tf.Tensor): RPN predicted logits.
            Shape [B, H_FM * W_FM * 9, 1].
        rpn_dlt (tf.Tensor): RPN predicted bounding box delta.
            Shape [B, H_FM * W_FM * 9, 4].

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: RPN classification, bounding
            box delta, and RoIs. All are filtered by valid anchor masks.
            Shape [B, N_VAL_AC, 1], [B, N_VAL_AC, 4], and [B, N_VAL_AC, 4].
    """
    # Get valid labels and deltas based on valid anchor masks.
    # shape: (B, N_VAL_AC, 1) and (B, N_VAL_AC, 4)
    log_val = tf.boolean_mask(rpn_log, anchor.MASK_RPNAC == 1, axis=1)
    dlt_val = tf.boolean_mask(rpn_dlt, anchor.MASK_RPNAC == 1, axis=1)
    # Computing YXYX RoIs from deltas.
    rois = anchor.delta2bbox(AC_VAL, dlt_val)  # (B, N_VAL_AC, 4)
    rois = tf.clip_by_value(rois, -BUFFER, 1. + BUFFER)  # (B, N_VAL_AC, 4)

    return log_val, dlt_val, rois
