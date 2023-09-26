"""Region of Interest (RoI) Block."""
import tensorflow as tf

from rcnn import box

# Buffer to clip the RoIs. Defaults to 1e-1.
BUFFER = 1e-1
# valid anchors
AC_VAL = tf.constant(box.val_anchors, dtype=tf.float32)  # (N_val_ac, 4)


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
            shape: (B, N_val_ac, 1), (B, N_val_ac, 4), and (B, N_val_ac, 4).
            where N_val_ac <= H_FM * W_FM * 9 is the number of valid anchors.
    """
    # Get valid labels and deltas based on valid anchor masks.
    # shape: (B, N_val_ac, 1) and (B, N_val_ac, 4)
    log_val = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(
        x, box.val_anchor_mask == 1, axis=1))(rpn_log)
    dlt_val = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(
        x, box.val_anchor_mask == 1, axis=1))(rpn_dlt)
    # Computing YXYX RoIs from deltas.
    rois = box.delta2bbox(AC_VAL, dlt_val)  # (B, N_val_ac, 4)
    rois = tf.clip_by_value(rois, -BUFFER, 1. + BUFFER)  # (B, n_val_ac, 4)

    return log_val, dlt_val, rois
