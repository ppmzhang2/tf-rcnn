"""Region of Interest (RoI) Block."""
import tensorflow as tf

from rcnn import anchor

# Buffer to clip the RoIs. Defaults to 1e-1.
BUFFER = 1e-1
# valid anchors
AC_VAL = tf.constant(anchor.RPNAC, dtype=tf.float32)  # (N_VAL_AC, 4)


def roi(dlt: tf.Tensor) -> tf.Tensor:
    """Get RoI bounding boxes from anchor deltas.

    Args:
        dlt (tf.Tensor): RPN predicted deltas. Shape [B, N_VAL_AC, 4].

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: RPN classification, bounding
            box delta, and RoIs. All are filtered by valid anchor masks.
            Shape [B, N_VAL_AC, 1], [B, N_VAL_AC, 4], and [B, N_VAL_AC, 4].
    """
    bsize = tf.shape(dlt)[0]
    # Computing YXYX RoIs from deltas. output shape: (B, N_VAL_AC, 4)
    rois = anchor.delta2bbox(tf.repeat(AC_VAL[tf.newaxis, ...], bsize, axis=0),
                             dlt)
    # clip the RoIs
    rois = tf.clip_by_value(rois, -BUFFER, 1. + BUFFER)  # (B, N_VAL_AC, 4)

    return rois
