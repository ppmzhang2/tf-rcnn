"""Region of Interest (RoI) Block."""
import tensorflow as tf

from rcnn import anchor

# valid anchors
AC_VAL = tf.constant(anchor.RPNAC, dtype=tf.float32)  # (N_VAL_AC, 4)


def roi(dlt: tf.Tensor, buffer: float = 1e-1) -> tf.Tensor:
    """Get RoI bounding boxes from anchor deltas.

    Args:
        dlt (tf.Tensor): RPN predicted deltas. Shape [B, N_VAL_AC, 4].
        buffer (float, optional): buffer to clip the RoIs. Defaults to 1e-1.

    Returns:
        tf.Tensor: clipped RoIs in shape [B, N_VAL_AC, 4].
    """
    bsize = tf.shape(dlt)[0]
    # Computing YXYX RoIs from deltas. output shape: (B, N_VAL_AC, 4)
    rois = anchor.delta2bbox(tf.repeat(AC_VAL[tf.newaxis, ...], bsize, axis=0),
                             dlt)
    # clip the RoIs
    return tf.clip_by_value(rois, -buffer, 1. + buffer)  # (B, N_VAL_AC, 4)
