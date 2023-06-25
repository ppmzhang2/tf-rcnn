"""Utils of module box."""
import tensorflow as tf

from . import bbox
from . import delta
from ._const import EPSILON


def delta2bbox(base: tf.Tensor, diff: tf.Tensor) -> tf.Tensor:
    """Apply delta to anchors to get bbox.

    e.g.: anchor (base) + delta (diff) = predicted (bbox)

    Args:
        base (tf.Tensor): base bbox tensor of shape (H, W, 9, 4)
        diff (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: bbox tensor of shape (H, W, 9, C), where C >= 4
    """
    xctr_ = bbox.xctr(base) + bbox.w(base) * delta.dx(diff)
    yctr_ = bbox.yctr(base) + bbox.h(base) * delta.dy(diff)
    w_ = bbox.w(base) * tf.exp(delta.dw(diff))
    h_ = bbox.h(base) * tf.exp(delta.dh(diff))
    xywh_ = tf.stack([xctr_, yctr_, w_, h_], axis=-1)
    return bbox.from_xywh(tf.concat([xywh_, bbox.rem(base)], axis=-1))


def bbox2delta(bbox_l: tf.Tensor, bbox_r: tf.Tensor) -> tf.Tensor:
    """Compute delta between two bounding boxes.

    e.g.:
        - GT (bbox_l) - anchor (bbox_r) = RPN target (delta)
        - GT (bbox_l) - ROI (bbox_r) = RCNN target (delta)

    Args:
        bbox_l (tf.Tensor): minuend bbox tensor (left operand) of shape
            (H, W, 9, C), where C >= 4
        bbox_r (tf.Tensor): subtrahend bbox tensor (right operand) of shape
            (H, W, 9, C), where C >= 4

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9, 4)
    """
    xctr_r = bbox.xctr(bbox_r)
    yctr_r = bbox.yctr(bbox_r)
    w_r = tf.math.maximum(bbox.w(bbox_r), EPSILON)
    h_r = tf.math.maximum(bbox.h(bbox_r), EPSILON)
    xctr_l = bbox.xctr(bbox_l)
    yctr_l = bbox.yctr(bbox_l)
    w_l = bbox.w(bbox_l)
    h_l = bbox.h(bbox_l)
    return tf.stack(
        [
            (xctr_l - xctr_r) / w_r,
            (yctr_l - yctr_r) / h_r,
            tf.math.log(w_l / w_r),
            tf.math.log(h_l / h_r),
        ],
        axis=-1,
    )
