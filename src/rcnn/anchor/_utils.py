"""Transformations between bounding boxes and deltas."""
import tensorflow as tf

from rcnn import cfg
from rcnn.anchor import _box
from rcnn.anchor import _delta


def delta2bbox(base: tf.Tensor, diff: tf.Tensor) -> tf.Tensor:
    """Apply delta to anchors to get bbox.

    e.g.: anchor (base) + delta (diff) = predicted (bbox)

    Args:
        base (tf.Tensor): base bbox tensor of shape (N1, N2, ..., Nk, 4)
        diff (tf.Tensor): delta tensor of shape (N1, N2, ..., Nk, 4)

    Returns:
        tf.Tensor: bbox tensor of shape (N1, N2, ..., Nk, 4)
    """
    xctr_ = _box.xctr(base) + _box.w(base) * _delta.dx(diff)
    yctr_ = _box.yctr(base) + _box.h(base) * _delta.dy(diff)
    w_ = _box.w(base) * tf.exp(_delta.dw(diff))
    h_ = _box.h(base) * tf.exp(_delta.dh(diff))
    xywh_ = tf.stack([xctr_, yctr_, w_, h_], axis=-1)
    return _box.from_xywh(xywh_)


def bbox2delta(bbox_l: tf.Tensor, bbox_r: tf.Tensor) -> tf.Tensor:
    """Compute delta between two bounding boxes.

    e.g.:
        - GT (bbox_l) - anchor (bbox_r) = RPN target (delta)
        - GT (bbox_l) - ROI (bbox_r) = RCNN target (delta)

    Args:
        bbox_l (tf.Tensor): minuend bbox tensor (left operand) of shape
            (N1, N2, ..., Nk, C), where C >= 4, or other broadcastable shape
        bbox_r (tf.Tensor): subtrahend bbox tensor (right operand) of shape
            (N1, N2, ..., Nk, C), where C >= 4, or other broadcastable shape

    Returns:
        tf.Tensor: delta tensor of shape (N1, N2, ..., Nk, 4)
    """
    xctr_r = _box.xctr(bbox_r)
    yctr_r = _box.yctr(bbox_r)
    w_r = tf.math.maximum(_box.w(bbox_r), cfg.EPS)
    h_r = tf.math.maximum(_box.h(bbox_r), cfg.EPS)
    xctr_l = _box.xctr(bbox_l)
    yctr_l = _box.yctr(bbox_l)
    w_l = _box.w(bbox_l)
    h_l = _box.h(bbox_l)
    bx_del = tf.stack(
        [
            (xctr_l - xctr_r) / w_r,
            (yctr_l - yctr_r) / h_r,
            tf.math.log(w_l / w_r),
            tf.math.log(h_l / h_r),
        ],
        axis=-1,
    )
    return tf.clip_by_value(bx_del, -999.0, 999.0)
