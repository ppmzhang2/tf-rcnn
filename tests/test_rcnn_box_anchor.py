"""Test Faster R-CNN `box` module."""
import tensorflow as tf

from rcnn.box import anchor
from rcnn.box import bbox

EPS = 1e-3


def test_box_anchor() -> None:
    """Test `rcnn.box.anchor`."""
    # pre-defined anchors of ratio 1:1
    h = 24
    w = 32
    pre_anchor = anchor(w, h, 32)
    pre_anchor_11 = pre_anchor[:, :, 3:6, :]
    res = bbox.xmax(pre_anchor_11) - bbox.xmin(pre_anchor_11)
    exp = tf.tile(
        tf.constant(
            [128, 256, 512],
            dtype=tf.float32,
        )[tf.newaxis, tf.newaxis, :],
        (h, w, 1),
    )
    assert tf.get_static_value(tf.reduce_sum(tf.math.abs(res - exp))) < EPS
