"""Test Faster R-CNN `box` module."""
import tensorflow as tf

from rcnn import abox


def test_abox_anchors() -> None:
    """Test `abox.anchors`."""
    # pre-defined anchors of ratio 1:1
    h = 24
    w = 32
    pre_anchor = abox.anchors(w, h, 32)
    pre_anchor_11 = pre_anchor[:, :, 3:6, :]
    res = abox.xmax(pre_anchor_11) - abox.xmin(pre_anchor_11)
    exp = tf.tile(
        tf.constant(
            [128, 256, 512],
            dtype=tf.float32,
        )[tf.newaxis, tf.newaxis, :],
        (h, w, 1),
    )
    assert tf.get_static_value(tf.reduce_sum(res - exp)) == 0


BBOX1 = (0.0, 0.4, 5.0, 6.4)
BBOX2 = (-0.4, 2.3, 5.6, 6.3)
LMT_UP = 0.5883
LMT_LOW = 0.5882

aboxes1 = tf.constant((BBOX1, BBOX1))
aboxes2 = tf.constant((BBOX2, BBOX2))


def test_bbox_iou() -> None:
    """Test bbox.iou."""
    res = abox.iou(aboxes1, aboxes2)
    assert tf.get_static_value(tf.reduce_all(res >= LMT_LOW))
    assert tf.get_static_value(tf.reduce_all(res <= LMT_UP))
