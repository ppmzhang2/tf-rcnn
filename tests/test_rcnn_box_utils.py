"""Test utils of `rcnn.box`."""
import tensorflow as tf

from rcnn.box import anchor
from rcnn.box import bbox2delta
from rcnn.box import delta2bbox

EPS = 1e-3


def test_box_bbox_delta() -> None:
    """Test `bbox2delta` and `delta2bbox`."""
    anchors = anchor(3, 4, 16)
    diff = tf.random.uniform((3, 4, 9, 4))
    bx = delta2bbox(anchors, diff)  # anchors + diff = bx
    diff2 = bbox2delta(bx, anchors)  # bx - anchors = diff2
    assert tf.reduce_all(tf.abs(diff - diff2) < EPS)
