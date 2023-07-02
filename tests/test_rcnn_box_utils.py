"""Test utils of `rcnn.box`."""
import tensorflow as tf

from src.rcnn.box import abs_anchors
from src.rcnn.box import bbox2delta
from src.rcnn.box import delta2bbox

EPS = 1e-3


def test_box_bbox_delta() -> None:
    """Test `bbox2delta` and `delta2bbox`."""
    anchors = abs_anchors(3, 4, 16)
    diff = tf.random.uniform((3, 4, 9, 4))
    bx = delta2bbox(anchors, diff)  # anchors + diff = bx
    diff2 = bbox2delta(bx, anchors)  # bx - anchors = diff2
    assert tf.reduce_all(tf.abs(diff - diff2) < EPS)
