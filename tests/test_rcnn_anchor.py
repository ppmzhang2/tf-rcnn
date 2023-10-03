"""Test Faster R-CNN `box` module."""
import numpy as np
import tensorflow as tf

from rcnn.anchor import _box
from rcnn.anchor import bbox2delta
from rcnn.anchor import delta2bbox
from rcnn.anchor import get_abs_anchor

EPS = 1e-3


def test_anchor() -> None:
    """Test anchor generaters."""
    # pre-defined anchors of ratio 1:1
    h = 24
    w = 32
    stride = 32
    pre_anchor = get_abs_anchor(h, w, stride, flat=False)
    pre_anchor_11 = pre_anchor[:, :, 3:6, :]
    res = _box.xmax(pre_anchor_11) - _box.xmin(pre_anchor_11)
    exp = np.tile(
        np.array(
            [128, 256, 512],
            dtype=np.float32,
        )[np.newaxis, np.newaxis, :],
        (h, w, 1),
    )
    assert (res - exp).sum() < EPS


def test_anchor_utils() -> None:
    """Test `bbox2delta` and `delta2bbox` in `anchor._utils`."""
    anchors = get_abs_anchor(3, 4, 16, flat=False)
    diff = tf.random.uniform((3, 4, 9, 4))
    bx = delta2bbox(anchors, diff)  # anchors + diff = bx
    diff2 = bbox2delta(bx, anchors)  # bx - anchors = diff2
    assert tf.reduce_all(tf.abs(diff - diff2) < EPS)
