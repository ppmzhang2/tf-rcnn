"""Test the `anchor._box` module."""
import pytest
import tensorflow as tf

from rcnn.anchor import _box

EPS = 1e-3
BX1 = (0.0, 0.4, 5.0, 6.4)
BX2 = (-0.4, 2.3, 5.6, 6.3)
LMT_UP = 0.5883
LMT_LOW = 0.5882

bx1 = tf.constant((BX1, BX1))
bx2 = tf.constant((BX2, BX2))


def test_anchor_box_iou() -> None:
    """Test `_box.iou`."""
    res = _box.iou(bx1, bx2)
    assert tf.get_static_value(tf.reduce_all(res >= LMT_LOW))
    assert tf.get_static_value(tf.reduce_all(res <= LMT_UP))


def test_anchor_box_trans() -> None:
    """Test `_box.from_xywh` and `_box.to_xywh`."""
    xywh = _box.to_xywh(bx1)
    exp = tf.constant(((3.4, 2.5, 6.0, 5.0), (3.4, 2.5, 6.0, 5.0)))
    assert tf.get_static_value(tf.reduce_all(tf.math.abs(xywh - exp) < EPS))


@pytest.mark.parametrize("bx", [
    tf.random.uniform((4, 3, 9, 4)),
    tf.random.uniform((15, 11, 9, 4)),
])
def test_anchor_box_trans_random(bx: tf.Tensor) -> None:
    """Test `_box.from_xywh` and `_box.to_xywh`."""
    xywh = _box.to_xywh(bx)
    res = _box.from_xywh(xywh)
    assert tf.get_static_value(tf.reduce_all(tf.math.abs(res - bx) < EPS))


def test_anchor_box_clip() -> None:
    """Test `_box.clip`."""
    # Create a bounding _box tensor
    bx = tf.constant([
        [10.0, 10.0, 30.0, 30.0, 0.6],  # within
        [-10.0, -10.0, 30.0, 30.0, 0.8],  # partly outside (top-left)
        [10.0, 10.0, 130.0, 130.0, 0.9],  # partly outside (bottom-right)
        [-10.0, -10.0, 130.0, 130.0, 0.7],  # completely outside
    ])

    # Set image dimensions
    h, w = 100.0, 100.0

    # Expected output
    expected_output = tf.constant([
        [10.0, 10.0, 30.0, 30.0, 0.6],
        [0.0, 0.0, 30.0, 30.0, 0.8],
        [10.0, 10.0, 100.0, 100.0, 0.9],
        [0.0, 0.0, 100.0, 100.0, 0.7],
    ])

    # Call clip function
    output = _box.clip(bx, h, w)

    # Check if the output matches the expected output
    assert tf.reduce_all(tf.equal(output, expected_output))
