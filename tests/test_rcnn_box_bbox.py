"""Test Faster R-CNN `box` module."""
import pytest
import tensorflow as tf

from rcnn import box

EPS = 1e-3
BBOX1 = (0.0, 0.4, 5.0, 6.4)
BBOX2 = (-0.4, 2.3, 5.6, 6.3)
LMT_UP = 0.5883
LMT_LOW = 0.5882

bx1 = tf.constant((BBOX1, BBOX1))
bx2 = tf.constant((BBOX2, BBOX2))


def test_box_bbox_iou() -> None:
    """Test `bbox.iou`."""
    res = box.bbox.iou(bx1, bx2)
    assert tf.get_static_value(tf.reduce_all(res >= LMT_LOW))
    assert tf.get_static_value(tf.reduce_all(res <= LMT_UP))


def test_box_bbox_trans() -> None:
    """Test `bbox.from_xywh` and `bbox.to_xywh`."""
    xywh = box.bbox.to_xywh(bx1)
    exp = tf.constant(((3.4, 2.5, 6.0, 5.0), (3.4, 2.5, 6.0, 5.0)))
    assert tf.get_static_value(tf.reduce_all(tf.math.abs(xywh - exp) < EPS))


@pytest.mark.parametrize("bx", [
    tf.random.uniform((4, 3, 9, 4)),
    tf.random.uniform((15, 11, 9, 4)),
])
def test_box_bbox_trans_random(bx: tf.Tensor) -> None:
    """Test `bbox.from_xywh` and `bbox.to_xywh`."""
    xywh = box.bbox.to_xywh(bx)
    res = box.bbox.from_xywh(xywh)
    assert tf.get_static_value(tf.reduce_all(tf.math.abs(res - bx) < EPS))


def test_box_bbox_clip() -> None:
    """Test `bbox.clip`."""
    # Create a bounding box tensor
    bbox = tf.constant([
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
    output = box.bbox.clip(bbox, h, w)

    # Check if the output matches the expected output
    assert tf.reduce_all(tf.equal(output, expected_output))
