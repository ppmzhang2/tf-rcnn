"""Tests for visualization."""
import cv2
import numpy as np
import tensorflow as tf
from _pytest.monkeypatch import MonkeyPatch

from rcnn import vis


def test_vis_draw_pred() -> None:
    """Test `vis.draw_pred` function."""
    # Prepare sample data
    img = tf.constant(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    bboxes = tf.constant([[0.1, 0.2, 0.6, 0.8]])
    labels = tf.constant([0])
    names = ["Object"]

    # Draw prediction on the image
    img_with_pred = vis.draw_pred(img, bboxes, labels, names)

    # Check types and shapes
    assert isinstance(img_with_pred, np.ndarray)
    assert img_with_pred.shape == (100, 100, 3)


def test_vis_draw_rois() -> None:
    """Test `vis.draw_rois` function."""
    # Prepare sample data
    img = tf.constant(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    rois = tf.constant([[0.1, 0.2, 0.6, 0.8]])

    # Draw ROIs on the image
    img_with_rois = vis.draw_rois(img, rois)

    # Check types and shapes
    assert isinstance(img_with_rois, np.ndarray)
    assert img_with_rois.shape == (100, 100, 3)


def test_vis_show_image(monkeypatch: MonkeyPatch) -> None:
    """Test `vis.show_image` function.

    This test ensures show_image doesn"t crash and works correctly.
    We will use monkeypatch to replace cv2.imshow and cv2.waitKey functions
    as we can"t have a real window in test environments.
    """
    # Mock imshow and waitKey
    monkeypatch.setattr(cv2, "imshow",
                        lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(cv2, "waitKey",
                        lambda *args, **kwargs: None)  # noqa: ARG005

    # Test show_image
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    vis.show_image(img)  # Should not crash
