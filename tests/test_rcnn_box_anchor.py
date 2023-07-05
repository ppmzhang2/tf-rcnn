"""Test Faster R-CNN `box` module."""
import numpy as np

from rcnn.box import bbox
from rcnn.box import grid_anchor

EPS = 1e-3


def test_box_anchor() -> None:
    """Test `rcnn.box.anchor`."""
    # pre-defined anchors of ratio 1:1
    h = 24
    w = 32
    stride = 32
    pre_anchor = grid_anchor(h, w, stride)
    pre_anchor_11 = pre_anchor[:, :, 3:6, :]
    res = bbox.xmax(pre_anchor_11) - bbox.xmin(pre_anchor_11)
    exp = np.tile(
        np.array(
            [128, 256, 512],
            dtype=np.float32,
        )[np.newaxis, np.newaxis, :],
        (h, w, 1),
    )
    assert (res - exp).sum() < EPS
