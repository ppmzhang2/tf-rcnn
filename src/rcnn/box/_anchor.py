"""NumPy implementation of anchors.

shape: [H_feature_map, W_feature_map, 9, 4]

format: (y_min, x_min, y_max, x_max)
"""
import numpy as np

from rcnn import cfg
from rcnn.box import bbox


def _scales(x: int) -> tuple[int, int, int]:
    """Get the scale sequence dynamically with closest (floor) power of 2.

    Example:
        >>> _scales(32)
        (8, 16, 32)
        >>> _scales(63)
        (8, 16, 32)
        >>> _scales(64)
        (16, 32, 64)

    Args:
        x (int): minimum shape value (width or height) of the input image

    Returns:
        tuple[int, int, int]: three scales from small to large
    """
    # closest (ceiling) power of 2
    scale_max = 2**(x).bit_length()
    return scale_max >> 3, scale_max >> 2, scale_max >> 1


def _scale_mat(x: int) -> np.ndarray:
    """Get the scale matrix for production.

    Args:
        x (int): minimum shape value (width or height) of the input image

    Returns:
        np.ndarray: scale matrix of shape (9, 3)
    """
    scale_min, scale_med, scale_max = _scales(x)
    return np.array(
        [
            [scale_min, 0, 0],
            [scale_med, 0, 0],
            [scale_max, 0, 0],
            [0, scale_min, 0],
            [0, scale_med, 0],
            [0, scale_max, 0],
            [0, 0, scale_min],
            [0, 0, scale_med],
            [0, 0, scale_max],
        ],
        dtype=np.float32,
    )


def _one_hw(x: int) -> np.ndarray:
    """Generate a single group (9) of anchors.

    Args:
        x (int): minimum shape value (width or height) of the input image

    Returns:
        np.ndarray: tensor (9, 2) of format (height, width)
    """
    sqrt2 = 1.4142135624
    ratio_hw = (
        (sqrt2, sqrt2 / 2),
        (1, 1),
        (sqrt2 / 2, sqrt2),
    )
    return np.matmul(_scale_mat(x), np.array(ratio_hw, dtype=np.float32))


def _hw(h: int, w: int, stride: int) -> np.ndarray:
    """Get (height, width) pair of the feature map.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32

    Returns:
        np.ndarray: tensor (H, W, 9, 2) of format (height, width)
    """
    size_min = min(w * stride, h * stride)
    raw_anchors_ = _one_hw(size_min)
    return np.tile(raw_anchors_[np.newaxis, np.newaxis, :], (h, w, 1, 1))


def _center_coord(h: int, w: int, stride: int) -> np.ndarray:
    """Center coordinates of each grid cell.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32

    Returns:
        np.ndarray: tensor (H, W, 9, 2) of format (y, x)
    """
    vx, vy = (
        np.arange(0, w, dtype=np.float32),
        np.arange(0, h, dtype=np.float32),
    )
    xs, ys = (
        vx[np.newaxis, :, np.newaxis],
        vy[:, np.newaxis, np.newaxis],
    )
    xss, yss = (
        np.tile(xs, (h, 1, cfg.N_ANCHOR)),
        np.tile(ys, (1, w, cfg.N_ANCHOR)),
    )
    # (H, W), NOT the other way around
    return np.stack([yss, xss], axis=-1) * stride + stride // 2


def grid_anchor(h: int, w: int, stride: int) -> np.ndarray:
    """Get anchors for each grid cell in absolute coordinates.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32

    Returns:
        np.ndarray: array (H, W, 9, 4) of format (y_min, x_min, y_max, x_max)
            in absolute coordinates
    """
    hw_half = 0.5 * _hw(h, w, stride)
    coords = _center_coord(h, w, stride)
    return np.concatenate([coords - hw_half, coords + hw_half], axis=-1)


def anchor_all() -> np.ndarray:
    """Get ALL flattened anchor tensor in relative coordinates.

    Returns:
        np.ndarray: anchor (N_ac, 4) of format (y_min, x_min, y_max, x_max)
    """
    _mat_trans = np.array([
        [1. / cfg.H, 0., 0., 0.],
        [0., 1. / cfg.W, 0., 0.],
        [0., 0., 1. / cfg.H, 0.],
        [0., 0., 0., 1. / cfg.W],
    ])  # matrix to convert absolute coordinates to relative ones
    ac_abs = np.reshape(
        grid_anchor(cfg.H_FM, cfg.W_FM, cfg.STRIDE),
        (-1, 4),
    )
    return np.matmul(ac_abs, _mat_trans)


# flattened relative anchors (N_ac, 4)
all_anchors = anchor_all()

# valid anchors mask based on image size of type np.float32 (N_ac,)
val_anchor_mask = np.where(
    (bbox.xmin(all_anchors) >= 0) & (bbox.ymin(all_anchors) >= 0)
    & (bbox.xmax(all_anchors) >= bbox.xmin(all_anchors))
    & (bbox.ymax(all_anchors) >= bbox.ymin(all_anchors))
    & (bbox.xmax(all_anchors) <= 1) & (bbox.ymax(all_anchors) <= 1),
    1.0,
    0.0,
)

# valid anchors (<N_ac, 4)
val_anchors = all_anchors[val_anchor_mask == 1]

# set as read-only
all_anchors.flags.writeable = False
val_anchors.flags.writeable = False
val_anchor_mask.flags.writeable = False
