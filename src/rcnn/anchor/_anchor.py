"""NumPy implementation of anchors.

shape: [H_feature_map, W_feature_map, 9, 4]

format: (y_min, x_min, y_max, x_max)

- N_ALL_AC = H_FM * W_FM * N_ANCHOR; e.g. 2304 = 16 * 16 * 9
- N_VAL_AC = sum(MASK_RPNAC); e.g. 568
"""
import numpy as np

from rcnn import cfg

__all__ = [
    "MASK_RPNAC",
    "N_RPNAC",
    "RPNAC",
]


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


def get_abs_anchor(
    h: int,
    w: int,
    stride: int,
    *,
    flat: bool = True,
) -> np.ndarray:
    """Get anchors for ALL grid cells in **ABSOLUTE** coordinates.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32
        flat (bool, optional): flatten the output tensor. Defaults to True.

    Returns:
        np.ndarray: anchor in absolute coordinates (y_min, x_min, y_max, x_max)
            - if flat: (N_ALL_AC, 4)
            - else: (H_FM, W_FM, 9, 4)
    """
    hw_half = 0.5 * _hw(h, w, stride)
    coords = _center_coord(h, w, stride)
    ac = np.concatenate([coords - hw_half, coords + hw_half], axis=-1)
    if flat:
        return np.reshape(ac, (-1, 4))
    return ac


def get_rel_anchor(
    h: int,
    w: int,
    stride: int,
    *,
    flat: bool = True,
) -> np.ndarray:
    """Get anchors for ALL grid cells in **RELATIVE** coordinates.

    Args:
        h (int): image height in pixels
        w (int): image width in pixels
        stride (int): stride of the backbone e.g. 32
        flat (bool, optional): flatten the output tensor. Defaults to True.

    Returns:
        np.ndarray: anchor of format (y_min, x_min, y_max, x_max)
            - if flat: (N_ALL_AC, 4)
            - else: (H_FM, W_FM, 9, 4)
    """
    _mat_trans = np.array([
        [1. / h, 0., 0., 0.],
        [0., 1. / w, 0., 0.],
        [0., 0., 1. / h, 0.],
        [0., 0., 0., 1. / w],
    ])  # matrix to convert absolute coordinates to relative ones
    ac_abs = get_abs_anchor(h // stride, w // stride, stride, flat=flat)
    return np.matmul(ac_abs, _mat_trans)


# flattened relative anchors (N_ALL_AC, 4) including invalid ones
anchors_raw = get_rel_anchor(cfg.H, cfg.W, cfg.STRIDE, flat=True)

# valid anchors mask based on image size of type np.float32 (N_ALL_AC,)
MASK_RPNAC = np.where(
    (anchors_raw[..., 0] >= 0) &  # y_min >= 0
    (anchors_raw[..., 1] >= 0) &  # x_min >= 0
    (anchors_raw[..., 2] <= 1) &  # y_max <= 1
    (anchors_raw[..., 3] <= 1) &  # x_max <= 1
    (anchors_raw[..., 2] > anchors_raw[..., 0]) &  # y_max > y_min
    (anchors_raw[..., 3] > anchors_raw[..., 1]),  # x_max > x_min
    1.0,
    0.0,
)

# valid anchors (N_VAL_AC, 4)
RPNAC = anchors_raw[MASK_RPNAC == 1]

# number of valid anchors
N_RPNAC = int(MASK_RPNAC.sum())

# set as read-only
RPNAC.flags.writeable = False
MASK_RPNAC.flags.writeable = False
