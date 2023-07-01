"""Anchors for R-CNN.

shape: [H_feature_map, W_feature_map, 9, 4]

format: (y_min, x_min, y_max, x_max)
"""
import tensorflow as tf

from src.rcnn import cfg

SQRT2 = 1.4142135624
RATIO_HW = (
    (SQRT2, SQRT2 / 2),
    (1, 1),
    (SQRT2 / 2, SQRT2),
)


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


def _scale_mat(x: int) -> tf.Tensor:
    """Get the scale matrix for production.

    Args:
        x (int): minimum shape value (width or height) of the input image

    Returns:
        tf.Tensor: 9 by 3 tensor
    """
    scale_min, scale_med, scale_max = _scales(x)
    return tf.constant(
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
        dtype=tf.float32,
    )


def _one_hw(x: int) -> tf.Tensor:
    """Generate a single group (9) of anchors.

    Args:
        x (int): minimum shape value (width or height) of the input image

    Returns:
        tf.Tensor: 9 by 2 tensor of format (height, width)
    """
    return tf.matmul(_scale_mat(x), tf.constant(RATIO_HW, dtype=tf.float32))


def _hw(h: int, w: int, stride: int) -> tf.Tensor:
    """Get (height, width) pair of the feature map.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32

    Returns:
        tf.Tensor: anchor tensor (H, W, 9, 2) of format (height, width)
    """
    size_min = min(w * stride, h * stride)
    raw_anchors_ = _one_hw(size_min)
    return tf.tile(raw_anchors_[tf.newaxis, tf.newaxis, :], (h, w, 1, 1))


def _center_coord(h: int, w: int, stride: int) -> tf.Tensor:
    """Center coordinates of each grid cell.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32

    Returns:
        tf.Tensor: anchor tensor (H, W, 9, 2) of format (y, x)
    """
    vx, vy = (
        tf.range(0, w, dtype=tf.float32),
        tf.range(0, h, dtype=tf.float32),
    )
    xs, ys = (
        vx[tf.newaxis, :, tf.newaxis],
        vy[:, tf.newaxis, tf.newaxis],
    )
    xss, yss = (
        tf.tile(xs, (h, 1, cfg.N_ANCHOR)),
        tf.tile(ys, (1, w, cfg.N_ANCHOR)),
    )
    # (H, W), NOT the other way around
    return tf.stack([yss, xss], axis=-1) * stride + stride // 2


def anchor(h: int, w: int, stride: int) -> tf.Tensor:
    """Get anchors for each grid cell.

    Args:
        h (int): feature map height
        w (int): feature map width
        stride (int): stride of the backbone e.g. 32

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9, 4) of format
            (y_min, x_min, y_max, x_max) in absolute coordinates
    """
    hw_ = 0.5 * _hw(h, w, stride)
    coords = _center_coord(h, w, stride)
    return tf.concat([coords - hw_, coords + hw_], axis=-1)
