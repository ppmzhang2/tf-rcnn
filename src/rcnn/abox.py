"""Manipulate anchor boxes.

shape: [..., 9, 5]

format: (x_min, y_min, x_max, y_max, objectness score)
"""
import numpy as np
import tensorflow as tf

EPSILON = 1e-3
N_ANCHOR = 9
SQRT2 = 1.4142135624
RATIO_HW = (
    (SQRT2, SQRT2 / 2),
    (1, 1),
    (SQRT2 / 2, SQRT2),
)


def _scales(x: int) -> tuple[int, int, int]:
    """Get the scale sequence dynamically with closest (floor) power of 2.

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


def _seq_wh(x: int) -> np.ndarray:
    """Generate a sequence of anchor width and height.

    Args:
        x (int): minimum shape value (width or height) of the input image

    Returns:
        np.ndarray: 9 by 2 tensor of format (width, height)
    """
    return tf.matmul(_scale_mat(x), tf.constant(RATIO_HW, dtype=tf.float32))


def _wh(w: int, h: int, stride: int) -> tf.Tensor:
    """Get fixed anchor size for each grid cell.

    Args:
        w (int): feature map width
        h (int): feature map height
        stride (int): stride of the backbone e.g. 32

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9, 2)
    """
    size_min = min(w * stride, h * stride)
    raw_anchors_ = _seq_wh(size_min)
    return tf.tile(raw_anchors_[tf.newaxis, tf.newaxis, :], (h, w, 1, 1))


def _center_coord(w: int, h: int, stride: int) -> tf.Tensor:
    """Center coordinates of each grid cell.

    Args:
        w (int): feature map width
        h (int): feature map height
        stride (int): stride of the backbone e.g. 32

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9, 2)
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
        tf.tile(xs, (h, 1, N_ANCHOR)),
        tf.tile(ys, (1, w, N_ANCHOR)),
    )
    # (H, W), NOT the other way around
    return tf.stack([xss, yss], axis=-1) * stride + stride // 2


def anchors(w: int, h: int, stride: int) -> tf.Tensor:
    """Get anchors for each grid cell.

    Args:
        w (int): feature map width
        h (int): feature map height
        stride (int): stride of the backbone e.g. 32

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9, 4)
    """
    hw_ = 0.5 * _wh(w, h, stride)
    coords = _center_coord(w, h, stride)
    return tf.concat([coords - hw_, coords + hw_], axis=-1)


def xmin(abox: tf.Tensor) -> tf.Tensor:
    """Get top-left x coordinate of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 0]


def ymin(abox: tf.Tensor) -> tf.Tensor:
    """Get top-left y coordinate of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 1]


def xmax(abox: tf.Tensor) -> tf.Tensor:
    """Get bottom-right x coordinate of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 2]


def ymax(abox: tf.Tensor) -> tf.Tensor:
    """Get bottom-right y coordinate of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 3]


def score(abox: tf.Tensor) -> tf.Tensor:
    """Get objectness score of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 4]


def width(abox: tf.Tensor) -> tf.Tensor:
    """Get width of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 2] - abox[..., 0]


def height(abox: tf.Tensor) -> tf.Tensor:
    """Get height of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return abox[..., 3] - abox[..., 1]


def x(abox: tf.Tensor) -> tf.Tensor:
    """Get center x coordinate of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return xmin(abox) + 0.5 * width(abox)


def y(abox: tf.Tensor) -> tf.Tensor:
    """Get center y coordinate of each anchor box.

    Returns:
        tf.Tensor: anchor tensor of shape (H, W, 9)
    """
    return ymin(abox) + 0.5 * height(abox)


def area(abox: tf.Tensor) -> tf.Tensor:
    """Get area of the anchor box."""
    return height(abox) * width(abox)


def pmax(abox: tf.Tensor) -> tf.Tensor:
    """Get bottom-right point from each anchor box."""
    return abox[..., 2:4]


def pmin(abox: tf.Tensor) -> tf.Tensor:
    """Get top-left point from each anchor box."""
    return abox[..., 0:2]


def confnd(abox: tf.Tensor) -> tf.Tensor:
    """Get object confidence from an anchor box (un-squeezed).

    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        abox (tf.Tensor): anchor box
    """
    return abox[..., 4:5]


def conf1d(abox: tf.Tensor) -> tf.Tensor:
    """Get object confidence from an anchor box (squeezed).

    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        abox (tf.Tensor): anchor box
    """
    return abox[..., 4]


def interarea(abox_pred: tf.Tensor, abox_label: tf.Tensor) -> tf.Tensor:
    """Get intersection area of two sets of anchor boxes.

    Returns:
        tf.Tensor: intersection area tensor with the same shape of the input
        only without the last rank
    """
    left_ups = tf.maximum(pmin(abox_pred), pmin(abox_label))
    right_downs = tf.minimum(pmax(abox_pred), pmax(abox_label))

    inter = tf.maximum(right_downs - left_ups, 0.0)
    return tf.multiply(inter[..., 0], inter[..., 1])


def iou(abox_pred: tf.Tensor, abox_label: tf.Tensor) -> tf.Tensor:
    """Calculate IoU of two bounding boxes."""
    area_pred = area(abox_pred)
    area_label = area(abox_label)
    area_inter = interarea(abox_pred, abox_label)
    area_union = area_pred + area_label - area_inter
    return (area_inter + EPSILON) / (area_union + EPSILON)
