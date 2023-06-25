"""Manipulate YXYX bounding boxes.

shape: [..., 9, C], where C >= 4

format: (y_min, x_min, y_max, x_max, objectness score, ...)
"""
import tensorflow as tf

from ._const import EPSILON


def xmin(bbox: tf.Tensor) -> tf.Tensor:
    """Get top-left x coordinate of each anchor box.

    Returns:
        tf.Tensor: X-min tensor of shape (H, W, 9)
    """
    return bbox[..., 1]


def ymin(bbox: tf.Tensor) -> tf.Tensor:
    """Get top-left y coordinate of each anchor box.

    Returns:
        tf.Tensor: Y-min tensor of shape (H, W, 9)
    """
    return bbox[..., 0]


def xmax(bbox: tf.Tensor) -> tf.Tensor:
    """Get bottom-right x coordinate of each anchor box.

    Returns:
        tf.Tensor: X-max tensor of shape (H, W, 9)
    """
    return bbox[..., 3]


def ymax(bbox: tf.Tensor) -> tf.Tensor:
    """Get bottom-right y coordinate of each anchor box.

    Returns:
        tf.Tensor: Y-max tensor of shape (H, W, 9)
    """
    return bbox[..., 2]


def rem(bbox: tf.Tensor) -> tf.Tensor:
    """Get remainders (excluding YXYX) of each anchor box.

    Returns:
        tf.Tensor: remainder tensor of shape (H, W, 9, C - 4)
    """
    return bbox[..., 4:]


def w(bbox: tf.Tensor) -> tf.Tensor:
    """Get width of each anchor box.

    Returns:
        tf.Tensor: width tensor of shape (H, W, 9)
    """
    return xmax(bbox) - xmin(bbox)


def h(bbox: tf.Tensor) -> tf.Tensor:
    """Get height of each anchor box.

    Returns:
        tf.Tensor: height tensor of shape (H, W, 9)
    """
    return ymax(bbox) - ymin(bbox)


def xctr(bbox: tf.Tensor) -> tf.Tensor:
    """Get center x coordinate of each anchor box.

    Returns:
        tf.Tensor: X-center tensor of shape (H, W, 9)
    """
    return xmin(bbox) + 0.5 * w(bbox)


def yctr(bbox: tf.Tensor) -> tf.Tensor:
    """Get center y coordinate of each anchor box.

    Returns:
        tf.Tensor: Y-center tensor of shape (H, W, 9)
    """
    return ymin(bbox) + 0.5 * h(bbox)


def area(bbox: tf.Tensor) -> tf.Tensor:
    """Get area of the anchor box.

    Returns:
        tf.Tensor: area tensor of shape (H, W, 9)
    """
    return h(bbox) * w(bbox)


def pmax(bbox: tf.Tensor) -> tf.Tensor:
    """Get bottom-right point from each anchor box.

    Returns:
        tf.Tensor: YX-max tensor of shape (H, W, 9, 2)
    """
    return bbox[..., 2:4]


def pmin(bbox: tf.Tensor) -> tf.Tensor:
    """Get top-left point from each anchor box.

    Returns:
        tf.Tensor: YX-min tensor of shape (H, W, 9, 2)
    """
    return bbox[..., 0:2]


def confnd(bbox: tf.Tensor) -> tf.Tensor:
    """Get object confidence from an anchor box (un-squeezed).

    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        bbox (tf.Tensor): anchor box

    Returns:
        tf.Tensor: object confidence tensor of shape (H, W, 9, 1)
    """
    return bbox[..., 4:5]


def conf1d(bbox: tf.Tensor) -> tf.Tensor:
    """Get object confidence from an anchor box (squeezed).

    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        bbox (tf.Tensor): anchor box

    Returns:
        tf.Tensor: object confidence tensor of shape (H, W, 9)
    """
    return bbox[..., 4]


def interarea(bbox_prd: tf.Tensor, bbox_lbl: tf.Tensor) -> tf.Tensor:
    """Get intersection area of two sets of anchor boxes.

    Returns:
        tf.Tensor: intersection area tensor of shape (H, W, 9)
    """
    left_ups = tf.maximum(pmin(bbox_prd), pmin(bbox_lbl))
    right_downs = tf.minimum(pmax(bbox_prd), pmax(bbox_lbl))

    inter = tf.maximum(right_downs - left_ups, 0.0)
    return tf.multiply(inter[..., 0], inter[..., 1])


def iou(bbox_prd: tf.Tensor, bbox_lbl: tf.Tensor) -> tf.Tensor:
    """Calculate IoU of two bounding boxes.

    Returns:
        tf.Tensor: IoU tensor of shape (H, W, 9)
    """
    area_pred = area(bbox_prd)
    area_label = area(bbox_lbl)
    area_inter = interarea(bbox_prd, bbox_lbl)
    area_union = area_pred + area_label - area_inter
    return (area_inter + EPSILON) / (area_union + EPSILON)


def from_xywh(xywh: tf.Tensor) -> tf.Tensor:
    """Convert bounding box from (x, y, w, h) to (ymin, xmin, ymax, xmax).

    Args:
        xywh (tf.Tensor): bounding box tensor (XYWH) of shape (H, W, 9, C)
            where C >= 4

    Returns:
        tf.Tensor: bounding box tensor (YXYX) of shape (H, W, 9, C) where
            C >= 4
    """
    x_, y_, w_, h_, rem_ = (xywh[..., 0], xywh[..., 1], xywh[..., 2],
                            xywh[..., 3], xywh[..., 4:])
    xmin_ = x_ - 0.5 * w_
    ymin_ = y_ - 0.5 * h_
    xmax_ = x_ + 0.5 * w_
    ymax_ = y_ + 0.5 * h_
    yxyx = tf.stack([ymin_, xmin_, ymax_, xmax_], axis=-1)
    return tf.concat([yxyx, rem_], axis=-1)


def to_xywh(bbox: tf.Tensor) -> tf.Tensor:
    """Convert bounding box from (ymin, xmin, ymax, xmax) to (x, y, w, h).

    Args:
        bbox (tf.Tensor): bounding box tensor (YXYX) of shape (H, W, 9, C)
            where C >= 4

    Returns:
        tf.Tensor: bounding box tensor (XYWH) of shape (H, W, 9, C) where
            C >= 4
    """
    xywh = tf.stack([xctr(bbox), yctr(bbox), w(bbox), h(bbox)], axis=-1)
    return tf.concat([xywh, rem(bbox)], axis=-1)
