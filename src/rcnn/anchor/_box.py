"""Manipulate YXYX bounding boxes.

shape: (N1, N2, ..., Nk, C) where C >= 4

format: (y_min, x_min, y_max, x_max, objectness score, ...)
"""
from typing import Union

import numpy as np
import tensorflow as tf

EPS = 1e-4
TensorT = Union[tf.Tensor, np.ndarray]  # noqa: UP007


def xmin(bbox: TensorT) -> TensorT:
    """Get top-left x coordinate of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: X-min tensor of shape (N1, N2, ..., Nk)
    """
    return bbox[..., 1]


def ymin(bbox: TensorT) -> TensorT:
    """Get top-left y coordinate of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: Y-min tensor of shape (N1, N2, ..., Nk)
    """
    return bbox[..., 0]


def xmax(bbox: TensorT) -> TensorT:
    """Get bottom-right x coordinate of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: X-max tensor of shape (N1, N2, ..., Nk)
    """
    return bbox[..., 3]


def ymax(bbox: TensorT) -> TensorT:
    """Get bottom-right y coordinate of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: Y-max tensor of shape (N1, N2, ..., Nk)
    """
    return bbox[..., 2]


def rem(bbox: TensorT) -> TensorT:
    """Get remainders (excluding YXYX) of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: remainder tensor of shape (N1, N2, ..., Nk, C - 4)
    """
    return bbox[..., 4:]


def w(bbox: TensorT) -> TensorT:
    """Get width of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: width tensor of shape (N1, N2, ..., Nk)
    """
    return xmax(bbox) - xmin(bbox)


def h(bbox: TensorT) -> TensorT:
    """Get height of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: height tensor of shape (N1, N2, ..., Nk)
    """
    return ymax(bbox) - ymin(bbox)


def xctr(bbox: TensorT) -> TensorT:
    """Get center x coordinate of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: X-center tensor of shape (N1, N2, ..., Nk)
    """
    return xmin(bbox) + 0.5 * w(bbox)


def yctr(bbox: TensorT) -> TensorT:
    """Get center y coordinate of each anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: Y-center tensor of shape (N1, N2, ..., Nk)
    """
    return ymin(bbox) + 0.5 * h(bbox)


def area(bbox: TensorT) -> TensorT:
    """Get area of the anchor box.

    Args:
        bbox (TensorT): Bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: area tensor of shape (N1, N2, ..., Nk)
    """
    return h(bbox) * w(bbox)


def pmax(bbox: TensorT) -> TensorT:
    """Get bottom-right point from each anchor box.

    Args:
        bbox (TensorT): bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: YX-max tensor of shape (N1, N2, ..., Nk, 2)
    """
    return bbox[..., 2:4]


def pmin(bbox: TensorT) -> TensorT:
    """Get top-left point from each anchor box.

    Args:
        bbox (TensorT): bounding box tensor of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: YX-min tensor of shape (N1, N2, ..., Nk, 2)
    """
    return bbox[..., 0:2]


def confnd(bbox: TensorT) -> TensorT:
    """Get object confidence from an anchor box (un-squeezed).

    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        bbox (TensorT): anchor box of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: object confidence tensor of shape (N1, N2, ..., Nk, 1)
    """
    return bbox[..., 4:5]


def conf1d(bbox: TensorT) -> TensorT:
    """Get object confidence from an anchor box (squeezed).

    Suppose the number of ranks of the input tensor is R, the #rank of the
    output tensor will be R - 1 is `squeezed`.
    Otherwise the #rank of the output will remain as R, and the last
    rank contains only 1 dimension

    Args:
        bbox (TensorT): anchor box of shape (N1, N2, ..., Nk, C)

    Returns:
        TensorT: object confidence tensor of shape (N1, N2, ..., Nk)
    """
    return bbox[..., 4]


def interarea(bbox_prd: tf.Tensor, bbox_lbl: tf.Tensor) -> tf.Tensor:
    """Get intersection area of two sets of anchor boxes.

    Args:
        bbox_prd (tf.Tensor): predicted bounding box tensor of shape
            (N1, N2, ..., Nk, C)
        bbox_lbl (tf.Tensor): label bounding box tensor of shape
            (N1, N2, ..., Nk, C)

    Returns:
        tf.Tensor: intersection area tensor of shape (N1, N2, ..., Nk)
    """
    left_ups = tf.maximum(pmin(bbox_prd), pmin(bbox_lbl))
    right_downs = tf.minimum(pmax(bbox_prd), pmax(bbox_lbl))

    inter = tf.maximum(right_downs - left_ups, 0.0)
    return tf.multiply(inter[..., 0], inter[..., 1])


def iou(bbox_prd: tf.Tensor, bbox_lbl: tf.Tensor) -> tf.Tensor:
    """Calculate IoU of two bounding boxes.

    Args:
        bbox_prd (tf.Tensor): predicted bounding box tensor of shape
            (N1, N2, ..., Nk, C)
        bbox_lbl (tf.Tensor): label bounding box tensor of shape
            (N1, N2, ..., Nk, C)

    Returns:
        tf.Tensor: IoU tensor of shape (N1, N2, ..., Nk)
    """
    area_pred = area(bbox_prd)
    area_label = area(bbox_lbl)
    area_inter = interarea(bbox_prd, bbox_lbl)
    area_union = area_pred + area_label - area_inter
    return (area_inter + EPS) / (area_union + EPS)


def iou_mat(bbox_prd: tf.Tensor, bbox_lbl: tf.Tensor) -> tf.Tensor:
    """Calculate IoU matrix of two sets of bounding boxes.

    Args:
        bbox_prd (tf.Tensor): predicted bounding boxes of shape (N1, C)
        bbox_lbl (tf.Tensor): ground truth bounding boxes of shape (N2, C)

    Returns:
        tf.Tensor: IoU tensor of shape (N1, N2)
    """
    n1, n2 = tf.shape(bbox_prd)[0], tf.shape(bbox_lbl)[0]
    # convert to shape (N1, N2, C)
    bbox_prd_ = tf.tile(tf.expand_dims(bbox_prd, axis=1), [1, n2, 1])
    bbox_lbl_ = tf.tile(tf.expand_dims(bbox_lbl, axis=0), [n1, 1, 1])
    return iou(bbox_prd_, bbox_lbl_)


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
#     ],
#     autograph=True,
# )
def iou_batch(bbox_prd: tf.Tensor, bbox_lbl: tf.Tensor) -> tf.Tensor:
    """Calculate IoU matrix for each batch of two sets of bounding boxes.

    Args:
        bbox_prd (tf.Tensor): predicted bounding boxes of shape (B, N1, C)
        bbox_lbl (tf.Tensor): ground truth bounding boxes of shape (B, N2, C)

    Returns:
        tf.Tensor: IoU tensor of shape (B, N1, N2)
    """
    n1, n2 = tf.shape(bbox_prd)[1], tf.shape(bbox_lbl)[1]
    # convert to shape (B, N1, N2, C)
    bbox_prd_ = tf.tile(tf.expand_dims(bbox_prd, axis=2), [1, 1, n2, 1])
    bbox_lbl_ = tf.tile(tf.expand_dims(bbox_lbl, axis=1), [1, n1, 1, 1])
    return iou(bbox_prd_, bbox_lbl_)


def from_xywh(xywh: tf.Tensor) -> tf.Tensor:
    """Convert bounding box from (x, y, w, h) to (ymin, xmin, ymax, xmax).

    Args:
        xywh (tf.Tensor): bounding box tensor (XYWH) of shape
            (N1, N2, ..., Nk, 4)

    Returns:
        tf.Tensor: bounding box tensor (YXYX) of shape (N1, N2, ..., Nk, 4)
    """
    x_, y_, w_, h_ = (xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3])
    xmin_ = x_ - 0.5 * w_
    ymin_ = y_ - 0.5 * h_
    xmax_ = x_ + 0.5 * w_
    ymax_ = y_ + 0.5 * h_
    return tf.stack([ymin_, xmin_, ymax_, xmax_], axis=-1)


def to_xywh(bbox: tf.Tensor) -> tf.Tensor:
    """Convert bounding box from (ymin, xmin, ymax, xmax) to (x, y, w, h).

    Args:
        bbox (tf.Tensor): bounding box tensor (YXYX) of shape
            (N1, N2, ..., Nk, C) where C >= 4

    Returns:
        tf.Tensor: bounding box tensor (XYWH) of shape (N1, N2, ..., Nk, C)
    """
    xywh = tf.stack([xctr(bbox), yctr(bbox), w(bbox), h(bbox)], axis=-1)
    return tf.concat([xywh, rem(bbox)], axis=-1)


def clip(bbox: tf.Tensor, h: float, w: float) -> tf.Tensor:
    """Clip bounding box to a given image shape.

    Args:
        bbox (tf.Tensor): bounding box tensor (YXYX) of shape
            (N1, N2, ..., Nk, C) where C >= 4
        h (float): image height
        w (float): image width

    Returns:
        tf.Tensor: clipped bounding box tensor (YXYX) of shape
            (N1, N2, ..., Nk, C)
    """
    ymin_ = tf.clip_by_value(ymin(bbox), 0.0, h)
    xmin_ = tf.clip_by_value(xmin(bbox), 0.0, w)
    ymax_ = tf.clip_by_value(ymax(bbox), 0.0, h)
    xmax_ = tf.clip_by_value(xmax(bbox), 0.0, w)
    yxyx = tf.stack([ymin_, xmin_, ymax_, xmax_], axis=-1)
    return tf.concat([yxyx, rem(bbox)], axis=-1)
