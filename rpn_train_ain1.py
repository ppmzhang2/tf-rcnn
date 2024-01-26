"""all-in-one script for training the RPN model of Mask R-CNN."""

from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# training
EPS = 1e-4

# model
STRIDE = 32
SIZE_RESIZE = 600  # image size for resizing
SIZE_IMG = 512  # image size
W = SIZE_IMG  # original image width
H = SIZE_IMG  # original image height
C = 3  # number of channels
W_FM = W // STRIDE  # feature map width
H_FM = H // STRIDE  # feature map height
N_ANCHOR = 9  # number of anchors per grid cell
N_OBJ = 20
N_SUPP_SCORE = 300  # number of boxes to keep after score suppression
N_SUPP_NMS = 10  # number of boxes to keep after nms
NMS_TH = 0.7  # nms threshold

# dataset
BUFFER_SIZE = 100
BATCH_SIZE_TR = 8
BATCH_SIZE_TE = 8

# RPN
R_DROP = 0.2  # dropout rate
IOU_TH = 0.5  # IoU threshold for calculating mean Average Precision (mAP)
IOU_SCALE = 10000  # scale for IoU for converting to int
NEG_TH_ACGT = int(IOU_SCALE * 0.30)  # lower bound of AC-GT highest IoU
POS_TH_ACGT = int(IOU_SCALE * 0.50)  # upper bound of AC-GT highest IoU (best)
NEG_TH_GTAC = int(IOU_SCALE * 0.01)  # lower bound of GT-AC highest IoU
NUM_POS_RPN = 128  # number of positive anchors
NUM_NEG_RPN = 128  # number of negative anchors

# Visualize
COLOR_BOX = (0, 255, 0)  # Green color for bounding box
COLOR_TXT = (0, 0, 255)  # Red color for class tag
THICKNESS_BOX = 2  # Line thickness
THICKNESS_TXT = 1  # Text thickness
SIZE_FONT = 0.5  # Font size

# image normalization
IMGNET_STD = np.array([58.393, 57.12, 57.375], dtype=np.float32)
IMGNET_MEAN = np.array([123.68, 116.78, 103.94], dtype=np.float32)

TensorT = Union[tf.Tensor, np.ndarray]  # noqa: UP007

# =============================================================================
# SECTION: dataset
# =============================================================================


def resize(img: tf.Tensor, size: int) -> tf.Tensor:
    """Resize an image and bounding boxes while preserving its aspect ratio.

    Args:
        img (tf.Tensor): image to resize
        bbx (tf.Tensor): bounding boxes to resize
        size (int): desired size of the shorter side of the image
        keep_ratio (bool, optional): whether to keep the aspect ratio.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: resized image and bounding boxes
    """
    h, w, _ = tf.unstack(tf.shape(img))
    scale = tf.maximum(
        tf.cast(size / h, tf.float32),
        tf.cast(size / w, tf.float32),
    )  # scale to the shorter side
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [new_h, new_w])
    return img


def rand_crop(
    img: tf.Tensor,
    bbx: tf.Tensor,
    h_target: int,
    w_target: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Randomly crops an image and adjusts the bounding boxes.

    Args:
        img (tf.Tensor): Tensor representing the image.
        bbx (tf.Tensor): Tensor of bounding boxes in relative coordinates,
        shape [num_boxes, 4].
        h_target (int): Target height for the cropped image.
        w_target (int): Target width for the cropped image.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: Cropped image and adjusted bounding boxes.
    """
    h_img, w_img, _ = tf.unstack(tf.shape(img))
    max_h_offset = h_img - h_target - 1
    max_w_offset = w_img - w_target - 1

    # Randomly choosing the offset for cropping
    h_offset = tf.random.uniform(shape=[],
                                 minval=0,
                                 maxval=max_h_offset,
                                 dtype=tf.int32)
    w_offset = tf.random.uniform(shape=[],
                                 minval=0,
                                 maxval=max_w_offset,
                                 dtype=tf.int32)

    # Cropping the image
    img_crop = tf.image.crop_to_bounding_box(img, h_offset, w_offset, h_target,
                                             w_target)

    # Adjust bounding boxes
    # Scale bounding box coordinates to the cropped image size
    ymin, xmin, ymax, xmax = bbx[..., 0], bbx[..., 1], bbx[..., 2], bbx[..., 3]

    # Clip bounding boxes to be within the cropped image
    ymin = (ymin * tf.cast(h_img, tf.float32) -
            tf.cast(h_offset, tf.float32)) / tf.cast(h_target, tf.float32)
    ymax = (ymax * tf.cast(h_img, tf.float32) -
            tf.cast(h_offset, tf.float32)) / tf.cast(h_target, tf.float32)
    xmin = (xmin * tf.cast(w_img, tf.float32) -
            tf.cast(w_offset, tf.float32)) / tf.cast(w_target, tf.float32)
    xmax = (xmax * tf.cast(w_img, tf.float32) -
            tf.cast(w_offset, tf.float32)) / tf.cast(w_target, tf.float32)

    # Clip bounding boxes to be within the cropped image
    bbx_clip = tf.stack(
        [
            tf.clip_by_value(ymin, 0, 1),
            tf.clip_by_value(xmin, 0, 1),
            tf.clip_by_value(ymax, 0, 1),
            tf.clip_by_value(xmax, 0, 1),
        ],
        axis=1,
    )

    return img_crop, bbx_clip


def batch_pad(
    tensor: tf.Tensor,
    max_box: int,
    value: int,
) -> tf.Tensor:
    """Pad a tensor (either labels or bounding boxes) as a batch of fixed size.

    Args:
        tensor (tf.Tensor): tensor to pad, must be at least 2nd order, e.g.
            `len(tensor.shape) >= 2`
        max_box (int): maximum number of boxes
        value (int): value to pad with

    Returns:
        tf.Tensor: padded tensor
    """
    padding_size = max_box - tf.shape(tensor)[0]
    if padding_size > 0:
        tensor = tf.pad(tensor,
                        paddings=[[0, padding_size], [0, 0]],
                        constant_values=value)
    else:
        tensor = tensor[:max_box]
    return tensor


def data_augment(
    img: tf.Tensor,
    bbx: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply at most one data augmentation to the image and bounding boxes.

    Args:
        img (tf.Tensor): Input image
        bbx (tf.Tensor): Bounding boxes associated with the image

    Returns:
        tuple[tf.Tensor, tf.Tensor]: Augmented image and adjusted bounding
        boxes.
    """
    # Proportion for each operation and idle
    proportion = 0.25

    # Random number for choosing the augmentation or idle
    rand_aug = tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32)

    # Horizontal flip
    if 0 <= rand_aug < 1 * proportion:
        img = tf.image.flip_left_right(img)
        ymin, xmin, ymax, xmax = tf.unstack(bbx, axis=1)
        flipped_bbx = tf.stack([ymin, 1.0 - xmax, ymax, 1.0 - xmin], axis=1)
        bbx = flipped_bbx

    # Gaussian noise
    elif 1 * proportion <= rand_aug < 2 * proportion:
        noise = tf.random.normal(
            shape=tf.shape(img),
            mean=0.0,
            stddev=0.5,
            dtype=tf.float32,
        )
        img = img + noise

    # Random brightness
    elif 2 * proportion <= rand_aug < 3 * proportion:
        img = tf.image.random_brightness(img, max_delta=2.0)

    # No operation is done in the range [3 * proportion, 1]

    return img, bbx


def preprcs_tr(sample: dict) -> tuple[tf.Tensor, tuple[tf.Tensor]]:
    """Preprocesses a sample from the dataset.

    Args:
        sample (dict): sample from the dataset
        e.g. {
            "image": tf.Tensor,
            "objects": {
                "bbox": tf.Tensor,
                "label": tf.Tensor,
            },
        }

    Returns:
        tuple[tf.Tensor, tuple[tf.Tensor]]: preprocessed image and targets
            - images: [batch_size, H, W, 3]
            - bounding boxes: [batch_size, max_box, 4] in relative coordinates
            - labels: [batch_size, max_box, 1]
    """
    img = sample["image"]
    # Normalize the image
    img = (tf.cast(img, dtype=tf.float32) - IMGNET_MEAN) / IMGNET_STD

    # Get the bounding boxes and labels
    bbx = sample["objects"]["bbox"]
    lab = sample["objects"]["label"]

    # resize the image and bounding boxes while preserving the aspect ratio
    img = resize(img, SIZE_RESIZE)
    # randomly crop the image and bounding boxes
    img, bbx = rand_crop(img, bbx, SIZE_IMG, SIZE_IMG)

    # randomly augment the image and bounding boxes
    img, bbx = data_augment(img, bbx)

    # pad the labels and bounding boxes to a fixed size
    bbx = batch_pad(bbx, max_box=N_OBJ, value=0)
    lab = batch_pad(lab[:, tf.newaxis], max_box=N_OBJ, value=-1)

    return img, (bbx, lab)


def preprcs_te(sample: dict) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Preprocess test dataset samples.

    Args:
        sample (dict): sample from the dataset
        e.g. {
            "image": tf.Tensor,
            "objects": {
                "bbox": tf.Tensor,
                "label": tf.Tensor,
            },
        }

    Returns:
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: images, bounding boxes, and
            labels
            - images: [batch_size, H, W, 3]
            - bounding boxes: [batch_size, max_box, 4] in relative coordinates
            - labels: [batch_size, max_box, 1]

    TODO: predict with full-sized images
    """
    img = sample["image"]
    # Normalize the image with ImageNet mean and std
    img = (tf.cast(img, dtype=tf.float32) - IMGNET_MEAN) / IMGNET_STD

    # Get the bounding boxes and labels
    bbx = sample["objects"]["bbox"]
    lbl = sample["objects"]["label"]

    # resize the image and bounding boxes while preserving the aspect ratio
    img = resize(img, SIZE_RESIZE)
    # randomly crop the image and bounding boxes
    img, bbx = rand_crop(img, bbx, SIZE_IMG, SIZE_IMG)

    # pad the labels and bounding boxes to a fixed size
    bbx = batch_pad(bbx, max_box=N_OBJ, value=0)
    lbl = batch_pad(lbl[:, tf.newaxis], max_box=N_OBJ, value=-1)

    return img, (bbx, lbl)


def load_train_voc2007(
        n: int) -> tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads the training dataset of Pascal VOC 2007.

    Args:
        name (str): name of the dataset
        n (int): number of training samples per batch

    Returns:
        tuple[tf.data.Dataset, tfds.core.DatasetInfo]: training datasets and
        dataset info.

    The training and validation datasets are shuffled and preprocessed with
    `ds_handler`:
        - images: [batch_size, H, W, 3]
        - bounding boxes: [batch_size, max_box, 4] in relative
        - labels: [batch_size, max_box, 1]
    """
    ds, ds_info = tfds.load(
        "voc/2007",
        split="train",
        shuffle_files=True,
        with_info=True,
    )
    ds = ds.map(
        preprcs_tr,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).shuffle(BUFFER_SIZE).batch(n).prefetch(tf.data.experimental.AUTOTUNE)
    return ds, ds_info


def load_test_voc2007(n: int) -> tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads the testing dataset of Pascal VOC 2007.

    Args:
        n (int): number of testing samples per batch

    Returns:
        tuple[tf.data.Dataset, tfds.core.DatasetInfo]: testing dataset and
            dataset info.

    The testing dataset is preprocessed with `ds_handler`:
        - images: [batch_size, H, W, 3]
        - bounding boxes: [batch_size, max_box, 4] in relative
        - labels: [batch_size, max_box, 1]
    """
    ds_te, ds_info = tfds.load(
        "voc/2007",
        split="validation",
        shuffle_files=False,
        with_info=True,
    )
    ds_te = ds_te.map(
        preprcs_te,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(n).prefetch(tf.data.experimental.AUTOTUNE)
    return ds_te, ds_info


# =============================================================================
# SECTION: anchor
# =============================================================================


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
        np.tile(xs, (h, 1, N_ANCHOR)),
        np.tile(ys, (1, w, N_ANCHOR)),
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
anchors_raw = get_rel_anchor(H, W, STRIDE, flat=True)

# valid anchors mask based on image size of type np.float32 (N_ALL_AC,)
BOUND_AC_LO = -0.2
BOUND_AC_HI = 1.2
MASK_RPNAC = np.where(
    (anchors_raw[..., 0] >= BOUND_AC_LO) &  # y_min >= lo
    (anchors_raw[..., 1] >= BOUND_AC_LO) &  # x_min >= lo
    (anchors_raw[..., 2] <= BOUND_AC_HI) &  # y_max <= hi
    (anchors_raw[..., 3] <= BOUND_AC_HI) &  # x_max <= hi
    (anchors_raw[..., 2] > anchors_raw[..., 0]) &  # y_max > y_min
    (anchors_raw[..., 3] > anchors_raw[..., 1]),  # x_max > x_min
    1.0,
    0.0,
)

RPNAC = anchors_raw[MASK_RPNAC == 1]  # valid anchors (N_VAL_AC, 4)

# number of valid anchors
N_RPNAC = int(MASK_RPNAC.sum())

# set as read-only
RPNAC.flags.writeable = False
MASK_RPNAC.flags.writeable = False

# valid anchors
AC_VAL = tf.constant(RPNAC, dtype=tf.float32)  # (N_VAL_AC, 4)

# =============================================================================
# SECTION: YXYX bounding boxes.
#
# shape: (N1, N2, ..., Nk, C) where C >= 4
#
# format: (y_min, x_min, y_max, x_max, objectness score, ...)
# =============================================================================


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


# TODO: GIoU
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


# =============================================================================
# SECTION: Delta
# =============================================================================


def dx(delta: tf.Tensor) -> tf.Tensor:
    """Get delta x coordinate of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 0]


def dy(delta: tf.Tensor) -> tf.Tensor:
    """Get delta y coordinate of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 1]


def dw(delta: tf.Tensor) -> tf.Tensor:
    """Get delta width of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 2]


def dh(delta: tf.Tensor) -> tf.Tensor:
    """Get delta height of each delta box.

    Args:
        delta (tf.Tensor): delta tensor of shape (H, W, 9, 4)

    Returns:
        tf.Tensor: delta tensor of shape (H, W, 9)
    """
    return delta[..., 3]


# =============================================================================
# SECTION: utils
# =============================================================================


def delta2bbox(base: tf.Tensor, diff: tf.Tensor) -> tf.Tensor:
    """Apply delta to anchors to get bbox.

    e.g.: anchor (base) + delta (diff) = predicted (bbox)

    Args:
        base (tf.Tensor): base bbox tensor of shape (N1, N2, ..., Nk, 4)
        diff (tf.Tensor): delta tensor of shape (N1, N2, ..., Nk, 4)

    Returns:
        tf.Tensor: bbox tensor of shape (N1, N2, ..., Nk, 4)
    """
    xctr_ = xctr(base) + w(base) * dx(diff)
    yctr_ = yctr(base) + h(base) * dy(diff)
    w_ = w(base) * tf.exp(dw(diff))
    h_ = h(base) * tf.exp(dh(diff))
    xywh_ = tf.stack([xctr_, yctr_, w_, h_], axis=-1)
    return from_xywh(xywh_)


def bbox2delta(bbox_l: tf.Tensor, bbox_r: tf.Tensor) -> tf.Tensor:
    """Compute delta between two bounding boxes.

    e.g.:
        - GT (bbox_l) - anchor (bbox_r) = RPN target (delta)
        - GT (bbox_l) - ROI (bbox_r) = RCNN target (delta)

    Args:
        bbox_l (tf.Tensor): minuend bbox tensor (left operand) of shape
            (N1, N2, ..., Nk, C), where C >= 4
        bbox_r (tf.Tensor): subtrahend bbox tensor (right operand) of shape
            (N1, N2, ..., Nk, C), where C >= 4

    Returns:
        tf.Tensor: delta tensor of shape (N1, N2, ..., Nk, 4)
    """
    xctr_r = xctr(bbox_r)
    yctr_r = yctr(bbox_r)
    w_r = tf.math.maximum(w(bbox_r), EPS)
    h_r = tf.math.maximum(h(bbox_r), EPS)
    xctr_l = xctr(bbox_l)
    yctr_l = yctr(bbox_l)
    w_l = w(bbox_l)
    h_l = h(bbox_l)
    bx_del = tf.stack(
        [
            (xctr_l - xctr_r) / w_r,
            (yctr_l - yctr_r) / h_r,
            tf.math.log(w_l / w_r),
            tf.math.log(h_l / h_r),
        ],
        axis=-1,
    )
    return tf.clip_by_value(bx_del, -999.0, 999.0)


# =============================================================================
# SECTION: RPN training utils
# =============================================================================
def sample_mask(mask: tf.Tensor, num: int) -> tf.Tensor:
    """Sample `num` anchors from `mask` for a batch of images.

    TODO: more precise sampling

    Args:
        mask (tf.Tensor): 0/1 mask of anchors (B, N_ac)
        num (int): number of positive anchors to sample

    Returns:
        tf.Tensor: 0/1 mask of anchors (B, N_ac)
    """
    th = num / (tf.reduce_sum(mask, axis=-1, keepdims=True) + 1e-6)
    rand = tf.random.uniform(
        tf.shape(mask),
        minval=0,
        maxval=1,
        dtype=tf.float32,
    )
    return tf.cast(rand < th, tf.float32) * mask


def get_gt_mask(bx_tgt: tf.Tensor, *, bkg: bool = False) -> tf.Tensor:
    """Get target mask for each anchor based on target boxes for RPN training.

    Args:
        bx_tgt (tf.Tensor): target ground truth boxes (B, N_ac, 4)
        bkg (bool, optional): whether to indicate background. Defaults False.

    Returns:
        tf.Tensor: 0/1 mask for each box (B, N_ac) for each anchor
    """
    # 0. coordinate sum of target boxes (B, N_ac):
    #    - positive: foreground
    #    - -4.0: background
    #    - 0.0: ignore
    _coor_sum = tf.reduce_sum(bx_tgt, axis=-1)
    # init with tf.float32 zeros
    mask = tf.zeros_like(_coor_sum, dtype=tf.float32)  # (B, N_ac)
    if bkg:
        mask = tf.where(_coor_sum < 0, 1.0, mask)
        return sample_mask(mask, NUM_NEG_RPN)
    mask = tf.where(_coor_sum > 0, 1.0, mask)
    return sample_mask(mask, NUM_POS_RPN)


def filter_on_max(x: tf.Tensor) -> tf.Tensor:
    """Filter rows based on the maximum value in the second column.

    Given a 2D tensor, the function filters the rows based on the following
    condition:
    For rows having the same value in the first column (key), retain only the
    row that has the maximum value in the second column (score).

    Args:
        x (tf.Tensor): 2D tensor (M, 2) of format (key, score)

    Returns:
        tf.Tensor: 1D boolean mask (M,) indicating which rows to keep
    """
    # Get unique values and their indices from the first column
    k_unique, k_idx = tf.unique(x[:, 0])

    # Find the maximum values in the second column for each unique element in
    # the first column
    max_indices = tf.math.unsorted_segment_max(data=x[:, 1],
                                               segment_ids=k_idx,
                                               num_segments=tf.size(k_unique))

    # Create a boolean mask where each max value is True, others are False
    return tf.reduce_any(x[:, 1][:, tf.newaxis] == max_indices, axis=-1)


def get_gt_gtac(
    bx_ac: tf.Tensor,
    bx_gt: tf.Tensor,
    idx: tf.Tensor,
    ious: tf.Tensor,
) -> tf.Tensor:
    """Get target boxes for each anchor based on GT's best AC Match (GT-AC).

    Things are a bit different here compared to the AC-GT case:

    - for AC-GT, some anchors are truly representing background, therefore that
      low IoU means background makes sense
    - for GT-AC, however, all anchors are representing ground truth boxes,
      therefore only VERY low IoU should be ignored, otherwise they are all
      foreground.

    - get positions `coord_gt` of GT boxes with IoU higher than `NEG_TH_GTAC`;
      (using `flag_gtac` is also fine)
    - corresponding anchor positions `coord_ac` will also be selected
    - remove duplicates **anchors** (by batch index and anchor index) and keep
      the one with the highest IoU
    - update target boxes with selected GT boxes, and keep the rest as -1.0

    Args:
        bx_ac (tf.Tensor): anchor tensor (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)
        idx (tf.Tensor): pre-computed indices matrix (B, N_gt) where value in
            [0, N_ac), representing indices of the best mached anchor for each
            GT box
        ious (tf.Tensor): pre-computed IoU matrix (B, N_gt) where value in
            [0, 10000], representing the best IoU for each GT box

    Returns:
        tf.Tensor: best IoU matched GT boxes (B, N_ac, 4) for anchors
          - [-1.0, -1.0, -1.0, -1.0]: background
          - positive: foreground
    """
    # T/F matrix (B, N_gt) indicating whether the best IoU of each GT box is
    # above threshold
    flag_gtac = ious > NEG_TH_GTAC  # (B, N_gt)
    # coordinate pairs (M, 2) of ground truth boxes where `M <= B*N_gt`
    # of format (batch index, GT index), indicating the best matched GT boxes
    # no duplicates as `tf.where` here returns indices of non-zero elements,
    # which are unique
    coord_gt = tf.cast(tf.where(flag_gtac), tf.int32)
    # vector of best matched anchors' indices (M,) where value in [0, N_ac)
    # may have duplicates as multiple GT boxes may have the same best matched
    # anchor
    idx_ac = tf.gather_nd(idx, coord_gt)
    # coordinate pairs (M, 2) of anchors where `M <= B*N_gt` of format
    # (batch index, anchor index), indicating the best matched anchor
    # may have duplicates as `idx_ac` may have duplicates
    coord_ac = tf.stack([coord_gt[:, 0], idx_ac], axis=-1)

    # filtering: one anchor may have multiple matches with GT boxes, which can
    # lead to overwriting for the same anchor.
    # We only keep for one anchor the GT box with the highest IoU.
    # `arr` is a 2D tensor of format (M, 2) where `M <= B*N_gt`:
    # - the first column is the hash value of the coordinate pairs
    # - the second column is the best IoU of the corresponding coordinate pairs
    arr = tf.stack(
        [
            coord_ac[:, 0] * 10000 + coord_ac[:, 1],
            tf.boolean_mask(ious, flag_gtac, axis=0),
        ],
        axis=-1,
    )
    mask = filter_on_max(arr)  # (M,) with M1 True values where M1 <= M

    # update target boxes (B, N_ac, 4) with ground truth boxes
    # - indices: indicates the anchor positions, which have the best IoU
    # (against GT boxes) above threshold, to be updated
    # - updates: indicates the best matched GT boxes
    # corresponding to the anchor positions above
    return tf.tensor_scatter_nd_update(
        -tf.ones_like(bx_ac),  # init with -1, (B, N_ac, 4)
        tf.boolean_mask(coord_ac, mask),  # (M1, 2)
        tf.boolean_mask(tf.boolean_mask(bx_gt, flag_gtac), mask),  # (M1, 4)
    )


def get_gt_acgt(
    bx_ac: tf.Tensor,
    bx_gt: tf.Tensor,
    idx: tf.Tensor,
    ious: tf.Tensor,
) -> tf.Tensor:
    """Get target boxes for each anchor based on AC's best GT Match (AC-GT).

    There are three types of "target boxes":

    - background: AC-GT IoU < `NEG_TH_ACGT`; represented by all -1.0
    - foreground:
      - i.e. ground truth boxes
      - should be the best matched GT boxes for each anchor
      - but we are using the AC-GT IoU here, which is a bit awkward:
        - get coordinates (BATCH_INDEX, ANCHOR_INDEX) of anchors with the AC-GT
          IoU higher than `POS_TH_ACGT`
        - get corresponding GT box coordinates (BATCH_INDEX, GT_INDEX)
    - ignore: AC-GT IoU in [NEG_TH_ACGT, POS_TH_ACGT); represented by all 0.0

    Args:
        bx_ac (tf.Tensor): anchor tensor (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)
        idx (tf.Tensor): pre-computed indices matrix (B, N_ac) where
            value in [0, N_gt), representing indices of the best mached GT box
            for each anchor
        ious (tf.Tensor): pre-computed IoU matrix (B, N_ac) where value in
            [0, 10000], representing the best IoU for each anchor

    Returns:
        tf.Tensor: GT boxes (B, N_ac, 4) for each anchor of tf.float32
          - [-1.0, -1.0, -1.0, -1.0]: background
          - [0.0, 0.0, 0.0, 0.0]: ignore
          - otherwise: foreground
    """
    # idx_acgt = tf.argmax(ious, axis=2, output_type=tf.int32)
    # iou_acgt = tf.reduce_max(ious, axis=2)

    # initialize with zeros
    bx_tgt_acgt = tf.zeros_like(bx_ac, dtype=tf.float32)  # (B, N_ac, 4)

    # -------------------------------------------------------------------------
    # 1. detect background
    # -------------------------------------------------------------------------

    bx_tgt_acgt = tf.where(
        tf.repeat(ious[..., tf.newaxis], 4, axis=-1) < NEG_TH_ACGT,
        tf.constant([[-1.0]]),
        bx_tgt_acgt,
    )  # (N_ac, 4)

    # -------------------------------------------------------------------------
    # 2. detect AC-GT foreground
    # -------------------------------------------------------------------------

    # 2.1. identify foreground anchors
    #   - T/F matrix (B, N_ac) indicating whether the best IoU of each anchor
    #     is above threshold
    flag_acgt = ious > POS_TH_ACGT
    #   - coordinate pairs (M, 2) of anchors where `M <= B*N_ac` of format
    #     (batch index, AC index), indicating the best matched anchors
    #   - no duplicates as `tf.where` here returns indices of non-zero
    #     elements, which are unique
    coord_ac = tf.cast(tf.where(flag_acgt), tf.int32)
    # 2.2. For each anchor considered as foreground, finds the ground truth
    #      box that has the highest IoU with that anchor.
    #   - vector of best matched GT boxes' indices (M,) where value in
    #     [0, N_gt)
    #   - may have duplicates
    values = tf.gather_nd(idx, coord_ac)
    #   - coordinate pairs (M, 2) of GT boxes where `M <= B*N_ac` of format
    #     (batch index, GT index), indicating the best matched GT boxes
    #   - may have duplicates as `values` may have duplicates
    coord_gt = tf.stack([coord_ac[:, 0], values], axis=-1)

    # no need to filter as we only concern about the duplicates in `coord_ac`,
    # which can lead to overwriting of target boxes.
    return tf.tensor_scatter_nd_update(
        bx_tgt_acgt,  # (B, N_ac, 4)
        coord_ac,  # (M, 2)
        tf.gather_nd(bx_gt, coord_gt),  # (M, 4)
    )  # (B, N_ac, 4)


def get_gt_box(bx_ac: tf.Tensor, bx_gt: tf.Tensor) -> tf.Tensor:
    """Get ground truth boxes based on IoU for each anchor for RPN training.

    Args:
        bx_ac (tf.Tensor): anchor tensor (B, N_ac, 4)
        bx_gt (tf.Tensor): ground truth tensor (B, N_gt, 4)

    Returns:
        tf.Tensor: ground truth boxes (B, N_ac, 4) for each anchor (tf.float32)
          - [-1.0, -1.0, -1.0, -1.0]: background
          - [0.0, 0.0, 0.0, 0.0]: ignore
          - otherwise: foreground
    """
    ious = tf.cast(IOU_SCALE * iou_batch(bx_ac, bx_gt), tf.int32)
    idx_gtac = tf.argmax(ious, axis=1, output_type=tf.int32)
    iou_gtac = tf.reduce_max(ious, axis=1)
    idx_acgt = tf.argmax(ious, axis=2, output_type=tf.int32)
    iou_acgt = tf.reduce_max(ious, axis=2)
    bx_tgt_gtac = get_gt_gtac(bx_ac, bx_gt, idx_gtac, iou_gtac)
    bx_tgt_acgt = get_gt_acgt(bx_ac, bx_gt, idx_acgt, iou_acgt)
    return tf.where(bx_tgt_gtac >= 0, bx_tgt_gtac, bx_tgt_acgt)


# =============================================================================
# SECTION: models
# =============================================================================


def roi(dlt: tf.Tensor, buffer: float = 1e-1) -> tf.Tensor:
    """Get RoI bounding boxes from anchor deltas.

    Args:
        dlt (tf.Tensor): RPN predicted deltas. Shape [B, N_VAL_AC, 4].
        buffer (float, optional): buffer to clip the RoIs. Defaults to 1e-1.

    Returns:
        tf.Tensor: clipped RoIs in shape [B, N_VAL_AC, 4].
    """
    bsize = tf.shape(dlt)[0]
    # Computing YXYX RoIs from deltas. output shape: (B, N_VAL_AC, 4)
    rois = delta2bbox(tf.repeat(AC_VAL[tf.newaxis, ...], bsize, axis=0), dlt)
    # clip the RoIs
    return tf.clip_by_value(rois, -buffer, 1. + buffer)  # (B, N_VAL_AC, 4)


# -----------------------------------------------------------------------------
# RPN
# -----------------------------------------------------------------------------

SEED_INIT = 42
MASK_AC = tf.constant(MASK_RPNAC, dtype=tf.float32)  # (N_ANCHOR,)
N_VAL_AC = N_RPNAC  # number of valid anchors, also the dim of axis=1

reg_l2 = tf.keras.regularizers.l2(0.0005)
# Derive three unique seeds from the initial seed
seed_cnn_fm = hash(SEED_INIT) % (2**32)
seed_cnn_dlt = hash(seed_cnn_fm) % (2**32)
seed_cnn_lbl = hash(seed_cnn_dlt) % (2**32)


def rpn(
    fm: tf.Tensor,
    h_fm: int,
    w_fm: int,
) -> tuple[tf.Tensor, tf.Tensor, list[tf.keras.layers.Layer]]:
    """Region Proposal Network (RPN).

    Args:
        fm (tf.Tensor): feature map from the backbone.
        h_fm (int): The height of the feature map.
        w_fm (int): The width of the feature map.

    Returns:
        tuple[tf.Tensor, tf.Tensor, list[tf.keras.layers.Layer]]: predicted
        deltas, labels, and the RPN layers for weight freezing.
            - deltas: (n_batch, N_VAL_AC, 4)
            - labels: (n_batch, N_VAL_AC, 1)
    """
    # TBD: residual connections?
    layer_drop_entr = tf.keras.layers.Dropout(
        R_DROP,
        name="rpn_dropout_entr",
    )  # entrance dropout
    layer_conv_share = tf.keras.layers.Conv2D(
        SIZE_IMG,
        (3, 3),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_fm),
        kernel_regularizer=reg_l2,
        padding="same",
        activation=None,
        name="rpn_share",
    )  # shared convolutional layer
    layer_gn_share = tf.keras.layers.GroupNormalization()
    layer_relu_share = tf.keras.layers.Activation("relu")
    layer_drop_share = tf.keras.layers.Dropout(R_DROP)

    # (n_batch, h_fm, w_fm, SIZE_IMG)
    fm = layer_drop_entr(fm)
    fm = layer_conv_share(fm)
    fm = layer_gn_share(fm)
    fm = layer_relu_share(fm)
    fm = layer_drop_share(fm)

    # deltas
    layer_conv_dlt = tf.keras.layers.Conv2D(
        N_ANCHOR * 4,
        (1, 1),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_dlt),
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_dlt",
    )
    layer_bn_dlt = tf.keras.layers.BatchNormalization()
    layer_drop_dlt = tf.keras.layers.Dropout(R_DROP)
    # (n_batch, h_fm, w_fm, N_ANCHOR * 4)
    dlt = layer_conv_dlt(fm)
    dlt = layer_bn_dlt(dlt)
    dlt = layer_drop_dlt(dlt)

    # logits objectness score
    layer_conv_log = tf.keras.layers.Conv2D(
        N_ANCHOR,
        (1, 1),
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed_cnn_lbl),
        kernel_regularizer=reg_l2,
        activation=None,
        name="rpn_log",
    )
    layer_bn_log = tf.keras.layers.BatchNormalization()
    layer_drop_log = tf.keras.layers.Dropout(R_DROP)
    # (n_batch, h_fm, w_fm, N_ANCHOR)
    log = layer_conv_log(fm)
    log = layer_bn_log(log)
    log = layer_drop_log(log)

    # flatten the tensors
    # shape: (B, H_FM * W_FM * N_ANCHOR, 4) and (B, H_FM * W_FM * N_ANCHOR, 1)
    dlt_flat = tf.reshape(dlt, (-1, h_fm * w_fm * N_ANCHOR, 4))
    log_flat = tf.reshape(log, (-1, h_fm * w_fm * N_ANCHOR, 1))

    # Get valid labels and deltas based on valid anchor masks.
    # shape: (B, N_VAL_AC, 4) and (B, N_VAL_AC, 1)
    dlt_val = tf.boolean_mask(dlt_flat, MASK_AC == 1, axis=1)
    log_val = tf.boolean_mask(log_flat, MASK_AC == 1, axis=1)

    # for shape inference
    return (
        tf.reshape(dlt_val, (-1, N_VAL_AC, 4)),
        tf.reshape(log_val, (-1, N_VAL_AC, 1)),
        [
            layer_drop_entr,
            layer_conv_share,
            layer_gn_share,
            layer_relu_share,
            layer_drop_share,
            layer_conv_dlt,
            layer_bn_dlt,
            layer_drop_dlt,
            layer_conv_log,
            layer_bn_log,
            layer_drop_log,
        ],
    )


# -----------------------------------------------------------------------------
# suppress
# -----------------------------------------------------------------------------

MIN_SCORE = -9999.0


def nms_pad(
    bx: tf.Tensor,
    scores: tf.Tensor,
    n_nms: int,
    nms_th: float,
) -> tf.Tensor:
    """Get NMS index for a batch of images with -1 padding.

    CANNOT use `tf.image.non_max_suppression` as it outputs a tensor with
    dynamic shape

    Args:
        bx (tf.Tensor): boxes to perform NMS on. Shape [B, N, 4].
        scores (tf.Tensor): scores to perform NMS on. Shape [B, N].
        n_nms (int): number of boxes to keep after NMS.
        nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: indices of the boxes to keep after NMS. Shape [B, n_nms]
    """
    bx_dummy = tf.zeros_like(bx[:, 0:1, :])
    bx = tf.concat([bx_dummy, bx], axis=1)
    scores_dummy = tf.ones_like(scores[:, 0:1]) * MIN_SCORE
    scores = tf.concat([scores_dummy, scores], axis=1)
    selected_indices, _ = tf.image.non_max_suppression_padded(
        bx,
        scores,
        n_nms,
        iou_threshold=nms_th,
        pad_to_max_output_size=True,
    )
    return selected_indices - 1


def suppress_score(bx: tf.Tensor, log: tf.Tensor, n_score: int) -> tf.Tensor:
    """Suppression Block with score thresholding.

    It receives the RPN logits and RoIs and produces the suppressed Region of
    Interests (RoI).

    Args:
        bx (tf.Tensor): RoI bounding box. Shape [B, N_val_ac, 4].
        log (tf.Tensor): RoI logits. Shape [B, N_val_ac, 1].
        n_score (int): number of top scores to keep.

    Returns:
        tf.Tensor: proposed Region of Interests (RoIs) [B, n_score, 4]
    """
    # score thresholding
    scores = tf.squeeze(log, axis=-1)  # (B, N_val_ac)
    idx_topk = tf.math.top_k(scores, k=n_score).indices
    roi_topk = tf.gather(bx, idx_topk, batch_dims=1)  # B, n_score, 4
    return tf.reshape(roi_topk, (-1, n_score, 4))


def suppress(
    bx: tf.Tensor,
    log: tf.Tensor,
    n_score: int,
    n_nms: int,
    nms_th: float,
) -> tf.Tensor:
    """Suppression Block, including score thresholding and NMS.

    It receives the RPN logits and RoIs and produces the suppressed Region of
    Interests (RoI).

    Args:
        bx (tf.Tensor): RoI bounding box. Shape [B, N_val_ac, 4].
        log (tf.Tensor): RoI logits. Shape [B, N_val_ac, 1].
        n_score (int): number of top scores to keep.
        n_nms (int): number of RoIs to keep after NMS.
        nms_th (float): NMS threshold.

    Returns:
        tf.Tensor: proposed Region of Interests (RoIs) [B, n_nms, 4]
    """
    # score thresholding
    scores = tf.squeeze(log, axis=-1)  # (B, N_val_ac)
    idx_topk = tf.math.top_k(scores, k=n_score).indices
    score_topk = tf.gather(scores, idx_topk, batch_dims=1)  # B, n_score
    roi_topk = tf.gather(bx, idx_topk, batch_dims=1)  # B, n_score, 4
    # non-maximum suppression
    idx_nms = nms_pad(roi_topk, score_topk, n_nms, nms_th)  # (B, n_nms)
    # fetch the RoIs; -1 will result in (0., 0., 0., 0.)
    roi_nms = tf.gather(roi_topk, idx_nms, batch_dims=1)  # (B, n_nms, 4)
    # for shape inference
    return tf.reshape(roi_nms, (-1, n_nms, 4))


# =============================================================================
# SECTION: loss
# =============================================================================


def risk_rpn_reg(
    bx_prd: tf.Tensor,
    bx_tgt: tf.Tensor,
    mask: tf.Tensor,
) -> tf.Tensor:
    """Delta / RoI regression risk for RPN with smooth L1 loss.

    `bx` values cannot be `inf` or `-inf` otherwise the loss will be `nan`.

    Args:
        bx_prd (tf.Tensor): predicted box (N_ac, 4); could be predicted deltas
            or RoIs
        bx_tgt (tf.Tensor): target box (N_ac, 4); could be target deltas or
            ground truth boxes
        mask (tf.Tensor): 0/1 object mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    fn_huber = tf.keras.losses.Huber(reduction="none")
    prd = tf.boolean_mask(bx_prd, mask)  # (N_obj, 4)
    tgt = tf.boolean_mask(bx_tgt, mask)  # (N_obj, 4)
    loss = fn_huber(tgt, prd)  # (N_obj,)
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + EPS)


def risk_rpn_obj(logits: tf.Tensor, mask_obj: tf.Tensor) -> tf.Tensor:
    """Objectness classification risk for RPN with binary cross entropy loss.

    Args:
        logits (tf.Tensor): predicted logits (N_ac, 1)
        mask_obj (tf.Tensor): 0/1 object mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction="none",
    )
    prd = tf.boolean_mask(logits, mask_obj)  # (N_obj, 1)
    tgt = tf.ones_like(prd)
    loss = bce(tgt, prd)  # (N_obj,)
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask_obj) + EPS)


def risk_rpn_bkg(logits: tf.Tensor, mask_bkg: tf.Tensor) -> tf.Tensor:
    """Background classification risk for RPN with binary cross entropy loss.

    Args:
        logits (tf.Tensor): predicted logits (N_ac, 1)
        mask_bkg (tf.Tensor): 0/1 background mask (N_ac,)

    Returns:
        tf.Tensor: risk (scalar)
    """
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction="none",
    )
    prd = tf.boolean_mask(logits, mask_bkg)  # (N_bkg, 1)
    tgt = tf.zeros_like(prd)  # (N_bkg, 1)
    loss = bce(tgt, prd)  # (N_bkg,)
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask_bkg) + EPS)


def risk_rpn(
    bx_roi: tf.Tensor,
    bx_tgt: tf.Tensor,
    logits: tf.Tensor,
    mask_obj: tf.Tensor,
    mask_bkg: tf.Tensor,
) -> tf.Tensor:
    """RPN loss.

    Args:
        bx_roi (tf.Tensor): RoI boxes (B, N_ac, 4)
        bx_tgt (tf.Tensor): ground truth boxes matching RoIs (B, N_ac, 4)
        logits (tf.Tensor): predicted logits (B, N_ac, 1)
        mask_obj (tf.Tensor): 0/1 object mask (B, N_ac)
        mask_bkg (tf.Tensor): 0/1 background mask (B, N_ac)

    Returns:
        tf.Tensor: risk (scalar)
    """
    loss_reg = tf.map_fn(
        lambda x: risk_rpn_reg(x[0], x[1], x[2]),
        (bx_roi, bx_tgt, mask_obj),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )  # (B,)
    loss_obj = tf.map_fn(
        lambda x: risk_rpn_obj(x[0], x[1]),
        (logits, mask_obj),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )  # (B,)
    loss_bkg = tf.map_fn(
        lambda x: risk_rpn_bkg(x[0], x[1]),
        (logits, mask_bkg),
        fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32),
    )  # (B,)
    return tf.reduce_mean(loss_reg + loss_obj + loss_bkg)


def tf_label(
    bx_prd: tf.Tensor,
    bx_tgt: tf.Tensor,
    iou_th: float = 0.5,
) -> tf.Tensor:
    """Compute the True/False labels based on IoU for a batch of images.

    Args:
        bx_prd (tf.Tensor): predicted boxes (B, N_prd, 4)
        bx_tgt (tf.Tensor): target boxes (B, N_tgt, 4)
        iou_th (float, optional): IoU threshold. Defaults to 0.5.

    Returns:
        float: The mAP value.
    """
    ious = iou_batch(bx_prd, bx_tgt)  # (B, N_prd, N_tgt)
    iou_roi_max = tf.reduce_max(ious, axis=-1)  # (B, N_prd)
    return tf.where(iou_roi_max > iou_th, 1.0, 0.0)  # (B, N_prd)


def recall_rpn(
    bx_prd: tf.Tensor,
    bx_tgt: tf.Tensor,
    iou_th: float = 0.5,
) -> tf.Tensor:
    """Calculate recall for a batch of images."""
    ious = iou_batch(bx_prd, bx_tgt)  # (B, N_prd, N_tgt)
    max_ious = tf.reduce_max(ious, axis=1)  # (B, N_prd)
    tp = tf.reduce_sum(tf.cast(max_ious > iou_th, tf.float32), axis=1)  # (B,)
    total_gt = tf.cast(tf.shape(bx_tgt)[1], tf.float32)  # N_tgt
    return tf.reduce_mean(tp / (total_gt + EPS))


# =============================================================================
# SECTION: Training
# =============================================================================

LR_INIT = 0.01  # CANNOT be too large
LR_DECAY_FACTOR = 0.1  # Factor to reduce the learning rate
STEP_SIZE = 4  # Decay the learning rate every 'step_size' epochs


class ModelRPN(tf.keras.Model):
    """Custom Model for RPN."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.mean_loss = tf.keras.metrics.Mean(name="loss")
        self.mean_ap = tf.keras.metrics.AUC(name="meanap", curve="PR")

    def train_step(
        self,
        data: tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        """The logic for one training step."""
        x, (bx_gt, _) = data

        bsize = tf.shape(x)[0]
        # create anchors with matching batch size (B, N_ac, 4)
        # NOTE: cannot use a constant batch size as the last batch may have a
        # different size
        ac_ = tf.repeat(AC_VAL[tf.newaxis, ...], bsize, axis=0)

        with tf.GradientTape() as tape:
            # In TF2, the `training` flag affects, during both training and
            # inference, behavior of layers such as normalization (e.g. BN)
            # and dropout.
            dlt_prd, log_prd, bbx_prd = self(x, training=True)
            # NOTE: cannot use broadcasting for performance
            bx_tgt = get_gt_box(ac_, bx_gt)
            # NOTE: cannot use broadcasting for performance
            dlt_tgt = bbox2delta(bx_tgt, ac_)
            mask_obj = get_gt_mask(bx_tgt, bkg=False)
            mask_bkg = get_gt_mask(bx_tgt, bkg=True)
            loss = risk_rpn(dlt_prd, dlt_tgt, log_prd, mask_obj, mask_bkg)

        trainable_vars = self.trainable_variables

        grads = tape.gradient(loss, trainable_vars)
        # check NaN using assertions; it works both in
        # - Graph Construction Phase / Defining operations (blueprint)
        # - Session Execution Phase / Running operations (actual computation)
        grads = [
            tf.debugging.assert_all_finite(
                g, message="NaN/Inf gradient detected.") for g in grads
        ]
        # clip gradient
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(
            zip(  # noqa: B905
                grads,
                trainable_vars,
            ))

        self.mean_loss.update_state(loss)
        label = tf_label(bbx_prd, bx_tgt, iou_th=0.5)
        self.mean_ap.update_state(label, log_prd)

        return {
            "loss": self.mean_loss.result(),
            "meanap": self.mean_ap.result(),
        }

    @property
    def metrics(self) -> list[tf.keras.metrics.Metric]:
        """List of the model's metrics.

        We list our `Metric` objects here so that `reset_states()` can be
        called automatically at the start of each epoch
        or at the start of `evaluate()`.
        If you don't implement this property, you have to call
        `reset_states()` yourself at the time of your choosing.
        """
        return [self.mean_loss, self.mean_ap]

    def test_step(
        self,
        data: tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]],
    ) -> dict[str, tf.Tensor]:
        """Logic for one evaluation step."""
        x, (bx_gt, _) = data

        dlt_prd, log_prd, bbx_prd = self(x, training=False)
        label = tf_label(bbx_prd, bx_gt, iou_th=0.5)
        self.mean_ap.update_state(label, log_prd)

        return {
            "meanap": self.mean_ap.result(),
        }


def get_rpn_model(
    *,
    freeze_backbone: bool = False,
    freeze_rpn: bool = False,
) -> ModelRPN:
    """Create a RPN model for training or prediction."""
    # Backbone
    bb = tf.keras.applications.resnet50.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(H, W, 3),
    )

    # Add RPN layers on top of the backbone
    tmp_dlt, tmp_log, rpn_layers = rpn(bb.output, H_FM, W_FM)
    bbx = roi(tmp_dlt)

    # Freeze all layers in the backbone for training
    if freeze_backbone:
        for layer in bb.layers:
            layer.trainable = False
    # Freeze all layers in the RPN for training
    if freeze_rpn:
        for layer in rpn_layers:
            layer.trainable = False

    # Create the ModelRPN instance
    model = ModelRPN(
        inputs=bb.input,
        outputs=[tmp_dlt, tmp_log, bbx],
    )

    return model


# # Step Decay Learning Rate Scheduler
# def lr_schedule(epoch: int) -> float:
#     """Learning rate schedule."""
#     return LR_INIT * (LR_DECAY_FACTOR**(epoch // STEP_SIZE))
# # Callback for updating learning rate
# cb_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Callbacks
PATH_CKPT_RPN = "model_weights/rpn/ckpt"
PATH_CKPT_FINETUNE = "model_weights/rpn_finetune/ckpt"


def get_cb_ckpt(path: str) -> tf.keras.callbacks.ModelCheckpoint:
    """Get callback for saving model weights."""
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=path,
        monitor="val_meanap",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )


cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_meanap",
    patience=30,
    mode="max",
    restore_best_weights=True,
)
cb_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_meanap",
    mode="max",
    factor=0.2,
    patience=2,
    cooldown=4,
    min_lr=1e-8,
    min_delta=0.001,
    verbose=1,
)

# # Adam Optimizer
# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.0001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-08,
# )
# SGD Optimizer with Momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=LR_INIT, momentum=0.9)

ds_tr, _ = load_train_voc2007(BATCH_SIZE_TR)
ds_va, _ = load_test_voc2007(BATCH_SIZE_TE)

# training
model = get_rpn_model(freeze_backbone=False, freeze_rpn=False)
model.compile(optimizer=optimizer)
model.fit(
    ds_tr,
    epochs=100,
    validation_data=ds_va,
    callbacks=[get_cb_ckpt(PATH_CKPT_RPN), cb_earlystop, cb_lr],
)

# # fine-tuning
# model = get_rpn_model(freeze_backbone=False, freeze_rpn=False)
# model.load_weights(PATH_CKPT_RPN)
# model.compile(optimizer=optimizer)
# model.fit(
#     ds_tr,
#     epochs=100,
#     validation_data=ds_va,
#     callbacks=[get_cb_ckpt(PATH_CKPT_FINETUNE), cb_earlystop, cb_lr],
# )
