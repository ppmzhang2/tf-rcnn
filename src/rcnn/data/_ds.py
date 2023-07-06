"""Preprocesses the Pascal VOC dataset for training."""
import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn import cfg


def ratio_resize(
    img: tf.Tensor,
    bbx: tf.Tensor,
    h_img: int,
    w_img: int,
    *,
    relative_coord: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Resize an image and bounding boxes while preserving its aspect ratio.

    Args:
        img (tf.Tensor): image to resize
        bbx (tf.Tensor): bounding boxes to resize
        h_img (int): desired height of the image
        w_img (int): desired width of the image
        relative_coord (bool, optional): whether the bounding boxes are in
            relative coordinates. Defaults to True.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: resized image and bounding boxes
    """
    h, w, _ = tf.unstack(tf.shape(img))
    scale_h = tf.cast(h_img / h, tf.float32)
    scale_w = tf.cast(w_img / w, tf.float32)
    scale = tf.minimum(scale_h, scale_w)
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * scale), tf.int32)
    img = tf.image.resize(img, [new_h, new_w])
    if not relative_coord:
        bbx = bbx * scale
    return img, bbx


def image_pad(
    img: tf.Tensor,
    bbx: tf.Tensor,
    h_img: int,
    w_img: int,
    *,
    relative_coord: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Pads an image and bounding boxes to a fixed size.

    Args:
        img (tf.Tensor): image to pad
        bbx (tf.Tensor): bounding boxes to pad
        h_img (int): desired height of the image
        w_img (int): desired width of the image
        relative_coord (bool, optional): whether the bounding boxes are in
            relative coordinates. Defaults to True.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: padded image and bounding boxes
    """
    image_h, image_w, _ = tf.unstack(tf.shape(img))
    pad_height = h_img - image_h
    pad_width = w_img - image_w
    img = tf.pad(img, paddings=[[0, pad_height], [0, pad_width], [0, 0]])
    if relative_coord:
        abs_boxes = bbx * tf.cast(
            tf.stack([image_h, image_w, image_h, image_w]),
            tf.float32,
        )
        bbx = abs_boxes / tf.cast(
            tf.stack([cfg.H, cfg.W, cfg.H, cfg.W]),
            tf.float32,
        )
    return img, bbx


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


def ds_handler(sample: dict) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
        tuple[tf.Tensor, tf.Tensor, tf.Tensor]: images, bounding boxes, and
            labels
            - images: [batch_size, H, W, 3]
            - bounding boxes: [batch_size, max_box, 4] in relative coordinates
            - labels: [batch_size, max_box, 1]
    """
    img = sample["image"]
    img = tf.cast(img, tf.float32) / 255.0

    # Get the bounding boxes and labels
    bbx = sample["objects"]["bbox"]
    lbl = sample["objects"]["label"]

    # resize the image and bounding boxes while preserving the aspect ratio
    img, bbx = ratio_resize(img, bbx, cfg.H, cfg.W)
    # Pad to a fixed size
    img, bbx = image_pad(img, bbx, cfg.H, cfg.W)

    # pad the labels and bounding boxes to a fixed size
    bbx = batch_pad(bbx, max_box=cfg.MAX_BOX, value=0)
    lbl = batch_pad(lbl[:, tf.newaxis], max_box=cfg.MAX_BOX, value=-1)

    return img, bbx, lbl


def load_train_valid(
    name: str,
    n_tr: int,
    n_te: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads the training and validation datasets.

    Args:
        name (str): name of the dataset
        n_tr (int): number of training samples
        n_te (int): number of testing samples

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
            training and validation datasets, and dataset info.

    The training and validation datasets are shuffled and preprocessed with
    `ds_handler`:
        - images: [batch_size, H, W, 3]
        - bounding boxes: [batch_size, max_box, 4] in relative
        - labels: [batch_size, max_box, 1]
    """
    (ds_tr, ds_va), ds_info = tfds.load(
        name,
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
    )
    ds_tr = ds_tr.map(
        ds_handler,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(n_tr).prefetch(tf.data.experimental.AUTOTUNE)
    ds_va = ds_va.map(
        ds_handler,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(n_te).prefetch(tf.data.experimental.AUTOTUNE)
    return ds_tr, ds_va, ds_info


def load_test(
    name: str,
    n_te: int,
    *,
    shuffle: bool = False,
) -> tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
    """Loads the testing dataset.

    Args:
        name (str): name of the dataset
        n_te (int): number of testing samples
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to
            False.

    Returns:
        tuple[tf.data.Dataset, tfds.core.DatasetInfo]: testing dataset and
            dataset info.

    The testing dataset is preprocessed with `ds_handler`:
        - images: [batch_size, H, W, 3]
        - bounding boxes: [batch_size, max_box, 4] in relative
        - labels: [batch_size, max_box, 1]
    """
    ds_te, ds_info = tfds.load(
        name,
        split="validation",
        shuffle_files=shuffle,
        with_info=True,
    )
    ds_te = ds_te.map(
        ds_handler,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).batch(n_te).prefetch(tf.data.experimental.AUTOTUNE)
    return ds_te, ds_info
