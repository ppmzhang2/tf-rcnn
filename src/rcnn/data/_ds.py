"""Preprocesses the Pascal VOC dataset for training."""
import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn import cfg

SIZE_RESIZE = 600  # image size for resizing


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


def preprcs_tr(sample: dict) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Preprocesses a sample from the training dataset.

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
    # Normalize the image with ImageNet mean and std
    img = (tf.cast(img, dtype=tf.float32) - cfg.IMGNET_MEAN) / cfg.IMGNET_STD

    # Get the bounding boxes and labels
    bbx = sample["objects"]["bbox"]
    lbl = sample["objects"]["label"]

    # resize the image and bounding boxes while preserving the aspect ratio
    img = resize(img, cfg.SIZE_RESIZE)
    # randomly crop the image and bounding boxes
    img, bbx = rand_crop(img, bbx, cfg.SIZE_IMG, cfg.SIZE_IMG)
    # randomly augment the image and bounding boxes
    img, bbx = data_augment(img, bbx)

    # pad the labels and bounding boxes to a fixed size
    bbx = batch_pad(bbx, max_box=cfg.N_OBJ, value=0)
    lbl = batch_pad(lbl[:, tf.newaxis], max_box=cfg.N_OBJ, value=-1)

    return img, (bbx, lbl)


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
    img = (tf.cast(img, dtype=tf.float32) - cfg.IMGNET_MEAN) / cfg.IMGNET_STD

    # Get the bounding boxes and labels
    bbx = sample["objects"]["bbox"]
    lbl = sample["objects"]["label"]

    # resize the image and bounding boxes while preserving the aspect ratio
    img = resize(img, cfg.SIZE_RESIZE)
    # randomly crop the image and bounding boxes
    img, bbx = rand_crop(img, bbx, cfg.SIZE_IMG, cfg.SIZE_IMG)

    # pad the labels and bounding boxes to a fixed size
    bbx = batch_pad(bbx, max_box=cfg.N_OBJ, value=0)
    lbl = batch_pad(lbl[:, tf.newaxis], max_box=cfg.N_OBJ, value=-1)

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
    ).shuffle(cfg.BUFFER_SIZE).batch(n).prefetch(tf.data.experimental.AUTOTUNE)
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
