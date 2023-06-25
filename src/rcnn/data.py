"""Preprocesses the Pascal VOC dataset for training."""
import tensorflow as tf
import tensorflow_datasets as tfds

# Set the constants for preprocessing
HEIGHT = 416
WIDTH = 416
MAX_BOX = 20


def ratio_resize(
    img: tf.Tensor,
    bbx: tf.Tensor,
    *,
    relative_coord: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Resize an image and bounding boxes while preserving its aspect ratio.

    Args:
        img (tf.Tensor): image to resize
        bbx (tf.Tensor): bounding boxes to resize
        relative_coord (bool, optional): whether the bounding boxes are in
            relative coordinates. Defaults to True.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: resized image and bounding boxes
    """
    h, w, _ = tf.unstack(tf.shape(img))
    scale_h = tf.cast(HEIGHT / h, tf.float32)
    scale_w = tf.cast(WIDTH / w, tf.float32)
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
    *,
    relative_coord: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Pads an image and bounding boxes to a fixed size.

    Args:
        img (tf.Tensor): image to pad
        bbx (tf.Tensor): bounding boxes to pad
        relative_coord (bool, optional): whether the bounding boxes are in
            relative coordinates. Defaults to True.

    Returns:
        tuple[tf.Tensor, tf.Tensor]: padded image and bounding boxes
    """
    image_h, image_w, _ = tf.unstack(tf.shape(img))
    pad_height = HEIGHT - image_h
    pad_width = WIDTH - image_w
    img = tf.pad(img, paddings=[[0, pad_height], [0, pad_width], [0, 0]])
    if relative_coord:
        abs_boxes = bbx * tf.cast(
            tf.stack([image_h, image_w, image_h, image_w]),
            tf.float32,
        )
        bbx = abs_boxes / tf.cast(
            tf.stack([HEIGHT, WIDTH, HEIGHT, WIDTH]),
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


def process_data(sample: dict) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Prepares a sample for training."""
    img = sample["image"]
    img = tf.cast(img, tf.float32) / 255.0

    # Get the bounding boxes and labels
    bbx = sample["objects"]["bbox"]
    lbl = sample["objects"]["label"]

    # resize the image and bounding boxes while preserving the aspect ratio
    img, bbx = ratio_resize(img, bbx)
    # Pad to a fixed size
    img, bbx = image_pad(img, bbx)

    # pad the labels and bounding boxes to a fixed size
    bbx = batch_pad(bbx, max_box=MAX_BOX, value=0)
    lbl = batch_pad(lbl[:, tf.newaxis], max_box=MAX_BOX, value=-1)

    return img, bbx, lbl


if __name__ == "__main__":
    # Load the dataset
    ds = tfds.load("voc/2007", split="train", shuffle_files=True)

    # Prepare the dataset for training
    ds = ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

    # Inspect the shapes
    for images, boxes, labels in ds.take(1):
        print("Images shape:", images.shape)
        print("Boxes shape:", boxes.shape)
        print("Labels shape:", labels.shape)
