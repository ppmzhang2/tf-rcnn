"""Data Loader."""
import tensorflow as tf
import tensorflow_datasets as tfds

N_OBJECT = 100  # maximum number of objects per image


def ds_handler(sample: dict) -> tuple[tf.Tensor, tuple[tf.Tensor, tf.Tensor]]:
    """Preprocess a single sample.

    Args:
        sample: A dict mapping sample keys to tensors. e.g.:
            {
                "image": tf.Tensor,
                "objects": {
                    "bbox": tf.Tensor,
                    "label": tf.Tensor
                }
            }

    Returns:
        tuple: A tuple of (image, (bbox, label)). e.g.:

            (
                tf.Tensor,  # image
                (
                    tf.Tensor,  # bbox
                    tf.Tensor,  # label
                )
            )
    """
    image = tf.image.resize(sample["image"], (224, 224))
    image = image / 255.0  # normalize to [0,1] range
    objects = sample["objects"]

    bbox = objects["bbox"]
    labels = objects["label"]

    # pad to fixed number of objects
    bbox_ = tf.pad(
        bbox,
        paddings=[[0, N_OBJECT - tf.shape(bbox)[0]], [0, 0]],
        mode="CONSTANT",
        constant_values=-1,
    )
    labels_ = tf.pad(
        labels,
        paddings=[[0, N_OBJECT - tf.shape(labels)[0]]],
        mode="CONSTANT",
        constant_values=-1,
    )

    return image, (bbox_, labels_)


if __name__ == "__main__":
    (ds_train, ds_test), ds_info = tfds.load(
        "voc/2007",
        split=["train", "validation"],
        shuffle_files=True,
        with_info=True,
    )
    ds_train = ds_train.map(ds_handler)
    # print(ds_train.element_spec[0])
    # print(ds_train.element_spec[1])

    for image, (bbox, label) in ds_train.take(1):
        print("Image shape:", image.shape)
        print("Bounding box:", bbox.numpy())
        print("Label:", label.numpy())
