"""Test the RPN model."""
import tensorflow as tf
import tensorflow_datasets as tfds

from src.rcnn import cfg
from src.rcnn.data import process_data

B = 16


def setup_io() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Set up the inputs / outputs."""
    ds = tfds.load("voc/2007", split="train", shuffle_files=True)
    ds = ds.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(B).prefetch(tf.data.AUTOTUNE)
    return ds


def test_data_process_data() -> None:
    """Test the data processing."""
    ds = setup_io()
    for img, bx, lbl in ds:
        assert img.shape == (B, cfg.H, cfg.W, 3)
        assert bx.shape == (B, cfg.MAX_BOX, 4)
        assert lbl.shape == (B, cfg.MAX_BOX, 1)
        break
