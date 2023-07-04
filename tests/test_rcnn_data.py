"""Test the RPN model."""
import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn import cfg
from rcnn.data import process_data


def setup_io() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Set up the inputs / outputs."""
    return tfds.load("voc/2007", split="train", shuffle_files=True)


def test_data_process_data() -> None:
    """Test the data processing."""
    ds = setup_io()
    for example in ds.take(1):
        img, bx, lbl = process_data(example)
        assert img.shape == (cfg.H, cfg.W, 3)
        assert bx.shape == (cfg.MAX_BOX, 4)
        assert lbl.shape == (cfg.MAX_BOX, 1)
