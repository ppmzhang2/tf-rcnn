"""Test the `data._ds` model."""
import tensorflow as tf
import tensorflow_datasets as tfds

from rcnn import cfg
from rcnn.data._ds import preprcs_tr


def setup_io() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Set up the inputs / outputs."""
    return tfds.load("voc/2007", split="train", shuffle_files=True)


def test_data_ds_handler() -> None:
    """Test the `data.ds_handler` function."""
    ds = setup_io()
    for example in ds.take(1):
        img, (bx, lbl) = preprcs_tr(example)
        assert img.shape == (cfg.H, cfg.W, 3)
        assert bx.shape == (cfg.N_OBJ, 4)
        assert lbl.shape == (cfg.N_OBJ, 1)
