"""Test the RPN model."""
import tensorflow as tf

from rcnn import anchor
from rcnn.model._roi import roi

BS = 2  # batch size


def test_roi_infer() -> None:
    """Test the shape inference of the RoI layer."""
    inputs = tf.keras.layers.Input(shape=(anchor.N_RPNAC, 4))
    outputs = roi(inputs)
    assert outputs.shape == (None, anchor.N_RPNAC, 4)


def test_roi_predict() -> None:
    """Test the RoI layer output."""
    inputs = tf.keras.layers.Input(shape=(anchor.N_RPNAC, 4))
    outputs = roi(inputs)
    mdl = tf.keras.Model(inputs=inputs, outputs=outputs)
    dlt = tf.random.uniform((BS, anchor.N_RPNAC, 4))
    rois = mdl(dlt)
    assert rois.shape == (BS, anchor.N_RPNAC, 4)
