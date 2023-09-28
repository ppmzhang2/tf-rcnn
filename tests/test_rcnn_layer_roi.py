"""Test the RPN model."""
import tensorflow as tf

from rcnn.model._roi import N_VAL_AC
from rcnn.model._roi import roi

BS = 2  # batch size
H_FM = 16  # height of the feature map
W_FM = 16  # width of the feature map
C_FM = 256  # number of channels in the feature map
N_ANCHOR = 9  # number of anchors


def setup_inputs() -> tf.keras.Model:
    """Set up the model."""
    log = tf.keras.layers.Input(shape=(H_FM * W_FM * N_ANCHOR, 1))
    dlt = tf.keras.layers.Input(shape=(H_FM * W_FM * N_ANCHOR, 4))
    return log, dlt


def test_roi_infer() -> None:
    """Test the shape inference of the RoI layer."""
    inputs = setup_inputs()
    outputs = roi(inputs[0], inputs[1])
    assert outputs[0].shape == (None, N_VAL_AC, 1)
    assert outputs[1].shape == (None, N_VAL_AC, 4)
    assert outputs[2].shape == (None, N_VAL_AC, 4)


def test_roi_predict() -> None:
    """Test the RoI layer output."""
    inputs = setup_inputs()
    outputs = roi(inputs[0], inputs[1])
    mdl = tf.keras.Model(inputs=inputs, outputs=outputs)
    log = tf.random.uniform((BS, H_FM * W_FM * N_ANCHOR, 1))
    dlt = tf.random.uniform((BS, H_FM * W_FM * N_ANCHOR, 4))
    log_val, dlt_val, rois = mdl([log, dlt])
    assert log_val.shape == (BS, N_VAL_AC, 1)
    assert dlt_val.shape == (BS, N_VAL_AC, 4)
    assert rois.shape == (BS, N_VAL_AC, 4)
