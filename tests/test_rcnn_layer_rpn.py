"""Test the RPN model."""
import tensorflow as tf

from rcnn import anchor
from rcnn.model._rpn import rpn

BS = 2  # batch size
H_FM = 16  # height of the feature map
W_FM = 16  # width of the feature map
C_FM = 256  # number of channels in the feature map


def setup_model() -> tf.keras.Model:
    """Set up the model."""
    inputs = tf.keras.Input(shape=(H_FM, W_FM, C_FM))
    outputs = rpn(inputs, H_FM, W_FM)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_rpt_training() -> None:
    """Test the RPN model training."""
    inputs = tf.random.uniform((BS, H_FM, W_FM, C_FM))
    dlt_tg = tf.random.uniform((BS, anchor.N_RPNAC, 4))
    lbl_tg = tf.random.uniform((BS, anchor.N_RPNAC, 1))
    mdl = setup_model()
    mdl.compile(optimizer="adam", loss="mse")
    history = mdl.fit(inputs, [dlt_tg, lbl_tg], epochs=1)
    assert history.history["loss"][0] > 0


def test_rpt_predict() -> None:
    """Test the RPN model output."""
    mdl = setup_model()
    inputs = tf.random.uniform((BS, H_FM, W_FM, C_FM))
    dlt, lbl = mdl(inputs)
    assert dlt.shape == (BS, anchor.N_RPNAC, 4)
    assert lbl.shape == (BS, anchor.N_RPNAC, 1)
