"""Test the RPN model."""
import tensorflow as tf
from tensorflow.keras.models import Model

from rcnn import anchor
from rcnn import cfg
from rcnn.model import get_rpn_model


def setup_model() -> Model:
    """Set up the model."""
    # model = VGG16 + RPN
    mdl_rpn = get_rpn_model()
    return mdl_rpn


def setup_io() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Set up the inputs / outputs."""
    # batch size, must be same as cfg.BATCH_SIZE_TR
    # TODO: customizable batch size
    b = 4
    inputs = tf.random.uniform((b, cfg.H, cfg.W, cfg.C))
    rois = tf.random.uniform((b, cfg.N_SUPP_NMS, 4))  # (B, N_rois, 4)
    logits = tf.random.uniform((b, anchor.N_RPNAC, 1))
    deltas = tf.random.uniform((b, anchor.N_RPNAC, 4))
    boxes = tf.random.uniform((b, anchor.N_RPNAC, 4))
    return inputs, rois, logits, deltas, boxes


def test_rpt_training() -> None:
    """Test the RPN model training."""
    inputs, rois, logits, deltas, boxes = setup_io()
    model = setup_model()
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x=inputs, y=[deltas, logits], epochs=1)
    assert history.history["loss"][0] > 0


def test_rpt_predict() -> None:
    """Test the RPN model output."""
    model = setup_model()
    inputs, rois, logits, deltas, boxes = setup_io()
    output = model(inputs)
    assert output[0].shape == deltas.shape
    assert output[1].shape == logits.shape
    assert output[2].shape == rois.shape
