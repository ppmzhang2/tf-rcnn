"""Test the RPN model."""
import tensorflow as tf
from tensorflow.keras.models import Model

from rcnn import cfg
from rcnn.model import get_rpn_model


def setup_model() -> Model:
    """Set up the model."""
    # model = VGG16 + RPN
    mdl_rpn = get_rpn_model()
    mdl_rpn.compile(optimizer="adam", loss="mse")
    return mdl_rpn


def setup_io() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Set up the inputs / outputs."""
    b = 16  # batch size
    inputs = tf.random.uniform((b, cfg.H, cfg.W, cfg.C))
    rois = tf.random.uniform((b, 10, 4))  # (B, N_rois, 4)
    boxes = tf.random.uniform((b, cfg.H_FM * cfg.W_FM * cfg.N_ANCHOR, 4))
    labels = tf.random.uniform((b, cfg.H_FM * cfg.W_FM * cfg.N_ANCHOR, 1))
    return inputs, rois, boxes, labels


def test_rpt_training() -> None:
    """Test the RPN model training."""
    model = setup_model()
    inputs, rois, boxes, labels = setup_io()
    history = model.fit(inputs, [rois, boxes, labels], epochs=1)
    assert history.history["loss"][0] > 0


def test_rpt_predict() -> None:
    """Test the RPN model output."""
    model = setup_model()
    inputs, rois, boxes, labels = setup_io()
    output = model.predict(inputs)
    assert output[0].shape == rois.shape
    assert output[1].shape == boxes.shape
    assert output[2].shape == labels.shape
